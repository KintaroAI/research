// vm_cuda.cu — the basic agent's update on the GPU (task 00011). See vm_cuda.h.
//
// Mirrors vm_agent.cpp (CNN path, prediction-only losses) with all S*K frames of a batch processed together:
// convolutions are im2col + one GEMM per layer, transposed convolutions GEMM + col2im, linear layers cuBLAS GEMMs
// (row-major wrappers, as in llm-fundamentals' GPT-2), the rest custom kernels. fp32 throughout (TF32 off), so the
// gradients match the C++ model's to float precision; sums run in another order and use atomics, so the results are
// not bitwise reproducible.
//
// Layouts (b = s*K + t is the frame index): encoder maps frame-major [B, C, H, W]; the decoder inputs, the decoder
// conv outputs and the predicted frame channel-major [C, B, H, W], so that each (transposed) convolution is one GEMM
// over all frames; token rows [B, width].
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "vm_cuda.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "vm_cuda: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "vm_cuda: cublas error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); return -1; } } while (0)

// The parameter tensors, in vm_agent.cpp's order (the flat layout is shared).
enum ParamId {
    C1W, C1B, C2W, C2B, C3W, C3B, EFW, EFB, INW, INB, WPE,
    LN1W, LN1B, QKVW, QKVB, APW, APB, LN2W, LN2B, FCW, FCB, FPW, FPB, LNFW, LNFB,
    PIW, PIB, PHW, PHB, LOGSTD, VW, VB,
    PR1W, PR1B, PR2W, PR2B, PBW, PBB, D1W, D1B, D2W, D2B, DRW, DRB, IV1W, IV1B, IV2W, IV2B,
    T1W, T1B, T2W, T2B, H1W, H1B, H2W, H2B, X1W, X1B, X2W, X2B,
    PEW, PEB, POS2D, TYP, DEW, DEB, MEW, MEB, CEW, CEB, NUM_P
};
static const float LS_MIN = -5.0f, LS_MAX = 1.0f, LOG_2PI = 1.8378770664093453f;
static const int KK = 16;   // 4x4 kernels, stride 2, padding 1

struct Dims {
    int img, in_ch, c1, c2, c3, h1, h2, h3, enc_flat, feat, d, L, nh, hs, dff, K, in_dim;
    int pimg, p0, p1, pc1, pc2, ph, pred_flat, pin_dim, small_px, pb, d1_in, d2_in, skip1_off, skip_off, pih;
};

static bool make_dims(const VMConfig& c, Dims& D) {
    if (!c.coord || !c.pred_residual || !c.pred_skip || c.inv_hidden > 0 || c.dyn_hidden > 0 || c.patch > 0) return false;
    if (c.img % 8 || c.pred_img % 4 || c.img / c.pred_img != 2 || c.d_model % c.n_head || c.pb <= 0) return false;
    D.img = c.img; D.in_ch = 5; D.c1 = c.c1; D.c2 = c.c2; D.c3 = c.c3;
    D.h1 = c.img / 2; D.h2 = c.img / 4; D.h3 = c.img / 8; D.enc_flat = c.c3 * D.h3 * D.h3;
    D.feat = c.feat; D.d = c.d_model; D.L = c.n_layer; D.nh = c.n_head; D.hs = c.d_model / c.n_head;
    D.dff = c.d_ff; D.K = c.ctx; D.in_dim = c.feat + 6;
    D.pimg = c.pred_img; D.p0 = c.pred_img / 4; D.p1 = c.pred_img / 2; D.pc1 = c.pc1; D.pc2 = c.pc2;
    D.ph = c.pred_hidden; D.pred_flat = c.pc2 * D.p0 * D.p0; D.pin_dim = c.d_model + 3;
    D.small_px = c.pred_img * c.pred_img * 3; D.pb = c.pb;
    D.skip1_off = D.pc2 + D.pb + 2; D.d1_in = D.skip1_off + D.c3 + 3;
    D.skip_off = D.pc1 + 2; D.d2_in = D.skip_off + D.c2 + 3;
    D.pih = c.pi_hidden > 0 ? c.pi_hidden : 0;
    return D.p1 == D.h2 && D.p0 == D.h3 && D.K <= 64;   // U-Net skips need equal map sizes; k_attn_bwd: K <= 64
}

static void expected_sizes(const Dims& D, long* s) {
    for (int i = 0; i < NUM_P; i++) s[i] = 0;
    const long d = D.d, L = D.L;
    s[C1W] = (long)D.c1 * D.in_ch * KK; s[C1B] = D.c1; s[C2W] = (long)D.c2 * D.c1 * KK; s[C2B] = D.c2;
    s[C3W] = (long)D.c3 * D.c2 * KK; s[C3B] = D.c3; s[EFW] = (long)D.feat * D.enc_flat; s[EFB] = D.feat;
    s[INW] = d * D.in_dim; s[INB] = d; s[WPE] = (long)D.K * d;
    s[LN1W] = L * d; s[LN1B] = L * d; s[QKVW] = L * 3 * d * d; s[QKVB] = L * 3 * d;
    s[APW] = L * d * d; s[APB] = L * d; s[LN2W] = L * d; s[LN2B] = L * d;
    s[FCW] = L * D.dff * d; s[FCB] = L * D.dff; s[FPW] = L * d * D.dff; s[FPB] = L * d; s[LNFW] = d; s[LNFB] = d;
    s[PIW] = 3L * (D.pih ? D.pih : d); s[PIB] = 3; s[PHW] = (long)D.pih * d; s[PHB] = D.pih;
    s[LOGSTD] = 2; s[VW] = d; s[VB] = 1;
    s[PR1W] = (long)D.ph * D.pin_dim; s[PR1B] = D.ph; s[PR2W] = (long)D.pred_flat * D.ph; s[PR2B] = D.pred_flat;
    s[PBW] = (long)D.pb * D.ph; s[PBB] = D.pb;
    s[D1W] = (long)D.d1_in * D.pc1 * KK; s[D1B] = D.pc1; s[D2W] = (long)D.d2_in * 3 * KK; s[D2B] = 3;
    s[DRW] = 2L * D.ph; s[DRB] = 2;
}

// ---------------------------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------------------------
#define GRID(n) (unsigned)(((n) + 255) / 256), 256
__device__ inline float relu_(float x) { return x > 0.f ? x : 0.f; }
__device__ inline float sigmoid_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float softplus_(float x) { return (x > 0.f ? x : 0.f) + log1pf(expf(-fabsf(x))); }
__device__ inline float clampf_(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
__device__ inline float coord_(int i, int S) { return (i + 0.5f) / S * 2.0f - 1.0f; }

// Box-average downsample by 2 (rounding half up): u8 [B, S, S, 3] -> [B, S/2, S/2, 3]
__global__ void k_downsample(const uint8_t* img, uint8_t* out, int B, int S) {
    const int So = S / 2;
    const long n = (long)B * So * So * 3;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int c = i % 3, x = (i / 3) % So, y = (i / (3 * So)) % So;
        const long b = i / (3L * So * So);
        const uint8_t* p = img + ((b * S + 2 * y) * S + 2 * x) * 3 + c;
        const int s = p[0] + p[3] + p[S * 3] + p[S * 3 + 3];
        out[i] = (uint8_t)((s + 2) / 4);
    }
}
// x0 [B, 5, S, S]: RGB / 255, then the x and y coordinate channels
__global__ void k_input(const uint8_t* img, float* x0, int B, int S) {
    const long n = (long)B * 5 * S * S;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int x = i % S, y = (i / S) % S, c = (i / ((long)S * S)) % 5;
        const long b = i / (5L * S * S);
        float v;
        if (c < 3) v = img[((b * S + y) * S + x) * 3 + c] / 255.0f;
        else v = c == 3 ? coord_(x, S) : coord_(y, S);
        x0[i] = v;
    }
}
// cols[(c*16+ky*4+kx), b*h*h + oy*h + ox] = src(b, c, 2oy-1+ky, 2ox-1+kx), src of side H = 2h at
// src[b*sb + c*sc + y*H + x]
__global__ void k_im2col(const float* src, long sb, long sc, int C, int H, int B, float* cols) {
    const int h = H / 2, P = h * h;
    const long BP = (long)B * P, n = (long)C * KK * BP;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const long col = i % BP;
        const int row = i / BP, c = row / KK, ky = (row % KK) / 4, kx = row % 4;
        const long b = col / P;
        const int p = col % P, oy = p / h, ox = p % h, iy = 2 * oy - 1 + ky, ix = 2 * ox - 1 + kx;
        cols[i] = (iy >= 0 && iy < H && ix >= 0 && ix < H) ? src[b * sb + c * sc + (long)iy * H + ix] : 0.f;
    }
}
// The adjoint of k_im2col as a gather: dst(b, c, Y, X) = [bias[c] +] sum of the cols entries that k_im2col would
// have read from (Y, X); relu optional. dst at dst[b*db + c*dc + Y*H + X], side H = 2h.
__global__ void k_col2im(const float* cols, int C, int H, int B, float* dst, long db, long dc, const float* bias,
                         int relu) {
    const int h = H / 2, P = h * h;
    const long BP = (long)B * P, n = (long)B * C * H * H;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int X = i % H, Y = (i / H) % H, c = (i / ((long)H * H)) % C;
        const long b = i / ((long)C * H * H);
        float s = bias ? bias[c] : 0.f;
        for (int ky = (Y + 1) & 1; ky < 4; ky += 2) {
            const int oy = (Y + 1 - ky) / 2;
            if (oy < 0 || oy >= h) continue;
            for (int kx = (X + 1) & 1; kx < 4; kx += 2) {
                const int ox = (X + 1 - kx) / 2;
                if (ox < 0 || ox >= h) continue;
                s += cols[(long)(c * KK + ky * 4 + kx) * BP + b * P + oy * h + ox];
            }
        }
        dst[b * db + c * dc + (long)Y * H + X] = relu ? relu_(s) : s;
    }
}
// a[b, c, p] = relu(Y[c, b*P + p] + bias[c])
__global__ void k_conv_out(const float* Y, const float* bias, float* a, int C, int B, int P) {
    const long n = (long)B * C * P;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int p = i % P, c = (i / P) % C;
        const long b = i / ((long)C * P);
        a[i] = relu_(Y[(long)c * B * P + b * P + p] + bias[c]);
    }
}
// dY[c, b*P + p] = a[b,c,p] > 0 ? da[b,c,p] + scale * extra(b,c,p) : 0 (the gradient at a conv's pre-activation,
// channel-major); extra at extra[b*eb + c*ec + p] (nullable)
__global__ void k_conv_grad(const float* da, const float* extra, long eb, long ec, float scale, const float* a,
                            float* dY, int C, int B, int P) {
    const long n = (long)B * C * P;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int p = i % P, c = (i / P) % C;
        const long b = i / ((long)C * P);
        float g = da[i];
        if (extra) g += scale * extra[b * eb + c * ec + p];
        dY[(long)c * B * P + b * P + p] = a[i] > 0.f ? g : 0.f;
    }
}
// out[r] += sum_j M[r, j] (one block per row)
__global__ void k_rowsum(const float* M, long N, float* out) {
    __shared__ float sh[256];
    const float* row = M + blockIdx.x * N;
    float s = 0.f;
    for (long j = threadIdx.x; j < N; j += blockDim.x) s += row[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] += sh[0];
}
// db[o] += sum_t dout[t, o]
__global__ void k_colsum(const float* dout, int T, int OC, float* db) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OC) return;
    float s = 0.f;
    for (int t = 0; t < T; t++) s += dout[(long)t * OC + o];
    db[o] += s;
}
// out[t, o] = act(out[t, o] + b[o]); act: 0 none, 1 relu, 2 sigmoid
__global__ void k_bias_act(float* out, const float* b, long T, int OC, int act) {
    const long n = T * OC;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        float v = out[i] + b[i % OC];
        out[i] = act == 1 ? relu_(v) : (act == 2 ? sigmoid_(v) : v);
    }
}
__global__ void k_token(const float* vf, const float* drv, const float* pa, const float* pr, float* tin, int B,
                        int F, int use_pr) {
    const int D = F + 6;
    const long n = (long)B * D;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % D;
        const long b = i / D;
        float v;
        if (j < F) v = vf[b * F + j];
        else if (j < F + 2) v = drv[b * 2 + j - F];
        else if (j < F + 5) v = pa[b * 3 + j - F - 2];
        else v = use_pr ? pr[b] : 0.f;
        tin[i] = v;
    }
}
__global__ void k_bias_wpe(float* x, const float* bias, const float* wpe, const int* pos, int B, int d) {
    const long n = (long)B * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % d;
        x[i] += bias[j] + wpe[(long)pos[i / d] * d + j];
    }
}
// LayerNorm, one warp per row
__global__ void k_ln_fwd(float* out, float* mean, float* rstd, const float* inp, const float* w, const float* b,
                         int T, int C) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= T) return;
    const float* x = inp + (long)row * C;
    float s = 0.f;
    for (int i = lane; i < C; i += 32) s += x[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    const float m = s / C;
    float v = 0.f;
    for (int i = lane; i < C; i += 32) { float xs = x[i] - m; v += xs * xs; }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const float r = 1.0f / sqrtf(v / C + 1e-5f);
    float* y = out + (long)row * C;
    for (int i = lane; i < C; i += 32) y[i] = (r * (x[i] - m)) * w[i] + b[i];
    if (lane == 0) { mean[row] = m; rstd[row] = r; }
}
// dinp += ; dw += ; db += (atomics over rows)
__global__ void k_ln_bwd(float* dinp, float* dw, float* db, const float* dout, const float* inp, const float* w,
                         const float* mean, const float* rstd, int T, int C) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= T) return;
    const float* dy = dout + (long)row * C;
    const float* x = inp + (long)row * C;
    float* dx = dinp + (long)row * C;
    const float m = mean[row], r = rstd[row];
    float dm = 0.f, dnm = 0.f;
    for (int i = lane; i < C; i += 32) {
        float nrm = (x[i] - m) * r, dn = w[i] * dy[i];
        dm += dn; dnm += dn * nrm;
    }
    for (int o = 16; o > 0; o >>= 1) {
        dm += __shfl_xor_sync(0xffffffff, dm, o);
        dnm += __shfl_xor_sync(0xffffffff, dnm, o);
    }
    dm /= C; dnm /= C;
    for (int i = lane; i < C; i += 32) {
        float nrm = (x[i] - m) * r, dn = w[i] * dy[i];
        atomicAdd(db + i, dy[i]);
        atomicAdd(dw + i, nrm * dy[i]);
        dx[i] += (dn - dm - nrm * dnm) * r;
    }
}
// Causal attention within a context block: query b attends to keys lo..b, lo = start[b]. One thread per (b, head);
// att[(b*nh + h)*K + (j - blk)] holds the weights, blk = the block's first frame.
__global__ void k_attn_fwd(const float* qkv, float* att, float* out, const int* start, int B, int K, int d, int nh) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * nh) return;
    const int b = i / nh, h = i % nh, hs = d / nh, d3 = 3 * d, lo = start[b], blk = (b / K) * K;
    const float scale = 1.0f / sqrtf((float)hs);
    const float* q = qkv + (long)b * d3 + h * hs;
    float* w = att + (long)i * K;
    float mx = -1e30f;
    for (int j = lo; j <= b; j++) {
        const float* k = qkv + (long)j * d3 + d + h * hs;
        float s = 0.f;
        for (int c = 0; c < hs; c++) s += q[c] * k[c];
        s *= scale;
        w[j - blk] = s;
        mx = fmaxf(mx, s);
    }
    float sum = 0.f;
    for (int j = lo; j <= b; j++) { float e = expf(w[j - blk] - mx); w[j - blk] = e; sum += e; }
    const float inv = 1.0f / sum;
    for (int j = lo; j <= b; j++) w[j - blk] *= inv;
    float* y = out + (long)b * d + h * hs;
    for (int c = 0; c < hs; c++) {
        float s = 0.f;
        for (int j = lo; j <= b; j++) s += w[j - blk] * qkv[(long)j * d3 + 2 * d + h * hs + c];
        y[c] = s;
    }
}
// dqkv must be zero; dk, dv of earlier rows are accumulated with atomics
__global__ void k_attn_bwd(float* dqkv, const float* dout, const float* qkv, const float* att, const int* start, int B,
                           int K, int d, int nh) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * nh) return;
    const int b = i / nh, h = i % nh, hs = d / nh, d3 = 3 * d, lo = start[b], blk = (b / K) * K;
    const float scale = 1.0f / sqrtf((float)hs);
    const float* q = qkv + (long)b * d3 + h * hs;
    const float* w = att + (long)i * K;
    const float* dy = dout + (long)b * d + h * hs;
    float da[64];   // K <= 64
    float dsum = 0.f;
    for (int j = lo; j <= b; j++) {
        const float* v = qkv + (long)j * d3 + 2 * d + h * hs;
        float* dv = dqkv + (long)j * d3 + 2 * d + h * hs;
        float s = 0.f;
        for (int c = 0; c < hs; c++) { s += v[c] * dy[c]; atomicAdd(dv + c, w[j - blk] * dy[c]); }
        da[j - blk] = s;
        dsum += w[j - blk] * s;
    }
    float* dq = dqkv + (long)b * d3 + h * hs;
    for (int j = lo; j <= b; j++) {
        const float dp = w[j - blk] * (da[j - blk] - dsum) * scale;
        const float* k = qkv + (long)j * d3 + d + h * hs;
        float* dk = dqkv + (long)j * d3 + d + h * hs;
        for (int c = 0; c < hs; c++) { atomicAdd(dq + c, dp * k[c]); atomicAdd(dk + c, dp * q[c]); }
    }
}
#define GELU_S 0.7978845608028654f
__global__ void k_gelu_fwd(float* out, const float* inp, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        float x = inp[i];
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_S * (x + 0.044715f * x * x * x)));
    }
}
__global__ void k_gelu_bwd(float* dinp, const float* inp, const float* dout, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        float x = inp[i], ta = GELU_S * (x + 0.044715f * x * x * x), th = tanhf(ta), ch = coshf(ta);
        float g = 0.5f * (1.0f + th) + x * 0.5f * (1.0f / (ch * ch)) * GELU_S * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = g * dout[i];
    }
}
// pin [B, d+3] = [z, executed action]
__global__ void k_pin(const float* z, const float* act, float* pin, int B, int d) {
    const int D = d + 3;
    const long n = (long)B * D;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % D;
        const long b = i / D;
        float v;
        if (j < d) v = z[b * d + j];
        else if (j < d + 2) v = clampf_(act[b * 3 + j - d], -1.f, 1.f);
        else v = act[b * 3 + 2] > 0.5f ? 1.f : 0.f;
        pin[i] = v;
    }
}
// Decoder input 1, channel-major [d1_in, B, q]: ph2 (pc2), ub tiled (pb), x/y coords, a3 (c3), executed action tiled
__global__ void k_dec1_in(const float* ph2, const float* ub, const float* a3, const float* pin, float* x1, int B,
                          int p0, int pc2, int pb, int c3, int d) {
    const int q = p0 * p0, C = pc2 + pb + 2 + c3 + 3;
    const long n = (long)C * B * q;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int p = i % q, c = i / ((long)B * q);
        const long b = (i / q) % B;
        float v;
        if (c < pc2) v = ph2[b * pc2 * q + (long)c * q + p];
        else if (c < pc2 + pb) v = ub[b * pb + c - pc2];
        else if (c == pc2 + pb) v = coord_(p % p0, p0);
        else if (c == pc2 + pb + 1) v = coord_(p / p0, p0);
        else if (c < pc2 + pb + 2 + c3) v = a3[b * c3 * q + (long)(c - pc2 - pb - 2) * q + p];
        else v = pin[b * (d + 3) + d + (c - pc2 - pb - 2 - c3)];
        x1[i] = v;
    }
}
// Decoder input 2, channels pc1.. of [d2_in, B, q] (channels 0..pc1 are deconv 1's output): coords, a2, action
__global__ void k_dec2_in(const float* a2, const float* pin, float* x2, int B, int p1, int pc1, int c2, int d) {
    const int q = p1 * p1, C = 2 + c2 + 3;
    const long n = (long)C * B * q;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int p = i % q, c = i / ((long)B * q);
        const long b = (i / q) % B;
        float v;
        if (c == 0) v = coord_(p % p1, p1);
        else if (c == 1) v = coord_(p / p1, p1);
        else if (c < 2 + c2) v = a2[b * c2 * q + (long)(c - 2) * q + p];
        else v = pin[b * (d + 3) + d + (c - 2 - c2)];
        x2[(long)(pc1 + c) * B * q + b * q + p] = v;
    }
}
// Predicted frame (channel-major [3, B, P]) += current frame / 255 (residual)
__global__ void k_residual(float* pout, const uint8_t* cs, int B, int P) {
    const long n = 3L * B * P;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int p = i % P, c = i / ((long)B * P);
        const long b = (i / P) % B;
        pout[i] += cs[(b * P + p) * 3 + c] / 255.0f;
    }
}
// Frame loss: one thread per (b, pixel). acc[0] += weighted SE / small_px, acc[1] += SE / small_px
__global__ void k_frame_loss(const float* pout, const uint8_t* cs, const uint8_t* ns, float* dpout, int B, int P,
                             float gscale, float wch, double* acc) {
    __shared__ double sh[4][256];
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    double sw = 0, s = 0, cse = 0, cn = 0;
    if (i < (long)B * P) {
        const long b = i / P;
        const int p = i % P;
        const uint8_t* c3 = cs + (b * P + p) * 3;
        const uint8_t* n3 = ns + (b * P + p) * 3;
        const bool changed = c3[0] != n3[0] || c3[1] != n3[1] || c3[2] != n3[2];
        const float wgt = changed ? wch : 1.0f;
        for (int c = 0; c < 3; c++) {
            const long k = (long)c * B * P + b * P + p;
            const float e = pout[k] - n3[c] / 255.0f;
            s += (double)e * e;
            sw += (double)wgt * e * e;
            if (changed) cse += (double)e * e;
            dpout[k] = gscale * wgt * e;
        }
        cn = changed ? 3 : 0;
    }
    sh[0][threadIdx.x] = sw; sh[1][threadIdx.x] = s; sh[2][threadIdx.x] = cse; sh[3][threadIdx.x] = cn;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k)
            for (int j = 0; j < 4; j++) sh[j][threadIdx.x] += sh[j][threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(acc + 0, sh[0][0] / (3.0 * P)); atomicAdd(acc + 1, sh[1][0] / (3.0 * P));
        atomicAdd(acc + 8, sh[2][0]); atomicAdd(acc + 9, sh[3][0]);
    }
}
// Drives and move losses, one thread per token. acc[2] += drive SE / 2, acc[3] += -logp, acc[4] += click p.
// Gradients: dpdrv, dpi, and the log-std's directly into G.
__global__ void k_head_loss(const float* pdrv, const float* ndrv, const float* pi, const float* act,
                            const float* logstd, float* dpdrv, float* dpi, float* glogstd, int B, float inv_B,
                            float drive_coef, float move_coef, double* acc) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    double dse = 0;
    for (int j = 0; j < 2; j++) {
        const float p = pdrv[b * 2 + j], e = p - ndrv[b * 2 + j];
        dse += (double)e * e;
        dpdrv[b * 2 + j] = inv_B * drive_coef * e * p * (1.0f - p);
    }
    atomicAdd(acc + 2, dse / 2.0);
    const float* p3 = pi + b * 3;
    const float* a = act + b * 3;
    float logp = 0.f;
    for (int k = 0; k < 2; k++) {
        const float mu = tanhf(p3[k]), ls = clampf_(logstd[k], LS_MIN, LS_MAX), sd = expf(ls);
        const bool inr = logstd[k] >= LS_MIN && logstd[k] <= LS_MAX;
        const float z = (a[k] - mu) / sd;
        logp += -0.5f * z * z - ls - 0.5f * LOG_2PI;
        dpi[b * 3 + k] = move_coef > 0.f ? -inv_B * move_coef * (z / sd) * (1.0f - mu * mu) : 0.f;
        if (move_coef > 0.f && inr) atomicAdd(glogstd + k, -inv_B * move_coef * (z * z - 1.0f));
    }
    const float l = p3[2], p = sigmoid_(l), c = a[2] > 0.5f ? 1.0f : 0.0f;
    logp += c > 0.5f ? -softplus_(-l) : -softplus_(l);
    dpi[b * 3 + 2] = move_coef > 0.f ? -inv_B * move_coef * (c - p) : 0.f;
    if (move_coef > 0.f) {   // reported only when the move loss is on (as the C++ model)
        atomicAdd(acc + 3, -(double)logp);
        atomicAdd(acc + 4, (double)p);
    }
}
// From deconv 1's input gradient (channel-major [d1_in, B, q]): dph2 (masked by ph2 > 0), dub (summed over the
// tiled positions, masked by ub > 0) and the a3 skip gradient
__global__ void k_dec1_grad(const float* dx1, const float* ph2, const float* ub, float* dph2, float* dub,
                            float* dskip3, int B, int q, int pc2, int pb, int c3, int skip1_off) {
    const long n1 = (long)B * pc2 * q, n2 = (long)B * pb, n3 = (long)B * c3 * q;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n1 + n2 + n3; i += (long)gridDim.x * blockDim.x) {
        if (i < n1) {
            const int p = i % q, c = (i / q) % pc2;
            const long b = i / ((long)pc2 * q);
            dph2[i] = ph2[i] > 0.f ? dx1[(long)c * B * q + b * q + p] : 0.f;
        } else if (i < n1 + n2) {
            const long j = i - n1, b = j / pb;
            const int k = j % pb;
            const float* r = dx1 + (long)(pc2 + k) * B * q + b * q;
            float s = 0.f;
            for (int p = 0; p < q; p++) s += r[p];
            dub[j] = ub[j] > 0.f ? s : 0.f;
        } else {
            const long j = i - n1 - n2;
            const int p = j % q, c = (j / q) % c3;
            const long b = j / ((long)c3 * q);
            dskip3[j] = dx1[(long)(skip1_off + c) * B * q + b * q + p];
        }
    }
}
__global__ void k_mask(float* x, const float* a, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        if (a[i] <= 0.f) x[i] = 0.f;
}
// dz[b, :] (+)= scale * src[b, :d] (src rows of width w)
__global__ void k_add_rows(float* dz, const float* src, int B, int d, int w, float scale, int acc) {
    const long n = (long)B * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const float v = scale * src[(i / d) * w + i % d];
        dz[i] = acc ? dz[i] + v : v;
    }
}
__global__ void k_wpe_grad(const float* dres, const int* pos, float* gwpe, int B, int d) {
    const long n = (long)B * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        atomicAdd(gwpe + (long)pos[i / d] * d + i % d, dres[i]);
}
// dvf = vf > 0 ? dtin[:, :F] : 0
__global__ void k_dvf(const float* dtin, const float* vf, float* dvf, int B, int F, int in_dim) {
    const long n = (long)B * F;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        dvf[i] = vf[i] > 0.f ? dtin[(i / F) * in_dim + i % F] : 0.f;
}
// Squared gradient norms of the three groups: [0] backbone, [1] policy/value heads [h0, h1), [2] prediction heads
// [p0, p1)
__global__ void k_sqnorm(const float* G, long n, long h0, long h1, long p0, long p1, double* out) {
    __shared__ double sh[3][256];
    double s[3] = {0, 0, 0};
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const double g = (double)G[i] * G[i];
        s[(i >= h0 && i < h1) ? 1 : ((i >= p0 && i < p1) ? 2 : 0)] += g;
    }
    for (int k = 0; k < 3; k++) sh[k][threadIdx.x] = s[k];
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k)
            for (int g = 0; g < 3; g++) sh[g][threadIdx.x] += sh[g][threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int g = 0; g < 3; g++) atomicAdd(out + g, sh[g][0]);
}
// clip[0]: the factor for everything but the heads, clip[1]: for the heads (as grad_norm_and_clip)
__global__ void k_clip(const double* gs, float max_norm, int own, float* clip) {
    auto factor = [&](float n) { return (max_norm > 0.f && n > max_norm) ? max_norm / (n + 1e-6f) : 1.0f; };
    const float norm = (float)sqrt(gs[0] + gs[1] + gs[2]);
    clip[0] = factor(own ? (float)sqrt(gs[0] + gs[2]) : norm);
    clip[1] = own ? factor((float)sqrt(gs[1])) : clip[0];
}
__global__ void k_adamw(float* P, const float* G, float* m, float* v, long n, float lr, float b1, float b2, float eps,
                        float wd, float bc1, float bc2, const float* clip, long h0, long h1) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const float c = (i >= h0 && i < h1) ? clip[1] : clip[0];
        const float g = c == 1.0f ? G[i] : G[i] * c;
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        const float mh = m[i] / bc1, vh = v[i] / bc2;
        P[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * P[i]);
    }
}

// Forward outputs per token: o[b*4 + 0..1] = tanh(pi[0..1]), o[b*4 + 2] = sigmoid(pi[2]), o[b*4 + 3] = value
__global__ void k_fwd_out(const float* pi, const float* z, const float* vw, const float* vb, float* o, int B, int d) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    float v = vb[0];
    for (int i = 0; i < d; i++) v += z[(long)b * d + i] * vw[i];
    o[b * 4 + 0] = tanhf(pi[b * 3]); o[b * 4 + 1] = tanhf(pi[b * 3 + 1]);
    o[b * 4 + 2] = sigmoid_(pi[b * 3 + 2]); o[b * 4 + 3] = v;
}
// Predicted frames, channel-major [3, B, P] -> HWC [B, P, 3]
__global__ void k_to_hwc(const float* cm, float* hwc, int B, int P) {
    const long n = 3L * B * P;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int c = i % 3;
        const long bp = i / 3;
        hwc[i] = cm[(long)c * B * P + bp];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------------------------
struct Buf {   // device float buffer, grown on demand
    float* p = nullptr; size_t n = 0;
    float* get(size_t need) {
        if (need > n) { if (p) cudaFree(p); n = need; if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) { p = nullptr; n = 0; } }
        return p;
    }
};

struct Model {
    VMConfig cfg;
    Dims D;
    long off[NUM_P], sz[NUM_P], n_params = 0;
    float *P = nullptr, *G = nullptr, *m = nullptr, *v = nullptr;
    long step = 0;
    cublasHandle_t cb = nullptr;
    int S = 0, T = 0, B = 0;   // the batch: S sequences of T <= ctx steps
    // batch
    uint8_t *img = nullptr, *ns = nullptr, *cs = nullptr, *nfull = nullptr;
    int *start = nullptr, *pos = nullptr;
    float *drv = nullptr, *pa = nullptr, *pr = nullptr, *act = nullptr, *ndrv = nullptr;
    int cap = 0;   // frames allocated
    // activations and gradients
    Buf x0, cols1, cols2, cols3, Y, a1, a2, a3, vf, tin, xin, ln1, ln1m, ln1r, qkv, att, atty, res1, ln2, ln2m, ln2r,
        fch, fchg, res2, lnf, lnfm, lnfr, pih, pi, pin, ph1, ph2, ub, pdrv, d1in, dc1, d2in, dc2, pout;
    Buf fout, fhwc;   // forward outputs: [B, 4] mu, click p, value; predicted frames HWC
    Buf dpout, dpdrv, dpi, dcols, dx2, dx1, dph2, dub, dskip3, dph1, dpin, dz, dpih, dres, dln, datty, dqkv, dfch, dfchg,
        dtin, dvf, da3, da2, da1, dY;
    double* acc = nullptr;    // [0] weighted frame SE, [1] frame SE, [2] drive SE, [3] -logp, [4] click p, [5..7] norms,
                              // [8] SE of the changed pixels, [9] their count (x3)
    float* clip = nullptr;
    float coef[3] = {0, 0, 0};   // pred, drive, move coefficients of the last loss (for the reported total)
};

static void gemm(Model* M, bool ta, bool tb, int m, int n, int k, const float* A, int lda, const float* B, int ldb,
                 float beta, float* C, int ldc) {
    // row-major C[m,n] = op(A) op(B) + beta C  <=>  column-major C^T = op(B)^T op(A)^T
    const float one = 1.0f;
    cublasSgemm(M->cb, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, &one, B, ldb, A, lda,
                &beta, C, ldc);
}
// out [T, OC] = inp [T, C] W^T (+ b, act); W [OC, C]
static void linear(Model* M, float* out, const float* inp, const float* W, const float* b, int T, int C, int OC,
                   int act) {
    gemm(M, false, true, T, OC, C, inp, C, W, C, 0.f, out, OC);
    k_bias_act<<<GRID((long)T * OC)>>>(out, b, T, OC, act);
}
// dinp [T, C] (= or +=) dout W; dW += dout^T inp; db += column sums of dout
static void linear_bwd(Model* M, const float* dout, const float* inp, const float* W, int T, int C, int OC,
                       float* dinp, bool acc, float* dW, float* db) {
    if (dinp) gemm(M, false, false, T, C, OC, dout, OC, W, C, acc ? 1.f : 0.f, dinp, C);
    gemm(M, true, false, OC, C, T, dout, OC, inp, C, 1.f, dW, C);
    k_colsum<<<(OC + 255) / 256, 256>>>(dout, T, OC, db);
}

static void alloc_batch(Model* M, int B) {
    const Dims& D = M->D;
    if (B > M->cap) {
        cudaFree(M->img); cudaFree(M->ns); cudaFree(M->cs); cudaFree(M->nfull); cudaFree(M->start); cudaFree(M->pos);
        cudaFree(M->drv); cudaFree(M->pa); cudaFree(M->pr); cudaFree(M->act); cudaFree(M->ndrv);
        cudaMalloc(&M->img, (size_t)B * D.img * D.img * 3);
        cudaMalloc(&M->ns, (size_t)B * D.small_px);
        cudaMalloc(&M->cs, (size_t)B * D.small_px);
        cudaMalloc(&M->nfull, (size_t)B * D.img * D.img * 3);
        cudaMalloc(&M->start, B * sizeof(int)); cudaMalloc(&M->pos, B * sizeof(int));
        cudaMalloc(&M->drv, B * 2 * sizeof(float)); cudaMalloc(&M->pa, B * 3 * sizeof(float));
        cudaMalloc(&M->pr, B * sizeof(float)); cudaMalloc(&M->act, B * 3 * sizeof(float));
        cudaMalloc(&M->ndrv, B * 2 * sizeof(float));
        M->cap = B;
    }
}

extern "C" {

void* vmc_create(const VMConfig* cfg, const float* params, long n_params, const long* offsets, const long* sizes,
                 int n_tensors) {
    Dims D;
    if (!make_dims(*cfg, D) || n_tensors != NUM_P) { fprintf(stderr, "vm_cuda: unsupported config\n"); return nullptr; }
    long es[NUM_P];
    expected_sizes(D, es);
    long o = 0;
    for (int i = 0; i < NUM_P; i++) {
        if (sizes[i] != es[i] || offsets[i] != o) {
            fprintf(stderr, "vm_cuda: parameter %d: size %ld offset %ld, expected %ld %ld\n", i, sizes[i], offsets[i], es[i], o);
            return nullptr;
        }
        o += es[i];
    }
    if (o != n_params) { fprintf(stderr, "vm_cuda: %ld parameters, expected %ld\n", n_params, o); return nullptr; }
    Model* M = new Model();
    M->cfg = *cfg; M->D = D; M->n_params = n_params;
    for (int i = 0; i < NUM_P; i++) { M->off[i] = offsets[i]; M->sz[i] = sizes[i]; }
    const size_t nb = n_params * sizeof(float);
    if (cudaMalloc(&M->P, nb) || cudaMalloc(&M->G, nb) || cudaMalloc(&M->m, nb) || cudaMalloc(&M->v, nb) ||
        cudaMalloc(&M->acc, 10 * sizeof(double)) || cudaMalloc(&M->clip, 2 * sizeof(float))) {
        fprintf(stderr, "vm_cuda: out of device memory\n"); return nullptr;
    }
    cudaMemcpy(M->P, params, nb, cudaMemcpyHostToDevice);
    cudaMemset(M->G, 0, nb); cudaMemset(M->m, 0, nb); cudaMemset(M->v, 0, nb); cudaMemset(M->acc, 0, 10 * sizeof(double));
    if (cublasCreate(&M->cb) != CUBLAS_STATUS_SUCCESS) return nullptr;
    cublasSetMathMode(M->cb, CUBLAS_PEDANTIC_MATH);   // no TF32
    return M;
}

void vmc_free(void* h) {
    Model* M = (Model*)h;
    if (!M) return;
    cudaFree(M->P); cudaFree(M->G); cudaFree(M->m); cudaFree(M->v); cudaFree(M->acc); cudaFree(M->clip);
    cudaFree(M->img); cudaFree(M->ns); cudaFree(M->cs); cudaFree(M->nfull); cudaFree(M->start); cudaFree(M->pos);
    cudaFree(M->drv); cudaFree(M->pa); cudaFree(M->pr); cudaFree(M->act); cudaFree(M->ndrv);
    for (Buf* b : {&M->x0, &M->cols1, &M->cols2, &M->cols3, &M->Y, &M->a1, &M->a2, &M->a3, &M->vf, &M->tin, &M->xin,
                   &M->ln1, &M->ln1m, &M->ln1r, &M->qkv, &M->att, &M->atty, &M->res1, &M->ln2, &M->ln2m, &M->ln2r,
                   &M->fch, &M->fchg, &M->res2, &M->lnf, &M->lnfm, &M->lnfr, &M->pih, &M->pi, &M->pin, &M->ph1,
                   &M->ph2, &M->ub, &M->pdrv, &M->d1in, &M->dc1, &M->d2in, &M->dc2, &M->pout, &M->dpout, &M->dpdrv,
                   &M->dpi, &M->dcols, &M->dx2, &M->dx1, &M->dph2, &M->dub, &M->dskip3, &M->dph1, &M->dpin, &M->dz,
                   &M->dpih, &M->dres, &M->dln, &M->datty, &M->dqkv, &M->dfch, &M->dfchg, &M->dtin, &M->dvf, &M->da3,
                   &M->da2, &M->da1, &M->dY})
        if (b->p) cudaFree(b->p);
    cublasDestroy(M->cb);
    delete M;
}

}   // extern "C"

// The inputs of S sequences of T steps (ep_start and action may be NULL: no episode starts after t = 0, zero
// actions); the current frames downsampled on the device.
static int upload_inputs(Model* M, int S, int T, const uint8_t* images, const float* drives, const float* prev_action,
                         const float* prev_reward, const uint8_t* ep_start, const float* action) {
    const Dims& D = M->D;
    if (S < 1 || T < 1 || T > D.K) return -1;
    const int B = S * T;
    alloc_batch(M, B);
    M->S = S; M->T = T; M->B = B;
    std::vector<int> start(B), pos(B);
    for (int s = 0; s < S; s++)
        for (int t = 0; t < T; t++) {
            const int b = s * T + t;
            const bool st = t == 0 || M->cfg.one_frame || (ep_start && ep_start[b]);
            start[b] = st ? b : start[b - 1];
            pos[b] = b - start[b];
        }
    CK(cudaMemcpy(M->img, images, (size_t)B * D.img * D.img * 3, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->start, start.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->pos, pos.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->drv, drives, B * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->pa, prev_action, B * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->pr, prev_reward, B * sizeof(float), cudaMemcpyHostToDevice));
    if (action) CK(cudaMemcpy(M->act, action, B * 3 * sizeof(float), cudaMemcpyHostToDevice));
    else CK(cudaMemset(M->act, 0, B * 3 * sizeof(float)));
    k_downsample<<<GRID((long)B * D.small_px)>>>(M->img, M->cs, B, D.img);
    CK(cudaGetLastError());
    return 0;
}

extern "C" {

int vmc_set_batch(void* h, int S, const uint8_t* images, const float* drives, const float* prev_action,
                  const float* prev_reward, const uint8_t* ep_start, const float* action, const uint8_t* next_small,
                  const float* next_drives, int next_full) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (upload_inputs(M, S, D.K, images, drives, prev_action, prev_reward, ep_start, action)) return -1;
    const int B = M->B;
    if (next_full) {   // full-size next frames: downsampled here
        CK(cudaMemcpy(M->nfull, next_small, (size_t)B * D.img * D.img * 3, cudaMemcpyHostToDevice));
        k_downsample<<<GRID((long)B * D.small_px)>>>(M->nfull, M->ns, B, D.img);
    } else {
        CK(cudaMemcpy(M->ns, next_small, (size_t)B * D.small_px, cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(M->ndrv, next_drives, B * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaGetLastError());
    return 0;
}

}   // extern "C"

// The forward pass of the uploaded batch (train = false: up to the outputs; the prediction head and decoder only
// if want_pred), or the forward pass, the losses and the gradients (train = true).
static int run_model(Model* M, const PPOConfig* pcp, bool train, bool want_pred) {
    if (!M->B) return -1;
    const Dims& D = M->D;
    const int B = M->B, K = M->T, d = D.d, L = D.L;   // K: the batch's sequence length (ctx when training)
    const float* P = M->P;
    float* G = M->G;
    const long* off = M->off;
    const int P1 = D.h1 * D.h1, P2 = D.h2 * D.h2, P3 = D.h3 * D.h3, q0 = D.p0 * D.p0, q1 = D.p1 * D.p1;
    const int sp = D.pimg * D.pimg;

    // ---- encoder (all frames) ----
    float* x0 = M->x0.get((size_t)B * 5 * D.img * D.img);
    float* a1 = M->a1.get((size_t)B * D.c1 * P1);
    float* a2 = M->a2.get((size_t)B * D.c2 * P2);
    float* a3 = M->a3.get((size_t)B * D.enc_flat);
    float* cols1 = M->cols1.get((size_t)5 * KK * B * P1);
    float* cols2 = M->cols2.get((size_t)D.c1 * KK * B * P2);
    float* cols3 = M->cols3.get((size_t)D.c2 * KK * B * P3);
    float* Y = M->Y.get((size_t)std::max(D.c1 * P1, std::max(D.c2 * P2, D.c3 * P3)) * B);
    k_input<<<GRID((long)B * 5 * D.img * D.img)>>>(M->img, x0, B, D.img);
    k_im2col<<<GRID((long)5 * KK * B * P1)>>>(x0, 5L * D.img * D.img, (long)D.img * D.img, 5, D.img, B, cols1);
    gemm(M, false, false, D.c1, B * P1, 5 * KK, P + off[C1W], 5 * KK, cols1, B * P1, 0.f, Y, B * P1);
    k_conv_out<<<GRID((long)B * D.c1 * P1)>>>(Y, P + off[C1B], a1, D.c1, B, P1);
    k_im2col<<<GRID((long)D.c1 * KK * B * P2)>>>(a1, (long)D.c1 * P1, P1, D.c1, D.h1, B, cols2);
    gemm(M, false, false, D.c2, B * P2, D.c1 * KK, P + off[C2W], D.c1 * KK, cols2, B * P2, 0.f, Y, B * P2);
    k_conv_out<<<GRID((long)B * D.c2 * P2)>>>(Y, P + off[C2B], a2, D.c2, B, P2);
    k_im2col<<<GRID((long)D.c2 * KK * B * P3)>>>(a2, (long)D.c2 * P2, P2, D.c2, D.h2, B, cols3);
    gemm(M, false, false, D.c3, B * P3, D.c2 * KK, P + off[C3W], D.c2 * KK, cols3, B * P3, 0.f, Y, B * P3);
    k_conv_out<<<GRID((long)B * D.c3 * P3)>>>(Y, P + off[C3B], a3, D.c3, B, P3);

    // ---- tokens and Transformer ----
    float* vf = M->vf.get((size_t)B * D.feat);
    float* tin = M->tin.get((size_t)B * D.in_dim);
    float* xin = M->xin.get((size_t)B * d);
    linear(M, vf, a3, P + off[EFW], P + off[EFB], B, D.enc_flat, D.feat, 1);
    k_token<<<GRID((long)B * D.in_dim)>>>(vf, M->drv, M->pa, M->pr, tin, B, D.feat, M->cfg.use_prev_reward);
    gemm(M, false, true, B, d, D.in_dim, tin, D.in_dim, P + off[INW], D.in_dim, 0.f, xin, d);
    k_bias_wpe<<<GRID((long)B * d)>>>(xin, P + off[INB], P + off[WPE], M->pos, B, d);
    float* ln1 = M->ln1.get((size_t)L * B * d); float* ln1m = M->ln1m.get((size_t)L * B); float* ln1r = M->ln1r.get((size_t)L * B);
    float* qkv = M->qkv.get((size_t)L * B * 3 * d); float* att = M->att.get((size_t)L * B * D.nh * K);
    float* atty = M->atty.get((size_t)L * B * d); float* res1 = M->res1.get((size_t)L * B * d);
    float* ln2 = M->ln2.get((size_t)L * B * d); float* ln2m = M->ln2m.get((size_t)L * B); float* ln2r = M->ln2r.get((size_t)L * B);
    float* fch = M->fch.get((size_t)L * B * D.dff); float* fchg = M->fchg.get((size_t)L * B * D.dff);
    float* res2 = M->res2.get((size_t)L * B * d);
    float* lnf = M->lnf.get((size_t)B * d); float* lnfm = M->lnfm.get(B); float* lnfr = M->lnfr.get(B);
    const int lnb = (B + 7) / 8;   // 8 warps per block
    const float* x = xin;
    for (int l = 0; l < L; l++) {
        const size_t bd = (size_t)l * B * d;
        k_ln_fwd<<<lnb, 256>>>(ln1 + bd, ln1m + l * B, ln1r + l * B, x, P + off[LN1W] + l * d, P + off[LN1B] + l * d, B, d);
        linear(M, qkv + bd * 3, ln1 + bd, P + off[QKVW] + (size_t)l * 3 * d * d, P + off[QKVB] + l * 3 * d, B, d, 3 * d, 0);
        k_attn_fwd<<<(B * D.nh + 127) / 128, 128>>>(qkv + bd * 3, att + (size_t)l * B * D.nh * K, atty + bd, M->start, B, K, d, D.nh);
        cudaMemcpyAsync(res1 + bd, x, sizeof(float) * B * d, cudaMemcpyDeviceToDevice);
        gemm(M, false, true, B, d, d, atty + bd, d, P + off[APW] + (size_t)l * d * d, d, 1.f, res1 + bd, d);
        k_bias_act<<<GRID((long)B * d)>>>(res1 + bd, P + off[APB] + l * d, B, d, 0);
        k_ln_fwd<<<lnb, 256>>>(ln2 + bd, ln2m + l * B, ln2r + l * B, res1 + bd, P + off[LN2W] + l * d, P + off[LN2B] + l * d, B, d);
        linear(M, fch + (size_t)l * B * D.dff, ln2 + bd, P + off[FCW] + (size_t)l * D.dff * d, P + off[FCB] + l * D.dff, B, d, D.dff, 0);
        k_gelu_fwd<<<GRID((long)B * D.dff)>>>(fchg + (size_t)l * B * D.dff, fch + (size_t)l * B * D.dff, (long)B * D.dff);
        cudaMemcpyAsync(res2 + bd, res1 + bd, sizeof(float) * B * d, cudaMemcpyDeviceToDevice);
        gemm(M, false, true, B, d, D.dff, fchg + (size_t)l * B * D.dff, D.dff, P + off[FPW] + (size_t)l * d * D.dff, D.dff, 1.f, res2 + bd, d);
        k_bias_act<<<GRID((long)B * d)>>>(res2 + bd, P + off[FPB] + l * d, B, d, 0);
        x = res2 + bd;
    }
    k_ln_fwd<<<lnb, 256>>>(lnf, lnfm, lnfr, x, P + off[LNFW], P + off[LNFB], B, d);

    // ---- heads ----
    float* pi = M->pi.get((size_t)B * 3);
    float* pih = D.pih ? M->pih.get((size_t)B * D.pih) : nullptr;
    if (D.pih) {
        linear(M, pih, lnf, P + off[PHW], P + off[PHB], B, d, D.pih, 1);
        linear(M, pi, pih, P + off[PIW], P + off[PIB], B, D.pih, 3, 0);
    } else {
        linear(M, pi, lnf, P + off[PIW], P + off[PIB], B, d, 3, 0);
    }
    float* pin = M->pin.get((size_t)B * D.pin_dim);
    float* ph1 = M->ph1.get((size_t)B * D.ph);
    float* ph2 = M->ph2.get((size_t)B * D.pred_flat);
    float* ub = M->ub.get((size_t)B * D.pb);
    float* pdrv = M->pdrv.get((size_t)B * 2);
    if (want_pred) {
    k_pin<<<GRID((long)B * D.pin_dim)>>>(lnf, M->act, pin, B, d);
    linear(M, ph1, pin, P + off[PR1W], P + off[PR1B], B, D.pin_dim, D.ph, 1);
    linear(M, ph2, ph1, P + off[PR2W], P + off[PR2B], B, D.ph, D.pred_flat, 1);
    linear(M, ub, ph1, P + off[PBW], P + off[PBB], B, D.ph, D.pb, 1);
    linear(M, pdrv, ph1, P + off[DRW], P + off[DRB], B, D.ph, 2, 2);
    }

    // ---- decoder (all frames, channel-major) ----
    float* d1in = M->d1in.get((size_t)D.d1_in * B * q0);
    float* dc1 = M->dc1.get((size_t)D.pc1 * KK * B * q0);
    float* d2in = M->d2in.get((size_t)D.d2_in * B * q1);
    float* dc2 = M->dc2.get((size_t)3 * KK * B * q1);
    float* pout = M->pout.get((size_t)3 * B * sp);
    if (want_pred) {
    k_dec1_in<<<GRID((long)D.d1_in * B * q0)>>>(ph2, ub, a3, pin, d1in, B, D.p0, D.pc2, D.pb, D.c3, d);
    gemm(M, true, false, D.pc1 * KK, B * q0, D.d1_in, P + off[D1W], D.pc1 * KK, d1in, B * q0, 0.f, dc1, B * q0);
    k_col2im<<<GRID((long)B * D.pc1 * q1)>>>(dc1, D.pc1, D.p1, B, d2in, q1, (long)B * q1, P + off[D1B], 1);
    k_dec2_in<<<GRID((long)(D.d2_in - D.pc1) * B * q1)>>>(a2, pin, d2in, B, D.p1, D.pc1, D.c2, d);
    gemm(M, true, false, 3 * KK, B * q1, D.d2_in, P + off[D2W], 3 * KK, d2in, B * q1, 0.f, dc2, B * q1);
    k_col2im<<<GRID((long)B * 3 * sp)>>>(dc2, 3, D.pimg, B, pout, sp, (long)B * sp, P + off[D2B], 0);
    k_residual<<<GRID(3L * B * sp)>>>(pout, M->cs, B, sp);
    }
    if (!train) { CK(cudaGetLastError()); return 0; }
    const PPOConfig& pc = *pcp;
    CK(cudaMemsetAsync(G, 0, M->n_params * sizeof(float)));
    CK(cudaMemsetAsync(M->acc, 0, 10 * sizeof(double)));
    M->coef[0] = pc.pred_coef; M->coef[1] = pc.drive_coef; M->coef[2] = pc.move_coef;

    // ---- losses ----
    const float inv_B = 1.0f / B;
    float* dpout = M->dpout.get((size_t)3 * B * sp);
    float* dpdrv = M->dpdrv.get((size_t)B * 2);
    float* dpi = M->dpi.get((size_t)B * 3);
    const bool have_pred = pc.pred_coef > 0.f || pc.drive_coef > 0.f;
    k_frame_loss<<<(B * sp + 255) / 256, 256>>>(pout, M->cs, M->ns, dpout, B, sp,
                                                 inv_B * pc.pred_coef * 2.0f / D.small_px, 1.0f + pc.pred_changed_weight, M->acc);
    k_head_loss<<<(B + 127) / 128, 128>>>(pdrv, M->ndrv, pi, M->act, P + off[LOGSTD], dpdrv, dpi, G + off[LOGSTD], B,
                                          inv_B, pc.drive_coef, pc.move_coef, M->acc);
    const float sc = pc.pred_backbone_scale;

    // ---- decoder backward ----
    float* dcols = M->dcols.get((size_t)std::max(std::max(3 * KK * B * q1, D.pc1 * KK * B * q0),
                                                 std::max(D.c2 * KK * B * P3, D.c1 * KK * B * P2)));
    float* dx2 = M->dx2.get((size_t)D.d2_in * B * q1);
    float* dx1 = M->dx1.get((size_t)D.d1_in * B * q0);
    float* dph2 = M->dph2.get((size_t)B * D.pred_flat);
    float* dub = M->dub.get((size_t)B * D.pb);
    float* dskip3 = M->dskip3.get((size_t)B * D.enc_flat);
    if (have_pred) {
        k_rowsum<<<3, 256>>>(dpout, (long)B * sp, G + off[D2B]);
        k_im2col<<<GRID(3L * KK * B * q1)>>>(dpout, sp, (long)B * sp, 3, D.pimg, B, dcols);
        gemm(M, false, true, D.d2_in, 3 * KK, B * q1, d2in, B * q1, dcols, B * q1, 1.f, G + off[D2W], 3 * KK);
        gemm(M, false, false, D.d2_in, B * q1, 3 * KK, P + off[D2W], 3 * KK, dcols, B * q1, 0.f, dx2, B * q1);
        k_mask<<<GRID((long)D.pc1 * B * q1)>>>(dx2, d2in, (long)D.pc1 * B * q1);   // ReLU of deconv 1 (channels 0..pc1)
        k_rowsum<<<D.pc1, 256>>>(dx2, (long)B * q1, G + off[D1B]);
        k_im2col<<<GRID((long)D.pc1 * KK * B * q0)>>>(dx2, q1, (long)B * q1, D.pc1, D.p1, B, dcols);
        gemm(M, false, true, D.d1_in, D.pc1 * KK, B * q0, d1in, B * q0, dcols, B * q0, 1.f, G + off[D1W], D.pc1 * KK);
        gemm(M, false, false, D.d1_in, B * q0, D.pc1 * KK, P + off[D1W], D.pc1 * KK, dcols, B * q0, 0.f, dx1, B * q0);
        const long nn = (long)B * D.pred_flat + (long)B * D.pb + (long)B * D.enc_flat;
        k_dec1_grad<<<GRID(nn)>>>(dx1, ph2, ub, dph2, dub, dskip3, B, q0, D.pc2, D.pb, D.c3, D.skip1_off);
    }

    // ---- heads backward ----
    float* dz = M->dz.get((size_t)B * d);
    CK(cudaMemsetAsync(dz, 0, sizeof(float) * B * d));
    float* dph1 = M->dph1.get((size_t)B * D.ph);
    float* dpin = M->dpin.get((size_t)B * D.pin_dim);
    if (have_pred) {
        linear_bwd(M, dpdrv, ph1, P + off[DRW], B, D.ph, 2, dph1, false, G + off[DRW], G + off[DRB]);
        linear_bwd(M, dub, ph1, P + off[PBW], B, D.ph, D.pb, dph1, true, G + off[PBW], G + off[PBB]);
        linear_bwd(M, dph2, ph1, P + off[PR2W], B, D.ph, D.pred_flat, dph1, true, G + off[PR2W], G + off[PR2B]);
        k_mask<<<GRID((long)B * D.ph)>>>(dph1, ph1, (long)B * D.ph);
        linear_bwd(M, dph1, pin, P + off[PR1W], B, D.pin_dim, D.ph, dpin, false, G + off[PR1W], G + off[PR1B]);
        if (sc != 0.f) k_add_rows<<<GRID((long)B * d)>>>(dz, dpin, B, d, D.pin_dim, sc, 1);
    }
    float* dz_pi = pc.pi_detach ? nullptr : dz;
    if (D.pih) {
        float* dpih = M->dpih.get((size_t)B * D.pih);
        linear_bwd(M, dpi, pih, P + off[PIW], B, D.pih, 3, dpih, false, G + off[PIW], G + off[PIB]);
        k_mask<<<GRID((long)B * D.pih)>>>(dpih, pih, (long)B * D.pih);
        linear_bwd(M, dpih, lnf, P + off[PHW], B, d, D.pih, dz_pi, true, G + off[PHW], G + off[PHB]);
    } else {
        linear_bwd(M, dpi, lnf, P + off[PIW], B, d, 3, dz_pi, true, G + off[PIW], G + off[PIB]);
    }

    // ---- Transformer backward ----
    float* dres = M->dres.get((size_t)B * d);
    float* dln = M->dln.get((size_t)B * d);
    float* datty = M->datty.get((size_t)B * d);
    float* dqkv = M->dqkv.get((size_t)B * 3 * d);
    float* dfch = M->dfch.get((size_t)B * D.dff);
    float* dfchg = M->dfchg.get((size_t)B * D.dff);
    CK(cudaMemsetAsync(dres, 0, sizeof(float) * B * d));
    k_ln_bwd<<<lnb, 256>>>(dres, G + off[LNFW], G + off[LNFB], dz, res2 + (size_t)(L - 1) * B * d, P + off[LNFW], lnfm, lnfr, B, d);
    for (int l = L - 1; l >= 0; l--) {
        const size_t bd = (size_t)l * B * d;
        const float* xl = l == 0 ? xin : res2 + (size_t)(l - 1) * B * d;
        linear_bwd(M, dres, fchg + (size_t)l * B * D.dff, P + off[FPW] + (size_t)l * d * D.dff, B, D.dff, d, dfchg, false,
                   G + off[FPW] + (size_t)l * d * D.dff, G + off[FPB] + l * d);
        k_gelu_bwd<<<GRID((long)B * D.dff)>>>(dfch, fch + (size_t)l * B * D.dff, dfchg, (long)B * D.dff);
        linear_bwd(M, dfch, ln2 + bd, P + off[FCW] + (size_t)l * D.dff * d, B, d, D.dff, dln, false,
                   G + off[FCW] + (size_t)l * D.dff * d, G + off[FCB] + l * D.dff);
        k_ln_bwd<<<lnb, 256>>>(dres, G + off[LN2W] + l * d, G + off[LN2B] + l * d, dln, res1 + bd, P + off[LN2W] + l * d,
                               ln2m + l * B, ln2r + l * B, B, d);
        linear_bwd(M, dres, atty + bd, P + off[APW] + (size_t)l * d * d, B, d, d, datty, false,
                   G + off[APW] + (size_t)l * d * d, G + off[APB] + l * d);
        CK(cudaMemsetAsync(dqkv, 0, sizeof(float) * B * 3 * d));
        k_attn_bwd<<<(B * D.nh + 127) / 128, 128>>>(dqkv, datty, qkv + bd * 3, att + (size_t)l * B * D.nh * K, M->start, B, K, d, D.nh);
        linear_bwd(M, dqkv, ln1 + bd, P + off[QKVW] + (size_t)l * 3 * d * d, B, d, 3 * d, dln, false,
                   G + off[QKVW] + (size_t)l * 3 * d * d, G + off[QKVB] + l * 3 * d);
        k_ln_bwd<<<lnb, 256>>>(dres, G + off[LN1W] + l * d, G + off[LN1B] + l * d, dln, xl, P + off[LN1W] + l * d,
                               ln1m + l * B, ln1r + l * B, B, d);
    }
    k_wpe_grad<<<GRID((long)B * d)>>>(dres, M->pos, G + off[WPE], B, d);
    float* dtin = M->dtin.get((size_t)B * D.in_dim);
    float* dvf = M->dvf.get((size_t)B * D.feat);
    float* da3 = M->da3.get((size_t)B * D.enc_flat);
    linear_bwd(M, dres, tin, P + off[INW], B, D.in_dim, d, dtin, false, G + off[INW], G + off[INB]);
    k_dvf<<<GRID((long)B * D.feat)>>>(dtin, vf, dvf, B, D.feat, D.in_dim);
    linear_bwd(M, dvf, a3, P + off[EFW], B, D.enc_flat, D.feat, da3, false, G + off[EFW], G + off[EFB]);

    // ---- encoder backward ----
    const bool skip = have_pred && sc != 0.f;
    float* dY = M->dY.get((size_t)std::max(D.c1 * P1, std::max(D.c2 * P2, D.c3 * P3)) * B);
    float* da2 = M->da2.get((size_t)B * D.c2 * P2);
    float* da1 = M->da1.get((size_t)B * D.c1 * P1);
    float* dcl = dcols;
    // conv 3
    k_conv_grad<<<GRID((long)B * D.c3 * P3)>>>(da3, skip ? dskip3 : nullptr, D.enc_flat, P3, sc, a3, dY, D.c3, B, P3);
    gemm(M, false, true, D.c3, D.c2 * KK, B * P3, dY, B * P3, cols3, B * P3, 1.f, G + off[C3W], D.c2 * KK);
    k_rowsum<<<D.c3, 256>>>(dY, (long)B * P3, G + off[C3B]);
    gemm(M, true, false, D.c2 * KK, B * P3, D.c3, P + off[C3W], D.c2 * KK, dY, B * P3, 0.f, dcl, B * P3);
    k_col2im<<<GRID((long)B * D.c2 * P2)>>>(dcl, D.c2, D.h2, B, da2, (long)D.c2 * P2, P2, nullptr, 0);
    // conv 2 (+ the decoder's skip gradient of a2: channels skip_off.. of dx2, channel-major)
    k_conv_grad<<<GRID((long)B * D.c2 * P2)>>>(da2, skip ? dx2 + (size_t)D.skip_off * B * q1 : nullptr, q1, (long)B * q1,
                                               sc, a2, dY, D.c2, B, P2);
    gemm(M, false, true, D.c2, D.c1 * KK, B * P2, dY, B * P2, cols2, B * P2, 1.f, G + off[C2W], D.c1 * KK);
    k_rowsum<<<D.c2, 256>>>(dY, (long)B * P2, G + off[C2B]);
    gemm(M, true, false, D.c1 * KK, B * P2, D.c2, P + off[C2W], D.c1 * KK, dY, B * P2, 0.f, dcl, B * P2);
    k_col2im<<<GRID((long)B * D.c1 * P1)>>>(dcl, D.c1, D.h1, B, da1, (long)D.c1 * P1, P1, nullptr, 0);
    // conv 1
    k_conv_grad<<<GRID((long)B * D.c1 * P1)>>>(da1, nullptr, 0, 0, 0.f, a1, dY, D.c1, B, P1);
    gemm(M, false, true, D.c1, 5 * KK, B * P1, dY, B * P1, cols1, B * P1, 1.f, G + off[C1W], 5 * KK);
    k_rowsum<<<D.c1, 256>>>(dY, (long)B * P1, G + off[C1B]);
    CK(cudaGetLastError());
    return 0;
}

extern "C" {

int vmc_loss_and_grads(void* h, const PPOConfig* pcp) {
    Model* M = (Model*)h;
    const PPOConfig& pc = *pcp;
    if (pc.use_ppo || pc.inv_coef > 0.f || pc.dyn_coef > 0.f || pc.dyn_h_coef > 0.f || pc.dyn_x_coef > 0.f) {
        fprintf(stderr, "vm_cuda: only the prediction-only update (use_ppo=0, no inverse or latent dynamics)\n");
        return -1;
    }
    if (M->T != M->D.K) { fprintf(stderr, "vm_cuda: training sequences are ctx steps (vmc_set_batch)\n"); return -1; }
    return run_model(M, pcp, true, true);
}

int vmc_forward(void* h, int S, int T, const uint8_t* images, const float* drives, const float* prev_action,
                const float* prev_reward, const uint8_t* ep_start, const float* pred_actions, float* z, float* value,
                float* mu, float* click_p, float* pred_frames, float* pred_drives) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (upload_inputs(M, S, T, images, drives, prev_action, prev_reward, ep_start, pred_actions)) return -1;
    const bool want_pred = pred_actions != nullptr;
    if (run_model(M, nullptr, false, want_pred)) return -1;
    const int B = M->B, sp = D.pimg * D.pimg;
    float* o = M->fout.get((size_t)B * 4);
    k_fwd_out<<<(B + 127) / 128, 128>>>(M->pi.p, M->lnf.p, M->P + M->off[VW], M->P + M->off[VB], o, B, D.d);
    std::vector<float> ho((size_t)B * 4);
    CK(cudaMemcpy(ho.data(), o, ho.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int b = 0; b < B; b++) {
        if (mu) { mu[2 * b] = ho[4 * b]; mu[2 * b + 1] = ho[4 * b + 1]; }
        if (click_p) click_p[b] = ho[4 * b + 2];
        if (value) value[b] = ho[4 * b + 3];
    }
    if (z) CK(cudaMemcpy(z, M->lnf.p, (size_t)B * D.d * sizeof(float), cudaMemcpyDeviceToHost));
    if (want_pred && pred_frames) {
        float* f = M->fhwc.get((size_t)B * 3 * sp);
        k_to_hwc<<<GRID(3L * B * sp)>>>(M->pout.p, f, B, sp);
        CK(cudaMemcpy(pred_frames, f, (size_t)B * 3 * sp * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (want_pred && pred_drives) CK(cudaMemcpy(pred_drives, M->pdrv.p, (size_t)B * 2 * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_get_range(void* h, long offset, long n, float* out) {
    Model* M = (Model*)h;
    if (offset < 0 || n < 0 || offset + n > M->n_params) return -1;
    CK(cudaMemcpy(out, M->P + offset, n * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_update(void* h, const PPOConfig* pcp) {
    Model* M = (Model*)h;
    const PPOConfig& pc = *pcp;
    if (vmc_loss_and_grads(h, pcp)) return -1;
    const long h0 = M->off[PIW], h1 = M->off[VB] + M->sz[VB], p0 = M->off[PR1W], p1 = M->off[PEW];
    k_sqnorm<<<256, 256>>>(M->G, M->n_params, h0, h1, p0, p1, M->acc + 5);
    k_clip<<<1, 1>>>(M->acc + 5, pc.max_grad_norm, pc.pi_own_clip, M->clip);
    M->step++;
    const float bc1 = 1.0f - powf(pc.beta1, (float)M->step), bc2 = 1.0f - powf(pc.beta2, (float)M->step);
    k_adamw<<<GRID(M->n_params)>>>(M->P, M->G, M->m, M->v, M->n_params, pc.lr, pc.beta1, pc.beta2, pc.adam_eps,
                                   pc.weight_decay, bc1, bc2, M->clip, h0, h1);
    CK(cudaGetLastError());
    return 0;
}

int vmc_stats(void* h, VMCStats* out) {
    Model* M = (Model*)h;
    double a[10];
    CK(cudaMemcpy(a, M->acc, sizeof(a), cudaMemcpyDeviceToHost));
    const double n = M->B > 0 ? M->B : 1;
    memset(out, 0, sizeof(VMCStats));
    out->loss = (float)((M->coef[0] * a[0] + M->coef[1] * a[2] + M->coef[2] * a[3]) / n);
    out->pred_mse = (float)(a[1] / n);
    out->drive_mse = (float)(a[2] / n);
    out->move_nll = (float)(a[3] / n);
    out->click_p = (float)(a[4] / n);
    out->pred_changed_mse = a[9] > 0 ? (float)(a[8] / a[9]) : 0.f;
    out->grad_norm = (float)sqrt(a[5] + a[6] + a[7]);
    out->gn_backbone = (float)sqrt(a[5]); out->gn_policy = (float)sqrt(a[6]); out->gn_pred = (float)sqrt(a[7]);
    return 0;
}

int vmc_params(void* h, float* out) {
    Model* M = (Model*)h;
    CK(cudaMemcpy(out, M->P, M->n_params * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}
int vmc_grads(void* h, float* out) {
    Model* M = (Model*)h;
    CK(cudaMemcpy(out, M->G, M->n_params * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}
int vmc_set_params(void* h, const float* params) {
    Model* M = (Model*)h;
    CK(cudaMemcpy(M->P, params, M->n_params * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}
int vmc_sync(void* h) { CK(cudaDeviceSynchronize()); return 0; }

}   // extern "C"
