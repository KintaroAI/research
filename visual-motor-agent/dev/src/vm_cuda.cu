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
    PEW, PEB, POS2D, TYP, DEW, DEB, MEW, MEB, CEW, CEB,
    ACTE, AX1W, AX1B, AXDW, AXDB, AXFW, AXFB,         // act token (act_token = 1) and its auxiliary heads
    UIW, UIB, USL, UOW, UOB, NUM_P                    // the universal model (universal = 1)
};
#define ACC_N 34   // loss accumulators (see the kernels that fill them; 24..33: the universal model)
#define UNI_SLOT 6  // ints per slot in the universal model's device slot table (kind, index, block y, x, fy, fx)
static const float LS_MIN = -5.0f, LS_MAX = 1.0f, LOG_2PI = 1.8378770664093453f;
static const int KK = 16;   // 4x4 kernels, stride 2, padding 1

struct Dims {
    int img, in_ch, c1, c2, c3, h1, h2, h3, enc_flat, feat, d, L, nh, hs, dff, K, in_dim;
    int pimg, p0, p1, pc1, pc2, ph, pred_flat, pin_dim, small_px, pb, d1_in, d2_in, skip1_off, skip_off, pih;
    int ih;   // inverse-dynamics hidden size (0: no inverse head)
    int dh, rimg, rpx;   // latent dynamics hidden size (0: none); coarse decoder frame side and values
    int patch, gp, npch, ntok, pdim, cell, cpx;   // patch tokens (0: the CNN model): side, grid, patches, tokens per
                                                  // step, values per patch, predicted cell side and values
    int act, cds, cfx;   // act token on/off (task 00020); prediction frame -> coarse frame block side; coarse values
    int ib, VI;                 // bin-embedding input (task 00036): on/off; the input map's width (V or V * bins)
    int uni, V, nbin, OV, ch;   // universal model (task 00023): on/off; values per token; bins per value; head
                                // outputs; channels per patch pixel (1 with gray, else 3)
    int nsum, nnot;             // summary tokens per step (tasks 00047, 00048; 0: none); tokens without a target
};
static const int ACT_CG = 8;   // act token: side of the coarse next frame it predicts (as vm_agent.cpp)

static bool make_dims(const VMConfig& c, Dims& D) {
    D.patch = c.patch > 0 ? c.patch : 0;
    D.act = (D.patch && c.act_token) ? 1 : 0; D.cds = c.pred_img / ACT_CG; D.cfx = ACT_CG * ACT_CG * 3;
    if (c.act_token && (!D.patch || c.pred_img % ACT_CG)) return false;
    if (D.patch) {   // patch tokens (vm-00010): residual frames; no inverse or latent dynamics (as vm_agent.cpp)
        if ((!c.pred_residual && !c.universal) || c.inv_hidden > 0 || c.dyn_hidden > 0 || c.img % D.patch) return false;
        D.gp = c.img / D.patch; D.npch = D.gp * D.gp; D.ntok = D.npch + 3 + D.act; D.pdim = D.patch * D.patch * 3;
        D.uni = c.universal ? 1 : 0;
        if (D.uni) { if (c.act_token || c.bins < 2) return false; D.ntok = vm_uni_nt(&c); }
        D.nsum = D.uni ? vm_uni_nsum(&c) : 0; D.nnot = D.uni ? vm_uni_untargeted(&c) : 0;
        if (!vm_attn_valid(&c)) return false;
        D.ch = (D.uni && c.gray) ? 1 : 3;
        D.V = D.uni ? D.patch * D.patch * D.ch : 0; D.nbin = D.uni ? c.bins : 0; D.OV = D.V * D.nbin;
        D.ib = D.uni ? c.in_bins : 0; D.VI = D.ib ? D.V * D.nbin : D.V;
        if (c.pred_img % D.gp || c.d_model % c.n_head) return false;
        D.cell = c.pred_img / D.gp; D.cpx = D.cell * D.cell * 3;
        D.img = c.img; D.in_ch = 5; D.c1 = c.c1; D.c2 = c.c2; D.c3 = c.c3;
        D.h1 = c.img / 2; D.h2 = c.img / 4; D.h3 = c.img / 8; D.enc_flat = c.c3 * D.h3 * D.h3;
        D.feat = c.feat; D.d = c.d_model; D.L = c.n_layer; D.nh = c.n_head; D.hs = c.d_model / c.n_head;
        D.dff = c.d_ff; D.K = c.ctx; D.in_dim = c.feat + 6;
        D.pimg = c.pred_img; D.p0 = c.pred_img / 4; D.p1 = c.pred_img / 2; D.pc1 = c.pc1; D.pc2 = c.pc2;
        D.ph = c.pred_hidden; D.pred_flat = c.pc2 * D.p0 * D.p0; D.pin_dim = c.d_model + 3;
        D.small_px = c.pred_img * c.pred_img * 3; D.pb = 0; D.skip1_off = D.d1_in = D.skip_off = D.d2_in = 0;
        D.pih = c.pi_hidden > 0 ? c.pi_hidden : 0; D.ih = 0; D.dh = 0; D.rimg = c.pred_img / 2; D.rpx = 0;
        return true;
    }
    D.gp = D.npch = D.pdim = D.cell = D.cpx = 0; D.ntok = 1;
    D.uni = D.V = D.nbin = D.OV = 0; D.ch = 3; D.nsum = D.nnot = 0;
    if (c.universal || c.gray) return false;
    if (!c.coord || !c.pred_residual || !c.pred_skip) return false;
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
    D.ih = c.inv_hidden > 0 ? c.inv_hidden : 0;
    D.dh = c.dyn_hidden > 0 ? c.dyn_hidden : 0; D.rimg = c.pred_img / 2; D.rpx = D.rimg * D.rimg * 3;
    return D.p1 == D.h2 && D.p0 == D.h3;   // U-Net skips need equal map sizes
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
    s[IV1W] = (long)D.ih * 2 * D.feat; s[IV1B] = D.ih; s[IV2W] = 2L * D.ih; s[IV2B] = D.ih ? 2 : 0;
    const long dh = D.dh, F = D.feat;
    s[T1W] = dh * (F + 3); s[T1B] = dh; s[T2W] = F * dh; s[T2B] = dh ? F : 0;
    s[H1W] = dh * (F + 4); s[H1B] = dh; s[H2W] = dh; s[H2B] = dh ? 1 : 0;
    s[X1W] = dh * F; s[X1B] = dh; s[X2W] = (long)D.rpx * dh; s[X2B] = dh ? D.rpx : 0;
    if (D.patch) {   // no CNN encoder or decoder; PR2W outputs one patch's cell
        for (int q : {C1W, C1B, C2W, C2B, C3W, C3B, EFW, EFB, INW, INB, PBW, PBB, D1W, D1B, D2W, D2B}) s[q] = 0;
        s[PR2W] = (long)D.cpx * D.ph; s[PR2B] = D.cpx;
        s[PEW] = d * D.pdim; s[PEB] = d; s[POS2D] = (long)D.npch * d; s[TYP] = (4 + D.act) * d;
        s[DEW] = 2 * d; s[DEB] = d; s[MEW] = 3 * d; s[MEB] = d; s[CEW] = d; s[CEB] = d;
        const long ac = D.act;
        s[ACTE] = ac * d; s[AX1W] = ac * D.ph * D.pin_dim; s[AX1B] = ac * D.ph; s[AXDW] = ac * 2 * D.ph; s[AXDB] = ac * 2;
        s[AXFW] = ac * D.cfx * D.ph; s[AXFB] = ac * D.cfx;
        if (D.uni) {   // one input map, slot embeddings, one output head; no per-type heads (as vm_agent.cpp)
            for (int q : {PIW, PIB, PHW, PHB, VW, VB, PR1W, PR1B, PR2W, PR2B, DRW, DRB, PEW, PEB, POS2D, TYP, DEW, DEB,
                          MEW, MEB, CEW, CEB}) s[q] = 0;
            s[UIW] = d * D.VI; s[UIB] = d; s[USL] = (long)D.ntok * d; s[UOW] = (long)D.OV * d; s[UOB] = D.OV;
        }
    }
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
// db[o] += sum_t dout[t, o]. Blocks of 32 columns x 8 row lanes (coalesced reads), each over COLSUM_ROWS rows; the
// blocks' partial sums meet in one atomic add per column (one thread per column over all rows left most of the GPU
// idle: a fifth of the 8x8-patch update, found with Nsight Compute)
#define COLSUM_ROWS 256
__global__ void k_colsum(const float* dout, int T, int OC, float* db) {
    __shared__ float sh[8][33];
    const int o = blockIdx.x * 32 + threadIdx.x, t1 = min(T, (blockIdx.y + 1) * COLSUM_ROWS);
    float s = 0.f;
    if (o < OC)
        for (int t = blockIdx.y * COLSUM_ROWS + threadIdx.y; t < t1; t += 8) s += dout[(long)t * OC + o];
    sh[threadIdx.y][threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.y == 0 && o < OC) {
        for (int k = 1; k < 8; k++) s += sh[k][threadIdx.x];
        atomicAdd(db + o, s);
    }
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
// allow (attention regions, task 00044; NULL = none): row b attends to key j only if allow[(b % nt) * nt + j % nt]
__global__ void k_attn_fwd(const float* qkv, float* att, float* out, const int* start, const int* end, int B, int K,
                           int d, int nh, const unsigned char* allow, int nt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * nh) return;
    const int b = i / nh, h = i % nh, hs = d / nh, d3 = 3 * d, lo = start[b], hi = end ? end[b] : b, blk = (b / K) * K;
    const float scale = 1.0f / sqrtf((float)hs);
    const float* q = qkv + (long)b * d3 + h * hs;
    const unsigned char* al = allow ? allow + (b % nt) * nt : nullptr;
    float* w = att + (long)i * K;
    float mx = -1e30f;
    for (int j = lo; j <= hi; j++) {
        if (al && !al[j % nt]) { w[j - blk] = -INFINITY; continue; }   // weight exp(-inf) = 0
        const float* k = qkv + (long)j * d3 + d + h * hs;
        float s = 0.f;
        for (int c = 0; c < hs; c++) s += q[c] * k[c];
        s *= scale;
        w[j - blk] = s;
        mx = fmaxf(mx, s);
    }
    float sum = 0.f;
    for (int j = lo; j <= hi; j++) { float e = expf(w[j - blk] - mx); w[j - blk] = e; sum += e; }
    const float inv = 1.0f / sum;
    for (int j = lo; j <= hi; j++) w[j - blk] *= inv;
    float* y = out + (long)b * d + h * hs;
    for (int c = 0; c < hs; c++) {
        float s = 0.f;
        for (int j = lo; j <= hi; j++) s += w[j - blk] * qkv[(long)j * d3 + 2 * d + h * hs + c];
        y[c] = s;
    }
}
// dqkv must be zero; dk, dv of earlier rows are accumulated with atomics
__global__ void k_attn_bwd(float* dqkv, const float* dout, const float* qkv, const float* att, const int* start,
                           const int* end, int B, int K, int d, int nh) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * nh) return;
    const int b = i / nh, h = i % nh, hs = d / nh, d3 = 3 * d, lo = start[b], hi = end ? end[b] : b, blk = (b / K) * K;
    const float scale = 1.0f / sqrtf((float)hs);
    const float* q = qkv + (long)b * d3 + h * hs;
    const float* w = att + (long)i * K;
    const float* dy = dout + (long)b * d + h * hs;
    float dsum = 0.f;   // the gradient at the weights, v_j . dy, is recomputed in the second loop: any number of keys
    for (int j = lo; j <= hi; j++) {
        const float* v = qkv + (long)j * d3 + 2 * d + h * hs;
        float* dv = dqkv + (long)j * d3 + 2 * d + h * hs;
        float s = 0.f;
        for (int c = 0; c < hs; c++) { s += v[c] * dy[c]; atomicAdd(dv + c, w[j - blk] * dy[c]); }
        dsum += w[j - blk] * s;
    }
    float* dq = dqkv + (long)b * d3 + h * hs;
    for (int j = lo; j <= hi; j++) {
        const float* v = qkv + (long)j * d3 + 2 * d + h * hs;
        float s = 0.f;
        for (int c = 0; c < hs; c++) s += v[c] * dy[c];
        const float dp = w[j - blk] * (s - dsum) * scale;
        const float* k = qkv + (long)j * d3 + d + h * hs;
        float* dk = dqkv + (long)j * d3 + d + h * hs;
        for (int c = 0; c < hs; c++) { atomicAdd(dq + c, dp * k[c]); atomicAdd(dk + c, dp * q[c]); }
    }
}
// Long blocks (K > ATT_GEMM_K, patch tokens): the scores come from batched GEMMs. One warp per (row b, head):
// softmax of the scaled scores in place over keys start[b]..end[b] (end = NULL: ..b) of its block, 0 elsewhere.
#define ATT_GEMM_K 32
__global__ void k_attn_softmax(float* att, const int* start, const int* end, int B, int K, int nh,
                               const unsigned char* allow, int nt) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (i >= B * nh) return;
    const int b = i / nh, blk = (b / K) * K, lo = start[b] - blk, hi = (end ? end[b] : b) - blk;
    const unsigned char* al = allow ? allow + (b % nt) * nt : nullptr;   // attention regions (as k_attn_fwd)
    float* w = att + (long)i * K;
    float mx = -1e30f;
    for (int c = lane; c < K; c += 32)
        if (c >= lo && c <= hi && (!al || al[(blk + c) % nt])) mx = fmaxf(mx, w[c]);
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    float sum = 0.f;
    for (int c = lane; c < K; c += 32) {
        const float e = (c >= lo && c <= hi && (!al || al[(blk + c) % nt])) ? expf(w[c] - mx) : 0.f;
        w[c] = e;
        sum += e;
    }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    const float inv = 1.0f / sum;
    for (int c = lane; c < K; c += 32) w[c] *= inv;
}
// dS = P * (dP - sum_c P dP) * scale, in place in dp ([b, h] rows of K, laid out as att)
__global__ void k_attn_softmax_bwd(float* dp, const float* att, int B, int K, int nh, float scale) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (i >= B * nh) return;
    const float* w = att + (long)i * K;
    float* g = dp + (long)i * K;
    float s = 0.f;
    for (int c = lane; c < K; c += 32) s += w[c] * g[c];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    for (int c = lane; c < K; c += 32) g[c] = w[c] * (g[c] - s) * scale;
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
__device__ inline float drive_weight_(const float* drv, const float* ndrv, float w, float eps) {
    return (w > 0.f && fabsf(ndrv[0] - drv[0]) > eps) ? 1.0f + w : 1.0f;
}
__global__ void k_head_loss(const float* pdrv, const float* ndrv, const float* pi, const float* act,
                            const float* logstd, float* dpdrv, float* dpi, float* glogstd, int B, float inv_B,
                            float drive_coef, float move_coef, double* acc, const float* drv, float dcw, float deps) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    double dse = 0;
    const float wd = drive_weight_(drv + b * 2, ndrv + b * 2, dcw, deps);
    for (int j = 0; j < 2; j++) {
        const float p = pdrv[b * 2 + j], e = p - ndrv[b * 2 + j];
        dse += (double)e * e;
        dpdrv[b * 2 + j] = wd * inv_B * drive_coef * e * p * (1.0f - p);
    }
    atomicAdd(acc + 2, dse / 2.0);
    atomicAdd(acc + 19, wd * dse / 2.0);
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
// dvf = vf > 0 ? dtin[:, :F] + the inverse head's gradient : 0. The head's input at step t is [v_t, v_t1], t1 = t + 1
// (t at the sequence's last step), so v_t receives div_in[t, :F], div_in[t-1, F:] (t > 0), and div_in[t, F:] at the
// last step. div_in [B, 2F] may be NULL.
__global__ void k_dvf(const float* dtin, const float* vf, const float* div_in, float* dvf, int B, int T, int F,
                      int in_dim) {
    const long n = (long)B * F;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const long b = i / F;
        const int f = i % F, t = b % T;
        float g = dtin[b * in_dim + f];
        if (div_in) {
            g += div_in[b * 2 * F + f];
            if (t > 0) g += div_in[(b - 1) * 2 * F + F + f];
            if (t == T - 1) g += div_in[b * 2 * F + F + f];
        }
        dvf[i] = vf[i] > 0.f ? g : 0.f;
    }
}
// Inverse-dynamics input [B, 2F] = [v_t, v_t1]
__global__ void k_inv_in(const float* vf, float* in, int B, int T, int F) {
    const long n = (long)B * 2 * F;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const long b = i / (2 * F);
        const int j = i % (2 * F);
        const long src = j < F ? b : (b % T + 1 < T ? b + 1 : b);
        in[i] = vf[src * F + j % F];
    }
}
// Inverse-dynamics loss where valid[b]: inv_coef * 1/2 |out - clip(a)|^2. acc[10] += SE/2, acc[11] += |a|^2/2,
// acc[12] += valid steps
__global__ void k_inv_loss(const float* out, const float* act, const int* valid, float* dout, int B, float inv_B,
                           float coef, double* acc) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    dout[b * 2] = dout[b * 2 + 1] = 0.f;
    if (!valid[b]) return;
    double se = 0, var = 0;
    for (int k = 0; k < 2; k++) {
        const float tgt = clampf_(act[b * 3 + k], -1.f, 1.f), e = out[b * 2 + k] - tgt;
        se += 0.5 * (double)e * e;
        var += 0.5 * (double)tgt * tgt;
        dout[b * 2 + k] = inv_B * coef * e;
    }
    atomicAdd(acc + 10, se); atomicAdd(acc + 11, var); atomicAdd(acc + 12, 1.0);
}
// Squared gradient norms of the three groups: [0] backbone, [1] policy/value heads [h0, h1), [2] prediction heads
// [p0, p1)
__global__ void k_sqnorm(const float* G, long n, long h0, long h1, long p0, long p1, long p2, double* out) {
    __shared__ double sh[3][256];
    double s[3] = {0, 0, 0};
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const double g = (double)G[i] * G[i];
        s[(i >= h0 && i < h1) ? 1 : (((i >= p0 && i < p1) || i >= p2) ? 2 : 0)] += g;   // p2..: the act token's heads
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

// ---- latent dynamics (dyn_hidden > 0; as vm_agent.cpp's seq_dyn, dyn_step, dyn_decode) ----
// Chain step inputs for rows [0, n) of one level: uT = [s, executed a_j], uH = [s, h, executed a_j], where s is v of
// the chain's start frame (par < 0) or the previous step's latent, and h the true hunger at j (train) or the
// imagined one (imagination: hcur, NULL in training)
__global__ void k_dyn_in(const float* vf, const float* ys, const int* b, const int* j, const int* par,
                         const float* act, const float* drv, const float* hcur, float* uT, float* uH, int n, int F) {
    const long tot = (long)n * (F + 4);
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x) {
        const int r = i / (F + 4), c = i % (F + 4);
        const int jj = j[r];
        const float* a = act + (long)jj * 3;
        const float e0 = clampf_(a[0], -1.f, 1.f), e1 = clampf_(a[1], -1.f, 1.f), e2 = a[2] > 0.5f ? 1.f : 0.f;
        float vu;
        if (c < F) {
            vu = par[r] < 0 ? vf[(long)b[r] * F + c] : ys[(long)par[r] * F + c];
            uT[(long)r * (F + 3) + c] = vu;
        } else if (c == F) {
            vu = hcur ? hcur[r] : drv[(long)jj * 2];
        } else {
            vu = c == F + 1 ? e0 : (c == F + 2 ? e1 : e2);
            uT[(long)r * (F + 3) + c - 1] = vu;
        }
        uH[(long)r * (F + 4) + c] = vu;
    }
}
// s_out = s + (T2 hT + b): s from uT [n, F+3]
__global__ void k_dyn_res(float* sout, const float* b, const float* uT, int n, int F) {
    const long tot = (long)n * F;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x)
        sout[i] += b[i % F] + uT[(i / F) * (F + 3) + i % F];
}
// Latent and hunger losses per chain step p: latent target v of frame lt[p] (-1: none), hunger loss if hl[p].
// acc[13] += SE/F, [14] += 1; acc[15] += eh^2, [16] += 1
__global__ void k_dyn_loss(const float* ys, const float* h, const float* vf, const int* lt, const int* hl,
                           const int* j, const float* ndrv, float* gs, float* gh, int n, int F, float cs, float ch,
                           double* acc, const float* drv, float dcw, float deps) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    float* g = gs + (long)p * F;
    if (lt[p] >= 0) {
        const float* sh = ys + (long)p * F;
        const float* tg = vf + (long)lt[p] * F;
        double se = 0;
        for (int i = 0; i < F; i++) { const float e = sh[i] - tg[i]; se += (double)e * e; g[i] = cs * e; }
        atomicAdd(acc + 13, se / F); atomicAdd(acc + 14, 1.0);
    } else {
        for (int i = 0; i < F; i++) g[i] = 0.f;
    }
    gh[p] = 0.f;
    if (hl[p]) {
        const float hh = h[p], eh = hh - ndrv[(long)j[p] * 2];
        const float wj = drive_weight_(drv + (long)j[p] * 2, ndrv + (long)j[p] * 2, dcw, deps);
        gh[p] = wj * ch * eh * hh * (1.0f - hh);
        atomicAdd(acc + 15, (double)eh * eh); atomicAdd(acc + 16, 1.0); atomicAdd(acc + 20, wj * (double)eh * eh);
    }
}
// Coarse decoder loss: x [n, R] (HWC at r = pimg/2) against src frame fr[p] of u8 frames [*, pimg, pimg, 3] box-
// averaged by 2; gx = cx * e. acc[17] += SE/R, [18] += 1 per row
__global__ void k_dyn_xloss(const float* x, const uint8_t* frames, const int* fr, float* gx, int n, int pimg, float cx,
                            double* acc) {
    __shared__ double sh[256];
    const int r = pimg / 2, R = r * r * 3;
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    double se = 0;
    if (i < (long)n * R) {
        const long p = i / R;
        const int k = i % R, c = k % 3, xx = (k / 3) % r, yy = k / (3 * r);
        const uint8_t* f = frames + (long)(fr ? fr[p] : p) * pimg * pimg * 3;
        const int sum = f[((2 * yy) * pimg + 2 * xx) * 3 + c] + f[((2 * yy) * pimg + 2 * xx + 1) * 3 + c] +
                        f[((2 * yy + 1) * pimg + 2 * xx) * 3 + c] + f[((2 * yy + 1) * pimg + 2 * xx + 1) * 3 + c];
        const float tg = (float)((sum + 2) / 4) / 255.0f, e = x[i] - tg;
        se = (double)e * e / R;
        gx[i] = cx * e;
        if (k == 0) atomicAdd(acc + 18, 1.0);
    }
    sh[threadIdx.x] = se;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) sh[threadIdx.x] += sh[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(acc + 17, sh[0]);
}
// The gradient into each step's previous latent: g_s[par] += g_s[p] + g_uT[p, :F] + g_uH[p, :F] (one child per parent)
__global__ void k_dyn_back(float* gs, const float* guT, const float* guH, const int* par, int p0, int n, int F) {
    const long tot = (long)n * F;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x) {
        const int r = i / F, c = i % F, p = p0 + r;
        if (par[p] < 0) continue;
        gs[(long)par[p] * F + c] += gs[(long)p * F + c] + guT[(long)p * (F + 3) + c] + guH[(long)p * (F + 4) + c];
    }
}

// Imagination step k inputs: uT = [s, executed a_k], uH = [s, h, executed a_k]; actions [n, H, 3]
__global__ void k_im_in(const float* sv, const float* h, const float* act, float* uT, float* uH, int n, int H, int k,
                        int F) {
    const long tot = (long)n * (F + 4);
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x) {
        const int r = i / (F + 4), c = i % (F + 4);
        const float* a = act + ((long)r * H + k) * 3;
        float vu;
        if (c < F) { vu = sv[(long)r * F + c]; uT[(long)r * (F + 3) + c] = vu; }
        else if (c == F) vu = h[r];
        else {
            vu = c == F + 1 ? clampf_(a[0], -1.f, 1.f) : (c == F + 2 ? clampf_(a[1], -1.f, 1.f) : (a[2] > 0.5f ? 1.f : 0.f));
            uT[(long)r * (F + 3) + c - 1] = vu;
        }
        uH[(long)r * (F + 4) + c] = vu;
    }
}
// dst[r * H + k] (rows of width w) = src[r] (rows of width w)
__global__ void k_store_step(float* dst, const float* src, int n, int H, int k, int w) {
    const long tot = (long)n * w;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x)
        dst[((i / w) * H + k) * w + i % w] = src[i];
}

// ---- patch tokens (vm-00010; as seq_fwd_patch / seq_bwd_patch in vm_agent.cpp) ----
// Patch pixels in [0, 1]: tpx [(b*np + p), (y*ps + x)*3 + c]
__global__ void k_patch_px(const uint8_t* img, float* tpx, int B, int S, int ps, int gp) {
    const int np = gp * gp, pd = ps * ps * 3;
    const long n = (long)B * np * pd;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int q = i % pd, c = q % 3, x = (q / 3) % ps, y = q / (3 * ps);
        const long bp = i / pd, b = bp / np;
        const int p = bp % np, py = (p / gp) * ps, px = (p % gp) * ps;
        tpx[i] = img[((b * S + py + y) * S + px + x) * 3 + c] / 255.0f;
    }
}
// Token inputs [B * nt, d]: patches (embedding + pos2d + type 0), drives (type 1), previous move (2), command (3);
// all + the time embedding of the step
__global__ void k_ptok(const float* pemb, const float* pos2d, const float* typ, const float* dw, const float* db,
                       const float* mw, const float* mb, const float* cw, const float* cb, const float* wpe,
                       const int* pos, const float* drv, const float* pa, const float* pr, int use_pr, float* xin,
                       int B, int np, int d, const float* ae) {
    const int nt = np + 3 + (ae ? 1 : 0);   // ae: the act token's vector (NULL: no act token)
    const long n = (long)B * nt * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % d, k = (i / d) % nt;
        const long b = i / ((long)d * nt);
        float v = wpe[(long)pos[b] * d + j];
        if (k < np) v += pemb[(b * np + k) * d + j] + pos2d[(long)k * d + j] + typ[j];
        else if (k == np) v += dw[j * 2] * drv[b * 2] + dw[j * 2 + 1] * drv[b * 2 + 1] + db[j] + typ[d + j];
        else if (k == np + 1) v += mw[j * 3] * pa[b * 3] + mw[j * 3 + 1] * pa[b * 3 + 1] + mw[j * 3 + 2] * pa[b * 3 + 2] + mb[j] + typ[2 * d + j];
        else if (k == np + 2) v += cw[j] * (use_pr ? pr[b] : 0.f) + cb[j] + typ[3 * d + j];
        else v += ae[j] + typ[4 * d + j];
        xin[i] = v;
    }
}
// The act token (task 00020). Coarse frame [b, ACT_CG^2 * 3] (HWC) of prediction-size u8 frames [b, p, p, 3]: block
// averages over f x f, rounded half up (as downsample_u8), scaled to [0, 1]
__device__ inline float coarse_px(const uint8_t* fr, int p, int f, int i) {
    const int c = i % 3, x = (i / 3) % ACT_CG, y = i / (3 * ACT_CG);
    int s = 0;
    for (int u = 0; u < f; u++)
        for (int v = 0; v < f; v++) s += fr[((y * f + u) * p + x * f + v) * 3 + c];
    return (float)((s + (f * f) / 2) / (f * f)) / 255.0f;
}
// frm [B, cfx] += the coarse current frame (the residual)
__global__ void k_act_residual(float* frm, const uint8_t* cs, int B, int p, int f) {
    const int cfx = ACT_CG * ACT_CG * 3;
    const long n = (long)B * cfx;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        frm[i] += coarse_px(cs + (i / cfx) * p * p * 3, p, f, i % cfx);
}
// The act token's losses, one thread per token: next drives (as k_head_loss's, weighted by drive_weight_) and the
// coarse next frame (MSE). acc[21] += wd SE/2, acc[22] += frame SE/cfx, acc[23] += SE/2
__global__ void k_act_loss(const float* adrv, const float* afrm, const float* ndrv, const uint8_t* ns, const float* drv,
                           float* dadrv, float* dafrm, int B, int p, int f, float inv_B, float coef, float dcw,
                           float deps, double* acc) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const int cfx = ACT_CG * ACT_CG * 3;
    const float wd = drive_weight_(drv + b * 2, ndrv + b * 2, dcw, deps);
    double ase = 0, fse = 0;
    for (int j = 0; j < 2; j++) {
        const float q = adrv[b * 2 + j], e = q - ndrv[b * 2 + j];
        ase += (double)e * e;
        dadrv[b * 2 + j] = wd * inv_B * coef * e * q * (1.0f - q);
    }
    const uint8_t* fr = ns + (long)b * p * p * 3;
    for (int i = 0; i < cfx; i++) {
        const float e = afrm[(long)b * cfx + i] - coarse_px(fr, p, f, i);
        fse += (double)e * e;
        dafrm[(long)b * cfx + i] = inv_B * coef * 2.0f * e / cfx;
    }
    atomicAdd(acc + 21, wd * ase / 2.0);
    atomicAdd(acc + 22, fse / cfx);
    atomicAdd(acc + 23, ase / 2.0);
}
// dst [B, d] += scale * src [B, w][:, :d]
__global__ void k_add_cols(float* dst, const float* src, int B, int d, int w, float scale) {
    const long n = (long)B * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        dst[i] += scale * src[(i / d) * w + i % d];
}
// dst[r] (+)= src[b*nt + k] for r = b (rows of width d): the move token's output z, and its gradient back
__global__ void k_tok_rows(float* rows, float* tok, int B, int nt, int k, int d, int to_tok) {
    const long n = (long)B * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const long b = i / d, t = (b * nt + k) * d + i % d;
        if (to_tok) tok[t] += rows[i]; else rows[i] = tok[t];
    }
}
// Prediction rows R = B*(np+1): rows b*np + p (patch p of frame b), then B*np + b (the drives token of frame b);
// pin = [token output, executed action]. back: tok[...] += scale * dpin[r, :d] instead.
__global__ void k_ppin(float* lnf, const float* act, float* pin, int B, int np, int nt, int d, int back, float scale) {
    const int D = d + 3;
    const long R = (long)B * (np + 1), n = R * (back ? d : D);
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int w = back ? d : D, j = i % w;
        const long r = i / w;
        const long b = r < (long)B * np ? r / np : r - (long)B * np;
        const int k = r < (long)B * np ? (int)(r % np) : np;
        const long t = (b * nt + k) * d + j;
        if (back) { lnf[t] += scale * pin[r * D + j]; continue; }
        float v;
        if (j < d) v = lnf[t];
        else if (j < d + 2) v = clampf_(act[b * 3 + j - d], -1.f, 1.f);
        else v = act[b * 3 + 2] > 0.5f ? 1.f : 0.f;
        pin[i] = v;
    }
}
// Cells <-> the predicted frame (channel-major [3, B, P]): cell values [(b*np + p), (y*cs + x)*3 + c]
__global__ void k_cells_frame(float* cells, float* frame, int B, int pimg, int gp, int to_frame) {
    const int cs = pimg / gp, np = gp * gp, cpx = cs * cs * 3, P = pimg * pimg;
    const long n = (long)B * np * cpx;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int q = i % cpx, c = q % 3, x = (q / 3) % cs, y = q / (3 * cs);
        const long bp = i / cpx, b = bp / np;
        const int p = bp % np, oy = (p / gp) * cs, ox = (p % gp) * cs;
        const long f = (long)c * B * P + b * P + (long)(oy + y) * pimg + ox + x;
        if (to_frame) frame[f] = cells[i]; else cells[i] = frame[f];
    }
}
// The token inputs' gradients: time embedding, type, pos2d, the drives / move / command embeddings (atomics), and the
// patch embedding's output gradient dpemb
__global__ void k_ptok_grad(const float* dres, const int* pos, const float* drv, const float* pa, const float* pr,
                            int use_pr, float* gwpe, float* gtyp, float* gpos2d, float* gdw, float* gdb, float* gmw,
                            float* gmb, float* gcw, float* gcb, float* dpemb, int B, int np, int d, float* gae) {
    const int nt = np + 3 + (gae ? 1 : 0);
    const long n = (long)B * nt * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % d, k = (i / d) % nt;
        const long b = i / ((long)d * nt);
        const float g = dres[i];
        atomicAdd(gwpe + (long)pos[b] * d + j, g);
        atomicAdd(gtyp + (long)(k < np ? 0 : k - np + 1) * d + j, g);
        if (k < np) { atomicAdd(gpos2d + (long)k * d + j, g); dpemb[(b * np + k) * d + j] = g; }
        else if (k == np) { atomicAdd(gdw + j * 2, g * drv[b * 2]); atomicAdd(gdw + j * 2 + 1, g * drv[b * 2 + 1]); atomicAdd(gdb + j, g); }
        else if (k == np + 1) {
            for (int q = 0; q < 3; q++) atomicAdd(gmw + j * 3 + q, g * pa[b * 3 + q]);
            atomicAdd(gmb + j, g);
        } else if (k == np + 2) { atomicAdd(gcw + j, g * (use_pr ? pr[b] : 0.f)); atomicAdd(gcb + j, g); }
        else atomicAdd(gae + j, g);
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
struct Buf {   // device float buffer, grown on demand; freed with its owner (not copyable)
    float* p = nullptr; size_t n = 0;
    Buf() = default;
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() { if (p) cudaFree(p); }
    float* get(size_t need) {
        if (need > n) {
            if (p) cudaFree(p);
            n = need;
            if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) {   // the caller's next CUDA call then fails
                fprintf(stderr, "vm_cuda: out of device memory (a buffer of %zu floats)\n", need);
                p = nullptr; n = 0;
            }
        }
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
    int *start = nullptr, *pos = nullptr, *ivalid = nullptr;   // ivalid: t+1 is in the sequence and episode
    float *drv = nullptr, *pa = nullptr, *pr = nullptr, *act = nullptr, *ndrv = nullptr;
    // the universal model's token inputs when they differ from the batch (self-forcing, task 00039; vmc_set_uni_inputs)
    uint8_t* iimg = nullptr;
    float* idrv = nullptr;
    bool has_in = false;
    int cap = 0;   // frames allocated
    // activations and gradients
    Buf x0, cols1, cols2, cols3, Y, a1, a2, a3, vf, tin, xin, ln1, ln1m, ln1r, qkv, att, datt, atty, res1, ln2, ln2m, ln2r,
        fch, fchg, res2, lnf, lnfm, lnfr, pih, pi, pin, ph1, ph2, ub, pdrv, d1in, dc1, d2in, dc2, pout;
    Buf iv_in, iv_h, iv_out, div_out, div_h, div_in;   // inverse dynamics
    // latent dynamics: chain steps of all levels, level k at rows [lev[k-1], lev[k]); y_* forward, g_* gradients
    Buf y_uT, y_uH, y_hT, y_hH, y_h, y_s, y_hX, y_x, y_hX0, y_x0, g_s, g_h, g_hT, g_hH, g_uT, g_uH, g_hX, g_x, g_hX0, g_x0;
    int *ch_b = nullptr, *ch_j = nullptr, *ch_par = nullptr, *ch_lt = nullptr, *ch_hl = nullptr, *ch_fr = nullptr;
    int ch_cap = 0;
    std::vector<uint8_t> es_host;   // the batch's episode starts (chains stop at them)
    std::vector<int> start_host, pos_host;   // the batch's context starts and step positions (per frame)
    Buf im_s, im_s2, im_h, im_uT, im_uH, im_hT, im_hH, im_x, im_hX, im_act;   // imagination
    int *im_idx = nullptr, *im_par = nullptr;
    int im_cap = 0;
    Buf fout, fhwc;   // forward outputs: [B, 4] mu, click p, value; predicted frames HWC
    Buf tpx, pemb, dpemb, zm, dzm, pdel, dpdel;   // patch tokens
    Buf ax_in, ax_h, ax_drv, ax_frm, dax_in, dax_h, dax_drv, dax_frm;   // the act token's auxiliary heads
    Buf u_in, u_tgt, u_log, u_val;   // universal model: token inputs, targets, head logits (then gradients), values
    Buf u_hot, c_hot;                // bin-embedding input: the two-hot inputs of the batch and of the candidates
    Buf u_ce;                        // and each token's cross-entropy in the last update (S x T x ntok, its shape)
    Buf u_mode;                      // vmc_forward_uni_step's most probable values
    int ce_S = 0, ce_T = 0;
    // the instinct's candidates (vmc_uni_candidates): their inputs, activations and outputs, one row per candidate
    Buf c_in, c_x, c_ln, c_lnm, c_lnr, c_qkv, c_att, c_res, c_fch, c_fchg, c_lg, c_val;
    int *tstart = nullptr, *tend = nullptr;      // patch tokens: each token's key range
    unsigned char* allow = nullptr;              // attention regions (task 00044): [allow_nt^2] on the device, or NULL
    int* uslot = nullptr;                        // the universal model's slots [UNI_SLOT * ntok] on the device
    int allow_nt = 1;
    int tcap = 0;
    Buf dpout, dpdrv, dpi, dcols, dx2, dx1, dph2, dub, dskip3, dph1, dpin, dz, dpih, dres, dln, datty, dqkv, dfch, dfchg,
        dtin, dvf, da3, da2, da1, dY;
    double* acc = nullptr;    // [0] weighted frame SE, [1] frame SE, [2] drive SE, [3] -logp, [4] click p, [5..7] norms,
                              // [8] SE of the changed pixels, [9] their count (x3), [10..12] inverse SE, var, count,
                              // [13..18] latent dynamics: latent SE/F, n; hunger SE, n; decoder SE/R, n;
                              // [19], [20] the drive and hunger SE weighted by drive_changed_weight
    float* clip = nullptr;
    float coef[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // pred, drive, move, inverse, dyn latent, hunger, decoder, act token
};

static void gemm(Model* M, bool ta, bool tb, int m, int n, int k, const float* A, int lda, const float* B, int ldb,
                 float beta, float* C, int ldc) {
    // row-major C[m,n] = op(A) op(B) + beta C  <=>  column-major C^T = op(B)^T op(A)^T
    const float one = 1.0f;
    cublasSgemm(M->cb, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, &one, B, ldb, A, lda,
                &beta, C, ldc);
}
// gemm over `batch` independent problems, the i-th at A + i sA, B + i sB, C + i sC (row-major, as gemm)
static void gemm_sb(Model* M, bool ta, bool tb, int m, int n, int k, float alpha, const float* A, int lda, long sA,
                    const float* B, int ldb, long sB, float beta, float* C, int ldc, long sC, int batch) {
    cublasSgemmStridedBatched(M->cb, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, &alpha,
                              B, ldb, sB, A, lda, sA, &beta, C, ldc, sC, batch);
}
// Attention of B rows in blocks of K (row b of block s attends to rows start[b]..end[b] of it). Short blocks: one
// thread per (row, head). Long blocks: per head, scores = Q K^T and Y = P V as GEMMs batched over the blocks, with a
// masked softmax in between; att holds the weights of row b, head h at [(b nh + h) K + key - block start] as in
// k_attn_fwd (0 outside the row's keys).
static void attention_fwd(Model* M, const float* qkv, float* att, float* y, const int* start, const int* end, int B,
                          int K) {
    const Dims& D = M->D;
    const int d = D.d, nh = D.nh, hs = D.hs, ns = B / K;
    if (K <= ATT_GEMM_K) {
        k_attn_fwd<<<(B * nh + 127) / 128, 128>>>(qkv, att, y, start, end, B, K, d, nh, M->allow, M->allow_nt);
        return;
    }
    const float scale = 1.0f / sqrtf((float)hs);
    for (int h = 0; h < nh; h++)
        gemm_sb(M, false, true, K, K, hs, scale, qkv + h * hs, 3 * d, (long)K * 3 * d, qkv + d + h * hs, 3 * d,
                (long)K * 3 * d, 0.f, att + (long)h * K, nh * K, (long)K * nh * K, ns);
    k_attn_softmax<<<(unsigned)(((long)B * nh * 32 + 255) / 256), 256>>>(att, start, end, B, K, nh, M->allow,
                                                                          M->allow_nt);
    for (int h = 0; h < nh; h++)
        gemm_sb(M, false, false, K, hs, K, 1.f, att + (long)h * K, nh * K, (long)K * nh * K, qkv + 2 * d + h * hs,
                3 * d, (long)K * 3 * d, 0.f, y + h * hs, d, (long)K * d, ns);
}
// Backward of attention_fwd: dqkv (zeroed first) from dy
static void attention_bwd(Model* M, float* dqkv, const float* dy, const float* qkv, const float* att, const int* start,
                          const int* end, int B, int K) {
    const Dims& D = M->D;
    const int d = D.d, nh = D.nh, hs = D.hs, ns = B / K;
    cudaMemsetAsync(dqkv, 0, sizeof(float) * B * 3 * d);
    if (K <= ATT_GEMM_K) {
        k_attn_bwd<<<(B * nh + 127) / 128, 128>>>(dqkv, dy, qkv, att, start, end, B, K, d, nh);
        return;
    }
    float* dp = M->datt.get((size_t)B * nh * K);
    const long sq = (long)K * 3 * d, sa = (long)K * nh * K;
    for (int h = 0; h < nh; h++) {   // dP = dY V^T; dV = P^T dY
        gemm_sb(M, false, true, K, K, hs, 1.f, dy + h * hs, d, (long)K * d, qkv + 2 * d + h * hs, 3 * d, sq, 0.f,
                dp + (long)h * K, nh * K, sa, ns);
        gemm_sb(M, true, false, K, hs, K, 1.f, att + (long)h * K, nh * K, sa, dy + h * hs, d, (long)K * d, 0.f,
                dqkv + 2 * d + h * hs, 3 * d, sq, ns);
    }
    k_attn_softmax_bwd<<<(unsigned)(((long)B * nh * 32 + 255) / 256), 256>>>(dp, att, B, K, nh,
                                                                            1.0f / sqrtf((float)hs));
    for (int h = 0; h < nh; h++) {   // dQ = dS K; dK = dS^T Q
        gemm_sb(M, false, false, K, hs, K, 1.f, dp + (long)h * K, nh * K, sa, qkv + d + h * hs, 3 * d, sq, 0.f,
                dqkv + h * hs, 3 * d, sq, ns);
        gemm_sb(M, true, false, K, hs, K, 1.f, dp + (long)h * K, nh * K, sa, qkv + h * hs, 3 * d, sq, 0.f,
                dqkv + d + h * hs, 3 * d, sq, ns);
    }
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
    k_colsum<<<dim3((OC + 31) / 32, (T + COLSUM_ROWS - 1) / COLSUM_ROWS), dim3(32, 8)>>>(dout, T, OC, db);
}

static void alloc_batch(Model* M, int B) {
    const Dims& D = M->D;
    if (B > M->cap) {
        cudaFree(M->img); cudaFree(M->ns); cudaFree(M->cs); cudaFree(M->nfull); cudaFree(M->start); cudaFree(M->pos); cudaFree(M->ivalid);
        cudaFree(M->drv); cudaFree(M->pa); cudaFree(M->pr); cudaFree(M->act); cudaFree(M->ndrv);
        cudaFree(M->iimg); cudaFree(M->idrv);
        cudaMalloc(&M->iimg, (size_t)B * D.img * D.img * 3); cudaMalloc(&M->idrv, B * 2 * sizeof(float));
        cudaMalloc(&M->img, (size_t)B * D.img * D.img * 3);
        cudaMalloc(&M->ns, (size_t)B * D.small_px);
        cudaMalloc(&M->cs, (size_t)B * D.small_px);
        cudaMalloc(&M->nfull, (size_t)B * D.img * D.img * 3);
        cudaMalloc(&M->start, B * sizeof(int)); cudaMalloc(&M->pos, B * sizeof(int)); cudaMalloc(&M->ivalid, B * sizeof(int));
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
    if (const int nt = vm_attn_allow(cfg, nullptr)) {   // attention regions (task 00044)
        std::vector<unsigned char> al((size_t)nt * nt);
        vm_attn_allow(cfg, al.data());
        if (cudaMalloc(&M->allow, al.size()) ||
            cudaMemcpy(M->allow, al.data(), al.size(), cudaMemcpyHostToDevice)) {
            fprintf(stderr, "vm_cuda: out of device memory\n"); return nullptr;
        }
        M->allow_nt = nt;
    }
    if (D.uni) {   // the slot layout (tasks 00047, 00048)
        std::vector<VMSlot> sl(D.ntok);
        vm_uni_layout(cfg, sl.data());
        std::vector<int> ki(UNI_SLOT * D.ntok, 0);
        for (int k = 0; k < D.ntok; k++) {
            int* e = ki.data() + UNI_SLOT * k;
            e[0] = sl[k].kind; e[1] = sl[k].idx;
            if (sl[k].kind == 2 && cfg->sum_thumb) vm_uni_block(cfg, &sl[k], e + 2, e + 3, e + 4, e + 5);
        }
        if (cudaMalloc(&M->uslot, ki.size() * sizeof(int)) ||
            cudaMemcpy(M->uslot, ki.data(), ki.size() * sizeof(int), cudaMemcpyHostToDevice)) {
            fprintf(stderr, "vm_cuda: out of device memory\n"); return nullptr;
        }
    }
    for (int i = 0; i < NUM_P; i++) { M->off[i] = offsets[i]; M->sz[i] = sizes[i]; }
    const size_t nb = n_params * sizeof(float);
    if (cudaMalloc(&M->P, nb) || cudaMalloc(&M->G, nb) || cudaMalloc(&M->m, nb) || cudaMalloc(&M->v, nb) ||
        cudaMalloc(&M->acc, ACC_N * sizeof(double)) || cudaMalloc(&M->clip, 2 * sizeof(float))) {
        fprintf(stderr, "vm_cuda: out of device memory\n"); return nullptr;
    }
    cudaMemcpy(M->P, params, nb, cudaMemcpyHostToDevice);
    cudaMemset(M->G, 0, nb); cudaMemset(M->m, 0, nb); cudaMemset(M->v, 0, nb); cudaMemset(M->acc, 0, ACC_N * sizeof(double));
    if (cublasCreate(&M->cb) != CUBLAS_STATUS_SUCCESS) return nullptr;
    cublasSetMathMode(M->cb, CUBLAS_PEDANTIC_MATH);   // no TF32
    return M;
}

long vmc_device_free(void) {
    size_t fr = 0, tot = 0;
    return cudaMemGetInfo(&fr, &tot) == cudaSuccess ? (long)fr : -1;
}

void vmc_free(void* h) {
    Model* M = (Model*)h;
    if (!M) return;
    cudaFree(M->P); cudaFree(M->G); cudaFree(M->m); cudaFree(M->v); cudaFree(M->acc); cudaFree(M->clip);
    cudaFree(M->iimg); cudaFree(M->idrv);
    cudaFree(M->img); cudaFree(M->ns); cudaFree(M->cs); cudaFree(M->nfull); cudaFree(M->start); cudaFree(M->pos); cudaFree(M->ivalid);
    cudaFree(M->drv); cudaFree(M->pa); cudaFree(M->pr); cudaFree(M->act); cudaFree(M->ndrv);
    // every Buf member frees itself in delete M (they used to be listed here, and new ones were missed)
    for (int* q : {M->ch_b, M->ch_j, M->ch_par, M->ch_lt, M->ch_hl, M->ch_fr, M->im_idx, M->im_par, M->tstart, M->tend})
        cudaFree(q);
    if (M->allow) cudaFree(M->allow);
    if (M->uslot) cudaFree(M->uslot);
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
    M->has_in = false;
    std::vector<int> start(B), pos(B), valid(B);
    for (int s = 0; s < S; s++)
        for (int t = 0; t < T; t++) {
            const int b = s * T + t;
            const bool st = t == 0 || M->cfg.one_frame || (ep_start && ep_start[b]);
            start[b] = st ? b : start[b - 1];
            pos[b] = b - start[b];
            valid[b] = t + 1 < T && !(ep_start && ep_start[b + 1]);   // inverse dynamics (one_frame aside)
        }
    CK(cudaMemcpy(M->ivalid, valid.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    M->es_host.assign(B, 0);
    if (ep_start) for (int b = 0; b < B; b++) M->es_host[b] = ep_start[b];
    M->start_host = start; M->pos_host = pos;
    if (D.patch) {   // a token sees every token of its own step and of the earlier steps of its block
        const int nt = D.ntok, Bx = B * nt;
        std::vector<int> ts(Bx), te(Bx);
        for (int b = 0; b < B; b++)
            for (int k = 0; k < nt; k++) {   // universal model: fully causal (up to the token itself)
            ts[b * nt + k] = start[b] * nt;
            te[b * nt + k] = D.uni ? b * nt + k : b * nt + nt - 1;
        }
        if (Bx > M->tcap) {
            cudaFree(M->tstart); cudaFree(M->tend);
            CK(cudaMalloc(&M->tstart, Bx * sizeof(int))); CK(cudaMalloc(&M->tend, Bx * sizeof(int)));
            M->tcap = Bx;
        }
        CK(cudaMemcpy(M->tstart, ts.data(), Bx * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(M->tend, te.data(), Bx * sizeof(int), cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(M->img, images, (size_t)B * D.img * D.img * 3, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->start, start.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->pos, pos.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->drv, drives, B * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->pa, prev_action, B * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->pr, prev_reward, B * sizeof(float), cudaMemcpyHostToDevice));
    if (action) CK(cudaMemcpy(M->act, action, B * 3 * sizeof(float), cudaMemcpyHostToDevice));
    else CK(cudaMemset(M->act, 0, B * 3 * sizeof(float)));
    if (!D.uni) k_downsample<<<GRID((long)B * D.small_px)>>>(M->img, M->cs, B, D.img);   // universal: no frame targets
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
    if (D.uni || !next_small) {   // the universal model's targets come from the window and the next drives
    } else if (next_full) {   // full-size next frames: downsampled here
        CK(cudaMemcpy(M->nfull, next_small, (size_t)B * D.img * D.img * 3, cudaMemcpyHostToDevice));
        k_downsample<<<GRID((long)B * D.small_px)>>>(M->nfull, M->ns, B, D.img);
    } else {
        CK(cudaMemcpy(M->ns, next_small, (size_t)B * D.small_px, cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(M->ndrv, next_drives, B * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaGetLastError());
    return 0;
}

int vmc_set_uni_inputs(void* h, int S, const uint8_t* images, const float* drives) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (!D.uni || !M->B || S != M->S || !images || !drives) return -1;
    const int B = M->B;
    CK(cudaMemcpy(M->iimg, images, (size_t)B * D.img * D.img * 3, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->idrv, drives, B * 2 * sizeof(float), cudaMemcpyHostToDevice));
    M->has_in = true;
    return 0;
}

}   // extern "C"

// The encoder of B frames (M->img) -> the conv maps and v (M->vf).
static void encoder_fwd(Model* M, int B) {
    const Dims& D = M->D;
    const float* P = M->P;
    const long* off = M->off;
    const int P1 = D.h1 * D.h1, P2 = D.h2 * D.h2, P3 = D.h3 * D.h3;
    float* x0 = M->x0.get((size_t)B * 5 * D.img * D.img);
    float* a1 = M->a1.get((size_t)B * D.c1 * P1);
    float* a2 = M->a2.get((size_t)B * D.c2 * P2);
    float* a3 = M->a3.get((size_t)B * D.enc_flat);
    float* cols1 = M->cols1.get((size_t)5 * KK * B * P1);
    float* cols2 = M->cols2.get((size_t)D.c1 * KK * B * P2);
    float* cols3 = M->cols3.get((size_t)D.c2 * KK * B * P3);
    float* Y = M->Y.get((size_t)std::max(D.c1 * P1, std::max(D.c2 * P2, D.c3 * P3)) * B);
    float* vf = M->vf.get((size_t)B * D.feat);
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
    linear(M, vf, a3, P + off[EFW], P + off[EFB], B, D.enc_flat, D.feat, 1);
}

static int grow_ints(int** p, int* cap, int need, int which) {
    (void)which;
    if (need <= *cap) return 0;
    cudaFree(*p);
    return cudaMalloc(p, (size_t)need * sizeof(int)) == cudaSuccess ? 0 : -1;
}

// Latent dynamics losses and their gradients (heads only: v is detached), as seq_dyn in vm_agent.cpp, for all the
// sequences of the batch at once. Chains of imagined steps start at every frame t; step k uses the action of frame
// j = t + k - 1 and stops at the horizon, at the sequence's end and at an episode start.
static int dyn_losses(Model* M, const PPOConfig& pc, float inv_B) {
    const Dims& D = M->D;
    const int T = M->T, B = M->B, F = D.feat, Hd = D.dh, R = D.rpx, UT = F + 3, UH = F + 4;
    const float* P = M->P;
    float* G = M->G;
    const long* off = M->off;
    const int Hmax = std::min(pc.dyn_horizon, T);
    M->coef[4] = pc.dyn_coef; M->coef[5] = pc.dyn_h_coef; M->coef[6] = pc.dyn_x_coef;
    std::vector<int> hb, hj, hp, hlt, hhl, lev = {0};
    for (int k = 1; k <= Hmax; k++) {
        const int begin = (int)hb.size();
        if (k == 1) {
            for (int b = 0; b < B; b++) { hb.push_back(b); hj.push_back(b); hp.push_back(-1); }
        } else {
            for (int q = lev[k - 2]; q < lev[k - 1]; q++) {
                const int b = hb[q], jl = b % T + k - 1, jf = b + k - 1;
                if (jl > T - 1 || M->es_host[jf]) continue;    // past the sequence, or step j-1 ended the episode
                hb.push_back(b); hj.push_back(jf); hp.push_back(q);
            }
        }
        if ((int)hb.size() == begin) break;
        lev.push_back((int)hb.size());
    }
    const int np = (int)hb.size(), nl = (int)lev.size() - 1;
    for (int p = 0; p < np; p++) {
        const int jf = hj[p], jl = jf % T;
        int k = 1;
        while (lev[k] <= p) k++;
        hlt.push_back(jl + 1 <= T - 1 && !M->es_host[jf + 1] ? jf + 1 : -1);
        hhl.push_back(pc.dyn_h_horizon <= 0 || k <= pc.dyn_h_horizon);
    }
    int cap = M->ch_cap;
    if (grow_ints(&M->ch_b, &cap, np, 0) || grow_ints(&M->ch_j, &cap, np, 1) || grow_ints(&M->ch_par, &cap, np, 2) ||
        grow_ints(&M->ch_lt, &cap, np, 3) || grow_ints(&M->ch_hl, &cap, np, 4)) return -1;
    M->ch_cap = std::max(M->ch_cap, np);
    CK(cudaMemcpy(M->ch_b, hb.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->ch_j, hj.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->ch_par, hp.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->ch_lt, hlt.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(M->ch_hl, hhl.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    float* vf = M->vf.p;
    float* uT = M->y_uT.get((size_t)np * UT); float* uH = M->y_uH.get((size_t)np * UH);
    float* hT = M->y_hT.get((size_t)np * Hd); float* hH = M->y_hH.get((size_t)np * Hd);
    float* h = M->y_h.get(np); float* ys = M->y_s.get((size_t)np * F);
    float* hX = M->y_hX.get((size_t)np * Hd); float* x = M->y_x.get((size_t)np * R);
    float* hX0 = M->y_hX0.get((size_t)B * Hd); float* x0 = M->y_x0.get((size_t)B * R);
    for (int k = 1; k <= nl; k++) {   // level by level: each depends on the previous one
        const int p0 = lev[k - 1], n = lev[k] - p0;
        k_dyn_in<<<GRID((long)n * UH)>>>(vf, ys, M->ch_b + p0, M->ch_j + p0, M->ch_par + p0, M->act, M->drv, nullptr,
                                          uT + (size_t)p0 * UT, uH + (size_t)p0 * UH, n, F);
        linear(M, hH + (size_t)p0 * Hd, uH + (size_t)p0 * UH, P + off[H1W], P + off[H1B], n, UH, Hd, 1);
        linear(M, h + p0, hH + (size_t)p0 * Hd, P + off[H2W], P + off[H2B], n, Hd, 1, 2);
        linear(M, hT + (size_t)p0 * Hd, uT + (size_t)p0 * UT, P + off[T1W], P + off[T1B], n, UT, Hd, 1);
        gemm(M, false, true, n, F, Hd, hT + (size_t)p0 * Hd, Hd, P + off[T2W], Hd, 0.f, ys + (size_t)p0 * F, F);
        k_dyn_res<<<GRID((long)n * F)>>>(ys + (size_t)p0 * F, P + off[T2B], uT + (size_t)p0 * UT, n, F);
    }
    linear(M, hX, ys, P + off[X1W], P + off[X1B], np, F, Hd, 1);
    linear(M, x, hX, P + off[X2W], P + off[X2B], np, Hd, R, 0);
    linear(M, hX0, vf, P + off[X1W], P + off[X1B], B, F, Hd, 1);
    linear(M, x0, hX0, P + off[X2W], P + off[X2B], B, Hd, R, 0);
    float* gs = M->g_s.get((size_t)np * F); float* gh = M->g_h.get(np);
    float* ghT = M->g_hT.get((size_t)np * Hd); float* ghH = M->g_hH.get((size_t)np * Hd);
    float* guT = M->g_uT.get((size_t)np * UT); float* guH = M->g_uH.get((size_t)np * UH);
    float* ghX = M->g_hX.get((size_t)np * Hd); float* gx = M->g_x.get((size_t)np * R);
    float* ghX0 = M->g_hX0.get((size_t)B * Hd); float* gx0 = M->g_x0.get((size_t)B * R);
    const float cs = inv_B * pc.dyn_coef * 2.0f / F, chh = inv_B * pc.dyn_h_coef * 2.0f, cx = inv_B * pc.dyn_x_coef * 2.0f / R;
    k_dyn_loss<<<(np + 127) / 128, 128>>>(ys, h, vf, M->ch_lt, M->ch_hl, M->ch_j, M->ndrv, gs, gh, np, F, cs, chh, M->acc,
                                          M->drv, pc.drive_changed_weight, pc.drive_change_eps);
    k_dyn_xloss<<<(unsigned)(((long)np * R + 255) / 256), 256>>>(x, M->ns, M->ch_j, gx, np, D.pimg, cx, M->acc);
    k_dyn_xloss<<<(unsigned)(((long)B * R + 255) / 256), 256>>>(x0, M->cs, nullptr, gx0, B, D.pimg, cx, M->acc);
    // backward: the decoders, then the chains from the last level down
    linear_bwd(M, gx, hX, P + off[X2W], np, Hd, R, ghX, false, G + off[X2W], G + off[X2B]);
    k_mask<<<GRID((long)np * Hd)>>>(ghX, hX, (long)np * Hd);
    linear_bwd(M, ghX, ys, P + off[X1W], np, F, Hd, gs, true, G + off[X1W], G + off[X1B]);
    linear_bwd(M, gx0, hX0, P + off[X2W], B, Hd, R, ghX0, false, G + off[X2W], G + off[X2B]);
    k_mask<<<GRID((long)B * Hd)>>>(ghX0, hX0, (long)B * Hd);
    linear_bwd(M, ghX0, vf, P + off[X1W], B, F, Hd, nullptr, false, G + off[X1W], G + off[X1B]);
    for (int k = nl; k >= 1; k--) {
        const int p0 = lev[k - 1], n = lev[k] - p0;
        linear_bwd(M, gh + p0, hH + (size_t)p0 * Hd, P + off[H2W], n, Hd, 1, ghH + (size_t)p0 * Hd, false,
                   G + off[H2W], G + off[H2B]);
        k_mask<<<GRID((long)n * Hd)>>>(ghH + (size_t)p0 * Hd, hH + (size_t)p0 * Hd, (long)n * Hd);
        linear_bwd(M, ghH + (size_t)p0 * Hd, uH + (size_t)p0 * UH, P + off[H1W], n, UH, Hd, guH + (size_t)p0 * UH, false,
                   G + off[H1W], G + off[H1B]);
        linear_bwd(M, gs + (size_t)p0 * F, hT + (size_t)p0 * Hd, P + off[T2W], n, Hd, F, ghT + (size_t)p0 * Hd, false,
                   G + off[T2W], G + off[T2B]);
        k_mask<<<GRID((long)n * Hd)>>>(ghT + (size_t)p0 * Hd, hT + (size_t)p0 * Hd, (long)n * Hd);
        linear_bwd(M, ghT + (size_t)p0 * Hd, uT + (size_t)p0 * UT, P + off[T1W], n, UT, Hd, guT + (size_t)p0 * UT, false,
                   G + off[T1W], G + off[T1B]);
        k_dyn_back<<<GRID((long)n * F)>>>(gs, guT, guH, M->ch_par, p0, n, F);
    }
    CK(cudaGetLastError());
    return 0;
}

// The Transformer layers and final LayerNorm on B token rows M->xin -> M->lnf. Row b attends to rows start[b]..end[b]
// (end = NULL: ..b), all within its block of K rows.
static void transformer_fwd(Model* M, int B, int K, const int* start, const int* end) {
    const Dims& D = M->D;
    const int d = D.d, L = D.L;
    const float* P = M->P;
    const long* off = M->off;
    const float* xin = M->xin.p;
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
        attention_fwd(M, qkv + bd * 3, att + (size_t)l * B * D.nh * K, atty + bd, start, end, B, K);
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
}

// Backward of transformer_fwd: from M->dz (the gradient at M->lnf) to M->dres (at M->xin); parameter gradients into G.
static int transformer_bwd(Model* M, int B, int K, const int* start, const int* end) {
    const Dims& D = M->D;
    const int d = D.d, L = D.L;
    const float* P = M->P;
    float* G = M->G;
    const long* off = M->off;
    const int lnb = (B + 7) / 8;
    const float *xin = M->xin.p, *ln1 = M->ln1.p, *ln1m = M->ln1m.p, *ln1r = M->ln1r.p, *qkv = M->qkv.p, *att = M->att.p,
                *atty = M->atty.p, *res1 = M->res1.p, *ln2 = M->ln2.p, *ln2m = M->ln2m.p, *ln2r = M->ln2r.p,
                *fch = M->fch.p, *fchg = M->fchg.p, *res2 = M->res2.p, *lnfm = M->lnfm.p, *lnfr = M->lnfr.p, *dz = M->dz.p;
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
        attention_bwd(M, dqkv, datty, qkv + bd * 3, att + (size_t)l * B * D.nh * K, start, end, B, K);
        linear_bwd(M, dqkv, ln1 + bd, P + off[QKVW] + (size_t)l * 3 * d * d, B, d, 3 * d, dln, false,
                   G + off[QKVW] + (size_t)l * 3 * d * d, G + off[QKVB] + l * 3 * d);
        k_ln_bwd<<<lnb, 256>>>(dres, G + off[LN1W] + l * d, G + off[LN1B] + l * d, dln, xl, P + off[LN1W] + l * d,
                               ln1m + l * B, ln1r + l * B, B, d);
    }
    return 0;
}

// ---- the universal model (task 00023; as seq_fwd_uni / seq_loss_uni / seq_bwd_uni in vm_agent.cpp) ----
// Token (b, k) of frame b: its slot k from sl [UNI_SLOT * nt] (vm_uni_layout, tasks 00047-00049): kind (0 the body,
// 1 a patch, 2 a summary, 3 the move), index, and a summary's block (origin y, x and pooling fy, fx; fy = 0: no
// thumbnail). Values in [0, 1]; used: 2, V, V or 0 (a summary), 3.
__device__ inline int uni_used_(int k, int V, const int* sl) {
    const int kind = sl[UNI_SLOT * k];
    return kind == 0 ? 2 : (kind == 3 ? 3 : (kind == 2 ? (sl[UNI_SLOT * k + 4] ? V : 0) : V));
}
__device__ inline float uni_value_(const uint8_t* img, const float* drv, const float* act, long b, int k, int j, int nt,
                                   int V, int S, int ps, int gp, int ch, const int* sl) {
    const int* e = sl + UNI_SLOT * k;
    const int kind = e[0], p = e[1];
    if (kind == 2) {   // a summary: its block box-averaged to one patch (task 00049), or no values
        const int fy = e[4], fx = e[5];
        if (!fy) return 0.f;
        const int c = j % ch, x = (j / ch) % ps, y = j / (ch * ps), y0 = e[2] + y * fy, x0 = e[3] + x * fx;
        float a = 0.f;
        for (int i = 0; i < fy; i++)
            for (int q = 0; q < fx; q++) a += img[((b * S + y0 + i) * S + x0 + q) * 3 + c];
        return a / (255.0f * fy * fx);
    }
    if (k == 0) return j < 2 ? drv[b * 2 + j] : 0.f;
    if (k == nt - 1) {
        if (j < 2) return 0.5f * (clampf_(act[b * 3 + j], -1.f, 1.f) + 1.f);
        return j == 2 ? (act[b * 3 + 2] > 0.5f ? 1.f : 0.f) : 0.f;
    }
    const int c = j % ch, x = (j / ch) % ps, y = j / (ch * ps), py = (p / gp) * ps, px = (p % gp) * ps;
    return img[((b * S + py + y) * S + px + x) * 3 + c] / 255.0f;
}
// in [B*nt, V]: the centred inputs (unused 0), from iimg / idrv; tgt [B*nt, V]: the next token's values (the move:
// the next drives), from img / drv. The two differ only with self-forcing (task 00039).
__global__ void k_uni_tokens(const uint8_t* iimg, const float* idrv, const uint8_t* img, const float* drv,
                             const float* act, const float* ndrv, float* in, float* tgt, int B, int nt, int V, int S,
                             int ps, int gp, int ch, const int* sl) {
    const long n = (long)B * nt * V;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % V, k = (i / V) % nt;
        const long b = i / ((long)V * nt);
        const float v = uni_value_(iimg, idrv, act, b, k, j, nt, V, S, ps, gp, ch, sl);
        in[i] = j < uni_used_(k, V, sl) ? v - 0.5f : 0.f;
        if (k < nt - 1) tgt[i] = uni_value_(img, drv, act, b, k + 1, j, nt, V, S, ps, gp, ch, sl);
        else tgt[i] = (ndrv && j < 2) ? ndrv[b * 2 + j] : 0.f;
    }
}
// Bin-embedding input (task 00036): hot [rows, V*K] = each used value of in [rows, V] (centred) split two-hot over
// the K bins, unused values and bins 0. Row r is token r % nt (cand: every row a move token, nt - 1).
__global__ void k_uni_twohot(const float* in, float* hot, long rows, int nt, int V, int K, int cand, const int* sl) {
    const long n = rows * V;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % V;
        const long r = i / V;
        const int k = cand ? nt - 1 : (int)(r % nt);
        float* o = hot + i * K;
        for (int b = 0; b < K; b++) o[b] = 0.f;
        if (j >= uni_used_(k, V, sl)) continue;
        const float p = fminf(fmaxf(in[i] + 0.5f, 0.f), 1.f) * (K - 1);
        const int lo = min((int)floorf(p), K - 2);
        const float w = p - lo;
        o[lo] = 1.f - w;
        o[lo + 1] = w;
    }
}
// xin [B*nt, d] += slot[k] + wpe[pos[b]] (back: their gradients from x = dres, atomics)
__global__ void k_uni_embed(float* x, const float* slot, const float* wpe, const int* pos, int B, int nt, int d,
                            int back, float* gslot, float* gwpe) {
    const long n = (long)B * nt * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % d, k = (i / d) % nt;
        const long b = i / ((long)d * nt);
        if (back) { atomicAdd(gslot + (long)k * d + j, x[i]); atomicAdd(gwpe + (long)pos[b] * d + j, x[i]); }
        else x[i] += slot[(long)k * d + j] + wpe[(long)pos[b] * d + j];
    }
}
// One block per token row, one warp per value (a lane per bin): the softmax of each used value's K logits, the two-hot
// cross-entropy of the next token's values, and in place the logits' gradient (p - target) * inv_tok / used (0 for
// unused values). Accumulators: acc[24 + kind] += token CE, acc[27 + kind] += token squared error of the expected
// value, acc[30 + kind] += 1, acc[33] += token CE; kind 0 the next body (move tokens), 1 a patch, 2 the move (the last
// patch). A token whose next token is a summary (task 00047) has no target: gradient 0, not counted. Launch with
// UNI_LOSS_THREADS threads (a multiple of 32).
#define UNI_LOSS_THREADS 256
__global__ void k_uni_loss(float* lg, const float* tgt, int rows, int nt, int V, int K, float inv_tok, double* acc,
                           float* tok_ce, const int* sl) {
    __shared__ double sh[2][UNI_LOSS_THREADS / 32];
    const int r = blockIdx.x, k = r % nt, lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int u = k == nt - 1 ? 2 : uni_used_(k + 1, V, sl), kind = k == nt - 1 ? 0 : (k == nt - 2 ? 2 : 1);
    if (u == 0) {   // no target
        for (long i = threadIdx.x; i < (long)V * K; i += blockDim.x) lg[(long)r * V * K + i] = 0.f;
        if (threadIdx.x == 0) tok_ce[r] = 0.f;
        return;
    }
    double ce = 0, se = 0;   // lane 0 of each warp
    for (int v = w; v < V; v += nw) {
        float* l = lg + ((long)r * V + v) * K;
        if (v >= u) { for (int b = lane; b < K; b += 32) l[b] = 0.f; continue; }
        float mx = -1e30f;
        for (int b = lane; b < K; b += 32) mx = fmaxf(mx, l[b]);
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        float sum = 0.f;
        for (int b = lane; b < K; b += 32) { const float e = expf(l[b] - mx); l[b] = e; sum += e; }
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
        const float inv = 1.0f / sum, t = fminf(fmaxf(tgt[(long)r * V + v], 0.f), 1.f), pos = t * (K - 1);
        int lo = (int)pos;
        if (lo > K - 2) lo = K - 2;
        const float whi = pos - lo, wlo = 1.0f - whi, sc = inv_tok / u;
        float ev = 0.f, c = 0.f;
        for (int b = lane; b < K; b += 32) {
            const float p = l[b] * inv, tb = b == lo ? wlo : (b == lo + 1 ? whi : 0.f);
            ev += p * (float)b / (K - 1);
            if (b == lo || b == lo + 1) c -= tb * logf(fmaxf(p, 1e-30f));
            l[b] = (p - tb) * sc;
        }
        for (int o = 16; o; o >>= 1) { ev += __shfl_xor_sync(0xffffffff, ev, o); c += __shfl_xor_sync(0xffffffff, c, o); }
        if (lane == 0) { ce += c; se += (double)(ev - t) * (ev - t); }
    }
    if (lane == 0) { sh[0][w] = ce; sh[1][w] = se; }
    __syncthreads();
    if (threadIdx.x == 0) {
        double a0 = 0, a1 = 0;
        for (int i = 0; i < nw; i++) { a0 += sh[0][i]; a1 += sh[1][i]; }
        atomicAdd(acc + 24 + kind, a0 / u); atomicAdd(acc + 27 + kind, a1 / u);
        atomicAdd(acc + 30 + kind, 1.0); atomicAdd(acc + 33, a0 / u);
        tok_ce[r] = (float)(a0 / u);
    }
}
// The instinct's candidates: one query row per candidate (its own q, k, v in qc [R, 3d]) attending to the cached keys
// lo[r]..hi[r]-1 of one layer's qkv [rows, 3d] (the window's earlier tokens) and to itself. One warp per (row, head):
// each lane runs an online softmax over every 32nd key, then the lanes' partial results are merged (head size <= 64).
__global__ void k_attn_decode(const float* qc, const float* qkv, const int* lo, const int* hi, float* out, int R, int d,
                              int nh) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (i >= R * nh) return;
    const int r = i / nh, h = i % nh, hs = d / nh, d3 = 3 * d;
    const float scale = 1.0f / sqrtf((float)hs);
    const float* q = qc + (long)r * d3 + h * hs;
    float acc[64];
    for (int c = 0; c < hs; c++) acc[c] = 0.f;
    float m = -1e30f, l = 0.f;
    for (int j = lo[r] + lane; j <= hi[r]; j += 32) {   // hi[r]: the candidate itself
        const float* kv = j < hi[r] ? qkv + (long)j * d3 : qc + (long)r * d3;
        float s = 0.f;
        for (int c = 0; c < hs; c++) s += q[c] * kv[d + h * hs + c];
        s *= scale;
        const float mn = fmaxf(m, s), a = expf(m - mn), e = expf(s - mn);
        for (int c = 0; c < hs; c++) acc[c] = acc[c] * a + e * kv[2 * d + h * hs + c];
        l = l * a + e;
        m = mn;
    }
    float M = m;   // merge the lanes: rescale each to the warp's max
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
    const float f = l > 0.f ? expf(m - M) : 0.f;
    float L = l * f;
    for (int o = 16; o; o >>= 1) L += __shfl_xor_sync(0xffffffff, L, o);
    float* y = out + (long)r * d + h * hs;
    for (int c = 0; c < hs; c++) {
        float v = acc[c] * f;
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) y[c] = v / L;
    }
}
// x [R, d] += slot row + wpe[pos[r]]
__global__ void k_cand_embed(float* x, const float* slot, const float* wpe, const int* pos, int R, int d) {
    const long n = (long)R * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        x[i] += slot[i % d] + wpe[(long)pos[i / d] * d + i % d];
}
// out [B*2, d]: the outputs of the last patch and the move token of each frame (the move and next-body predictions)
__global__ void k_uni_last2(const float* lnf, float* out, int B, int nt, int d) {
    const long n = (long)B * 2 * d;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const int j = i % d, q = (i / d) % 2;
        const long b = i / (2L * d);
        out[i] = lnf[(b * nt + nt - 2 + q) * d + j];
    }
}
// Forward only: per (row, value) the softmax in place and the expected value val [rows, V]
__global__ void k_uni_probs(float* lg, float* val, long n, int K) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        float* l = lg + i * K;
        float mx = l[0];
        for (int b = 1; b < K; b++) mx = fmaxf(mx, l[b]);
        float sum = 0.f;
        for (int b = 0; b < K; b++) { l[b] = expf(l[b] - mx); sum += l[b]; }
        const float inv = 1.0f / sum;
        float ev = 0.f;
        for (int b = 0; b < K; b++) { l[b] *= inv; ev += l[b] * (float)b / (K - 1); }
        val[i] = ev;
    }
}

// Step t's values of S sequences of T frames, sharp (task 00029): prob [S*T*ntV, K] (k_uni_probs' probabilities) ->
// out [S, ntV]. mode 1: the most probable bin's centre. mode 2 ("pair", task 00038): the most probable bin and its more
// probable neighbour, their probability-weighted centre (a two-hot value decodes exactly; distinct values still choose).
__global__ void k_uni_mode_step(const float* prob, float* out, int S, int T, int t, int ntV, int K, int mode) {
    const long n = (long)S * ntV;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const long s = i / ntV, j = i % ntV;
        const float* p = prob + ((s * T + t) * (long)ntV + j) * K;
        int best = 0;
        for (int b = 1; b < K; b++) if (p[b] > p[best]) best = b;
        if (mode == 2) {
            const int nb = best == 0 ? 1 : best == K - 1 ? K - 2 : (p[best + 1] > p[best - 1] ? best + 1 : best - 1);
            const float w = p[best] + p[nb];
            out[i] = w > 0.f ? (p[best] * best + p[nb] * nb) / (w * (K - 1)) : (float)best / (K - 1);
        } else {
            out[i] = (float)best / (K - 1);
        }
    }
}

// Patch tokens (D.patch > 0): the forward pass, and with train the losses and gradients (as run_model for the CNN
// model; the frame, drives and move losses only).
static int run_patch(Model* M, const PPOConfig* pcp, bool train, bool want_pred) {
    const Dims& D = M->D;
    const int B = M->B, T = M->T, d = D.d, np = D.npch, nt = D.ntok, Bx = B * nt, KT = T * nt, sp = D.pimg * D.pimg;
    const float* P = M->P;
    float* G = M->G;
    const long* off = M->off;
    float* tpx = M->tpx.get((size_t)B * np * D.pdim);
    float* pemb = M->pemb.get((size_t)B * np * d);
    float* xin = M->xin.get((size_t)Bx * d);
    k_patch_px<<<GRID((long)B * np * D.pdim)>>>(M->img, tpx, B, D.img, D.patch, D.gp);
    linear(M, pemb, tpx, P + off[PEW], P + off[PEB], B * np, D.pdim, d, 0);
    k_ptok<<<GRID((long)Bx * d)>>>(pemb, P + off[POS2D], P + off[TYP], P + off[DEW], P + off[DEB], P + off[MEW],
                                    P + off[MEB], P + off[CEW], P + off[CEB], P + off[WPE], M->pos, M->drv, M->pa, M->pr,
                                    M->cfg.use_prev_reward, xin, B, np, d, D.act ? P + off[ACTE] : nullptr);
    transformer_fwd(M, Bx, KT, M->tstart, M->tend);
    float* lnf = M->lnf.p;
    float* zm = M->zm.get((size_t)B * d);
    const int zi = D.act ? np + 3 : np + 1;   // the token the policy reads: the act token, else the move token
    k_tok_rows<<<GRID((long)B * d)>>>(zm, lnf, B, nt, zi, d, 0);
    float* pi = M->pi.get((size_t)B * 3);
    float* pih = D.pih ? M->pih.get((size_t)B * D.pih) : nullptr;
    if (D.pih) {
        linear(M, pih, zm, P + off[PHW], P + off[PHB], B, d, D.pih, 1);
        linear(M, pi, pih, P + off[PIW], P + off[PIB], B, D.pih, 3, 0);
    } else {
        linear(M, pi, zm, P + off[PIW], P + off[PIB], B, d, 3, 0);
    }
    const long R = (long)B * (np + 1);
    float* pin = M->pin.get((size_t)R * D.pin_dim);
    float* ph1 = M->ph1.get((size_t)R * D.ph);
    float* pdel = M->pdel.get((size_t)B * np * D.cpx);
    float* pdrv = M->pdrv.get((size_t)B * 2);
    float* pout = M->pout.get((size_t)3 * B * sp);
    if (want_pred) {
        k_ppin<<<GRID(R * D.pin_dim)>>>(lnf, M->act, pin, B, np, nt, d, 0, 0.f);
        linear(M, ph1, pin, P + off[PR1W], P + off[PR1B], (int)R, D.pin_dim, D.ph, 1);
        linear(M, pdel, ph1, P + off[PR2W], P + off[PR2B], B * np, D.ph, D.cpx, 0);
        linear(M, pdrv, ph1 + (size_t)B * np * D.ph, P + off[DRW], P + off[DRB], B, D.ph, 2, 2);
        k_cells_frame<<<GRID((long)B * np * D.cpx)>>>(pdel, pout, B, D.pimg, D.gp, 1);
        k_residual<<<GRID(3L * B * sp)>>>(pout, M->cs, B, sp);
    }
    float *ax_in = nullptr, *ax_h = nullptr, *ax_drv = nullptr, *ax_frm = nullptr;
    if (want_pred && D.act) {   // the act token's auxiliary heads on [its output, executed move]
        ax_in = M->ax_in.get((size_t)B * D.pin_dim); ax_h = M->ax_h.get((size_t)B * D.ph);
        ax_drv = M->ax_drv.get((size_t)B * 2); ax_frm = M->ax_frm.get((size_t)B * D.cfx);
        k_pin<<<GRID((long)B * D.pin_dim)>>>(zm, M->act, ax_in, B, d);
        linear(M, ax_h, ax_in, P + off[AX1W], P + off[AX1B], B, D.pin_dim, D.ph, 1);
        linear(M, ax_drv, ax_h, P + off[AXDW], P + off[AXDB], B, D.ph, 2, 2);
        linear(M, ax_frm, ax_h, P + off[AXFW], P + off[AXFB], B, D.ph, D.cfx, 0);
        k_act_residual<<<GRID((long)B * D.cfx)>>>(ax_frm, M->cs, B, D.pimg, D.cds);
    }
    if (!train) { CK(cudaGetLastError()); return 0; }
    const PPOConfig& pc = *pcp;
    CK(cudaMemsetAsync(G, 0, M->n_params * sizeof(float)));
    CK(cudaMemsetAsync(M->acc, 0, ACC_N * sizeof(double)));
    M->coef[0] = pc.pred_coef; M->coef[1] = pc.drive_coef; M->coef[2] = pc.move_coef;
    M->coef[3] = M->coef[4] = M->coef[5] = M->coef[6] = 0.f;
    M->coef[7] = D.act ? pc.act_coef : 0.f;
    const float inv_B = 1.0f / B;
    float* dpout = M->dpout.get((size_t)3 * B * sp);
    float* dpdrv = M->dpdrv.get((size_t)B * 2);
    float* dpi = M->dpi.get((size_t)B * 3);
    const bool have_pred = pc.pred_coef > 0.f || pc.drive_coef > 0.f;
    k_frame_loss<<<(B * sp + 255) / 256, 256>>>(pout, M->cs, M->ns, dpout, B, sp,
                                                 inv_B * pc.pred_coef * 2.0f / D.small_px, 1.0f + pc.pred_changed_weight, M->acc);
    k_head_loss<<<(B + 127) / 128, 128>>>(pdrv, M->ndrv, pi, M->act, P + off[LOGSTD], dpdrv, dpi, G + off[LOGSTD], B,
                                          inv_B, pc.drive_coef, pc.move_coef, M->acc, M->drv, pc.drive_changed_weight,
                                          pc.drive_change_eps);
    // backward: the heads into the token outputs dz [Bx, d]
    float* dz = M->dz.get((size_t)Bx * d);
    CK(cudaMemsetAsync(dz, 0, sizeof(float) * Bx * d));
    float* dzm = M->dzm.get((size_t)B * d);
    CK(cudaMemsetAsync(dzm, 0, sizeof(float) * B * d));
    float* dzm_pi = pc.pi_detach ? nullptr : dzm;
    if (D.pih) {
        float* dpih = M->dpih.get((size_t)B * D.pih);
        linear_bwd(M, dpi, pih, P + off[PIW], B, D.pih, 3, dpih, false, G + off[PIW], G + off[PIB]);
        k_mask<<<GRID((long)B * D.pih)>>>(dpih, pih, (long)B * D.pih);
        linear_bwd(M, dpih, zm, P + off[PHW], B, d, D.pih, dzm_pi, true, G + off[PHW], G + off[PHB]);
    } else {
        linear_bwd(M, dpi, zm, P + off[PIW], B, d, 3, dzm_pi, true, G + off[PIW], G + off[PIB]);
    }
    const float sc = pc.pred_backbone_scale;
    if (have_pred && D.act) {   // the act token's losses -> its heads -> its output (with the prediction scale)
        float* dax_drv = M->dax_drv.get((size_t)B * 2);
        float* dax_frm = M->dax_frm.get((size_t)B * D.cfx);
        float* dax_h = M->dax_h.get((size_t)B * D.ph);
        float* dax_in = M->dax_in.get((size_t)B * D.pin_dim);
        k_act_loss<<<(B + 127) / 128, 128>>>(ax_drv, ax_frm, M->ndrv, M->ns, M->drv, dax_drv, dax_frm, B, D.pimg, D.cds,
                                             inv_B, pc.act_coef, pc.drive_changed_weight, pc.drive_change_eps, M->acc);
        linear_bwd(M, dax_drv, ax_h, P + off[AXDW], B, D.ph, 2, dax_h, false, G + off[AXDW], G + off[AXDB]);
        linear_bwd(M, dax_frm, ax_h, P + off[AXFW], B, D.ph, D.cfx, dax_h, true, G + off[AXFW], G + off[AXFB]);
        k_mask<<<GRID((long)B * D.ph)>>>(dax_h, ax_h, (long)B * D.ph);
        linear_bwd(M, dax_h, ax_in, P + off[AX1W], B, D.pin_dim, D.ph, dax_in, false, G + off[AX1W], G + off[AX1B]);
        if (sc != 0.f) k_add_cols<<<GRID((long)B * d)>>>(dzm, dax_in, B, d, D.pin_dim, sc);
    }
    k_tok_rows<<<GRID((long)B * d)>>>(dzm, dz, B, nt, zi, d, 1);
    if (have_pred) {
        float* dpdel = M->dpdel.get((size_t)B * np * D.cpx);
        float* dph1 = M->dph1.get((size_t)R * D.ph);
        float* dpin = M->dpin.get((size_t)R * D.pin_dim);
        k_cells_frame<<<GRID((long)B * np * D.cpx)>>>(dpdel, dpout, B, D.pimg, D.gp, 0);
        linear_bwd(M, dpdel, ph1, P + off[PR2W], B * np, D.ph, D.cpx, dph1, false, G + off[PR2W], G + off[PR2B]);
        linear_bwd(M, dpdrv, ph1 + (size_t)B * np * D.ph, P + off[DRW], B, D.ph, 2, dph1 + (size_t)B * np * D.ph, false,
                   G + off[DRW], G + off[DRB]);
        k_mask<<<GRID(R * D.ph)>>>(dph1, ph1, R * D.ph);
        linear_bwd(M, dph1, pin, P + off[PR1W], (int)R, D.pin_dim, D.ph, dpin, false, G + off[PR1W], G + off[PR1B]);
        if (sc != 0.f) k_ppin<<<GRID(R * d)>>>(dz, nullptr, dpin, B, np, nt, d, 1, sc);
    }
    if (transformer_bwd(M, Bx, KT, M->tstart, M->tend)) return -1;
    float* dpemb = M->dpemb.get((size_t)B * np * d);
    k_ptok_grad<<<GRID((long)Bx * d)>>>(M->dres.p, M->pos, M->drv, M->pa, M->pr, M->cfg.use_prev_reward, G + off[WPE],
                                         G + off[TYP], G + off[POS2D], G + off[DEW], G + off[DEB], G + off[MEW],
                                         G + off[MEB], G + off[CEW], G + off[CEB], dpemb, B, np, d,
                                         D.act ? G + off[ACTE] : nullptr);
    linear_bwd(M, dpemb, tpx, P + off[PEW], B * np, D.pdim, d, nullptr, false, G + off[PEW], G + off[PEB]);
    CK(cudaGetLastError());
    return 0;
}

// The universal model: tokens, the Transformer, the head; with train the loss and every gradient; else the
// probabilities and expected values (M->u_log, M->u_val) of every token, or (all = false) only of each frame's last
// patch and move token, rows 2b and 2b + 1 (the move and next-body predictions: 1/33 of the head's work and memory).
static int run_uni(Model* M, const PPOConfig* pcp, bool train, bool all = true) {
    const Dims& D = M->D;
    const int B = M->B, T = M->T, d = D.d, nt = D.ntok, V = D.V, K = D.nbin, rows = B * nt, KT = T * nt;
    const float* P = M->P;
    float* G = M->G;
    const long* off = M->off;
    float* in = M->u_in.get((size_t)rows * V);
    float* tgt = M->u_tgt.get((size_t)rows * V);
    float* xin = M->xin.get((size_t)rows * d);
    k_uni_tokens<<<GRID((long)rows * V)>>>(M->has_in ? M->iimg : M->img, M->has_in ? M->idrv : M->drv, M->img, M->drv,
                                         M->act, train ? M->ndrv : nullptr, in, tgt, B, nt, V, D.img, D.patch, D.gp,
                                         D.ch, M->uslot);
    const float* uin = in;   // W_in's input: the centred values, or their two-hot bins
    if (D.ib) {
        float* hot = M->u_hot.get((size_t)rows * D.VI);
        k_uni_twohot<<<GRID((long)rows * V)>>>(in, hot, rows, nt, V, K, 0, M->uslot);
        uin = hot;
    }
    linear(M, xin, uin, P + off[UIW], P + off[UIB], rows, D.VI, d, 0);
    k_uni_embed<<<GRID((long)rows * d)>>>(xin, P + off[USL], P + off[WPE], M->pos, B, nt, d, 0, nullptr, nullptr);
    transformer_fwd(M, rows, KT, M->tstart, M->tend);
    float* lnf = M->lnf.p;
    if (!train && !all) {
        float* z2 = M->zm.get((size_t)B * 2 * d);
        k_uni_last2<<<GRID((long)B * 2 * d)>>>(lnf, z2, B, nt, d);
        float* lg2 = M->u_log.get((size_t)B * 2 * D.OV);
        linear(M, lg2, z2, P + off[UOW], P + off[UOB], B * 2, d, D.OV, 0);
        k_uni_probs<<<GRID((long)B * 2 * V)>>>(lg2, M->u_val.get((size_t)B * 2 * V), (long)B * 2 * V, K);
        CK(cudaGetLastError());
        return 0;
    }
    float* lg = M->u_log.get((size_t)rows * D.OV);
    linear(M, lg, lnf, P + off[UOW], P + off[UOB], rows, d, D.OV, 0);
    if (!train) {
        k_uni_probs<<<GRID((long)rows * V)>>>(lg, M->u_val.get((size_t)rows * V), (long)rows * V, K);
        CK(cudaGetLastError());
        return 0;
    }
    CK(cudaMemsetAsync(G, 0, M->n_params * sizeof(float)));
    CK(cudaMemsetAsync(M->acc, 0, ACC_N * sizeof(double)));
    for (int i = 0; i < 8; i++) M->coef[i] = 0.f;
    k_uni_loss<<<rows, UNI_LOSS_THREADS>>>(lg, tgt, rows, nt, V, K, 1.0f / ((float)B * (nt - D.nnot)), M->acc,
                                           M->u_ce.get(rows), M->uslot);   // the mean over the tokens with a target
    M->ce_S = M->S;
    M->ce_T = T;
    float* dz = M->dz.get((size_t)rows * d);
    linear_bwd(M, lg, lnf, P + off[UOW], rows, d, D.OV, dz, false, G + off[UOW], G + off[UOB]);
    if (transformer_bwd(M, rows, KT, M->tstart, M->tend)) return -1;
    float* dres = M->dres.p;
    linear_bwd(M, dres, uin, P + off[UIW], rows, D.VI, d, nullptr, false, G + off[UIW], G + off[UIB]);
    k_uni_embed<<<GRID((long)rows * d)>>>(dres, nullptr, nullptr, M->pos, B, nt, d, 1, G + off[USL], G + off[WPE]);
    CK(cudaGetLastError());
    return 0;
}

// The forward pass of the uploaded batch (train = false: up to the outputs; the prediction head and decoder only
// if want_pred), or the forward pass, the losses and the gradients (train = true).
static int run_model(Model* M, const PPOConfig* pcp, bool train, bool want_pred) {
    if (!M->B) return -1;
    if (M->D.uni) return run_uni(M, pcp, train);
    if (M->D.patch) return run_patch(M, pcp, train, want_pred);
    const Dims& D = M->D;
    const int B = M->B, K = M->T, d = D.d;   // K: the batch's sequence length (ctx when training)
    const float* P = M->P;
    float* G = M->G;
    const long* off = M->off;
    const int P1 = D.h1 * D.h1, P2 = D.h2 * D.h2, P3 = D.h3 * D.h3, q0 = D.p0 * D.p0, q1 = D.p1 * D.p1;
    const int sp = D.pimg * D.pimg;

    // ---- encoder (all frames) ----
    encoder_fwd(M, B);
    float *a1 = M->a1.p, *a2 = M->a2.p, *a3 = M->a3.p, *cols1 = M->cols1.p, *cols2 = M->cols2.p, *cols3 = M->cols3.p;

    // ---- tokens and Transformer ----
    float* vf = M->vf.p;
    float* tin = M->tin.get((size_t)B * D.in_dim);
    float* xin = M->xin.get((size_t)B * d);
    k_token<<<GRID((long)B * D.in_dim)>>>(vf, M->drv, M->pa, M->pr, tin, B, D.feat, M->cfg.use_prev_reward);
    gemm(M, false, true, B, d, D.in_dim, tin, D.in_dim, P + off[INW], D.in_dim, 0.f, xin, d);
    k_bias_wpe<<<GRID((long)B * d)>>>(xin, P + off[INB], P + off[WPE], M->pos, B, d);
    transformer_fwd(M, B, K, M->start, nullptr);
    float* lnf = M->lnf.p;

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
    CK(cudaMemsetAsync(M->acc, 0, ACC_N * sizeof(double)));
    const bool have_inv = pc.inv_coef > 0.f && D.ih > 0;
    M->coef[0] = pc.pred_coef; M->coef[1] = pc.drive_coef; M->coef[2] = pc.move_coef; M->coef[3] = have_inv ? pc.inv_coef : 0.f;

    // ---- losses ----
    const float inv_B = 1.0f / B;
    float* dpout = M->dpout.get((size_t)3 * B * sp);
    float* dpdrv = M->dpdrv.get((size_t)B * 2);
    float* dpi = M->dpi.get((size_t)B * 3);
    const bool have_pred = pc.pred_coef > 0.f || pc.drive_coef > 0.f;
    k_frame_loss<<<(B * sp + 255) / 256, 256>>>(pout, M->cs, M->ns, dpout, B, sp,
                                                 inv_B * pc.pred_coef * 2.0f / D.small_px, 1.0f + pc.pred_changed_weight, M->acc);
    k_head_loss<<<(B + 127) / 128, 128>>>(pdrv, M->ndrv, pi, M->act, P + off[LOGSTD], dpdrv, dpi, G + off[LOGSTD], B,
                                          inv_B, pc.drive_coef, pc.move_coef, M->acc, M->drv, pc.drive_changed_weight,
                                          pc.drive_change_eps);
    float *iv_in = nullptr, *iv_h = nullptr, *div_out = nullptr;
    if (have_inv) {   // inverse dynamics: [v_t, v_t+1] -> the executed (dx, dy)
        const int F = D.feat;
        iv_in = M->iv_in.get((size_t)B * 2 * F);
        iv_h = M->iv_h.get((size_t)B * D.ih);
        float* iv_out = M->iv_out.get((size_t)B * 2);
        div_out = M->div_out.get((size_t)B * 2);
        k_inv_in<<<GRID((long)B * 2 * F)>>>(vf, iv_in, B, K, F);
        linear(M, iv_h, iv_in, P + off[IV1W], P + off[IV1B], B, 2 * F, D.ih, 1);
        linear(M, iv_out, iv_h, P + off[IV2W], P + off[IV2B], B, D.ih, 2, 0);
        k_inv_loss<<<(B + 127) / 128, 128>>>(iv_out, M->act, M->ivalid, div_out, B, inv_B, pc.inv_coef, M->acc);
    }
    const bool have_dyn = D.dh > 0 && pc.dyn_horizon > 0 && (pc.dyn_coef > 0.f || pc.dyn_h_coef > 0.f || pc.dyn_x_coef > 0.f);
    if (have_dyn && dyn_losses(M, pc, inv_B)) return -1;
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
    if (transformer_bwd(M, B, K, M->start, nullptr)) return -1;
    float* dres = M->dres.p;
    k_wpe_grad<<<GRID((long)B * d)>>>(dres, M->pos, G + off[WPE], B, d);
    float* dtin = M->dtin.get((size_t)B * D.in_dim);
    float* dvf = M->dvf.get((size_t)B * D.feat);
    float* da3 = M->da3.get((size_t)B * D.enc_flat);
    linear_bwd(M, dres, tin, P + off[INW], B, D.in_dim, d, dtin, false, G + off[INW], G + off[INB]);
    float* div_in = nullptr;
    if (have_inv) {
        float* div_h = M->div_h.get((size_t)B * D.ih);
        div_in = M->div_in.get((size_t)B * 2 * D.feat);
        linear_bwd(M, div_out, iv_h, P + off[IV2W], B, D.ih, 2, div_h, false, G + off[IV2W], G + off[IV2B]);
        k_mask<<<GRID((long)B * D.ih)>>>(div_h, iv_h, (long)B * D.ih);
        linear_bwd(M, div_h, iv_in, P + off[IV1W], B, 2 * D.feat, D.ih, div_in, false, G + off[IV1W], G + off[IV1B]);
    }
    k_dvf<<<GRID((long)B * D.feat)>>>(dtin, vf, div_in, dvf, B, K, D.feat, D.in_dim);
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
    if (pc.use_ppo) {
        fprintf(stderr, "vm_cuda: only the prediction-only update (use_ppo=0)\n");
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
    const float* zsrc = D.patch ? M->zm.p : M->lnf.p;   // z: the move token's output with patch tokens
    k_fwd_out<<<(B + 127) / 128, 128>>>(M->pi.p, zsrc, M->P + M->off[VW], M->P + M->off[VB], o, B, D.d);
    std::vector<float> ho((size_t)B * 4);
    CK(cudaMemcpy(ho.data(), o, ho.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int b = 0; b < B; b++) {
        if (mu) { mu[2 * b] = ho[4 * b]; mu[2 * b + 1] = ho[4 * b + 1]; }
        if (click_p) click_p[b] = ho[4 * b + 2];
        if (value) value[b] = ho[4 * b + 3];
    }
    if (z) CK(cudaMemcpy(z, zsrc, (size_t)B * D.d * sizeof(float), cudaMemcpyDeviceToHost));
    if (want_pred && pred_frames) {
        float* f = M->fhwc.get((size_t)B * 3 * sp);
        k_to_hwc<<<GRID(3L * B * sp)>>>(M->pout.p, f, B, sp);
        CK(cudaMemcpy(pred_frames, f, (size_t)B * 3 * sp * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (want_pred && pred_drives) CK(cudaMemcpy(pred_drives, M->pdrv.p, (size_t)B * 2 * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_forward_uni(void* h, int S, int T, const uint8_t* images, const float* drives, const float* actions,
                    const uint8_t* ep_start, float* values, float* move_probs, float* next_body) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (!D.uni) return -1;
    std::vector<float> zero((size_t)S * T, 0.f);
    if (upload_inputs(M, S, T, images, drives, actions, zero.data(), ep_start, actions)) return -1;
    const bool all = values != nullptr;
    if (run_uni(M, nullptr, false, all)) return -1;
    const int nt = D.ntok, V = D.V, K = D.nbin, B = M->B;
    const int stride = all ? nt : 2, mrow = all ? nt - 2 : 0, brow = all ? nt - 1 : 1;   // rows per frame; which
    if (values) CK(cudaMemcpy(values, M->u_val.p, (size_t)B * nt * V * sizeof(float), cudaMemcpyDeviceToHost));
    if (move_probs)   // the last patch of each frame predicts the move: its first 3 values
        CK(cudaMemcpy2D(move_probs, 3 * K * sizeof(float), M->u_log.p + ((size_t)mrow * V) * K,
                        (size_t)stride * V * K * sizeof(float), 3 * K * sizeof(float), B, cudaMemcpyDeviceToHost));
    if (next_body)   // the move token's expected next drives: its first 2 values
        CK(cudaMemcpy2D(next_body, 2 * sizeof(float), M->u_val.p + (size_t)brow * V, (size_t)stride * V * sizeof(float),
                        2 * sizeof(float), B, cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_forward_uni_step(void* h, int S, int T, const uint8_t* images, const float* drives, const float* actions,
                         const uint8_t* ep_start, int t, int mode, float* values_t, float* next_body) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (!D.uni || t < 0 || t >= T || mode < 0 || mode > 2) return -1;
    std::vector<float> zero((size_t)S * T, 0.f);
    if (upload_inputs(M, S, T, images, drives, actions, zero.data(), ep_start, actions)) return -1;
    if (run_uni(M, nullptr, false, true)) return -1;
    const int nt = D.ntok, V = D.V, B = M->B;
    const size_t row = (size_t)nt * V * sizeof(float);
    if (mode) {
        float* mv = M->u_mode.get((size_t)S * nt * V);
        k_uni_mode_step<<<GRID((long)S * nt * V)>>>(M->u_log.p, mv, S, T, t, nt * V, D.nbin, mode);
        CK(cudaGetLastError());
        CK(cudaMemcpy(values_t, mv, (size_t)S * row, cudaMemcpyDeviceToHost));
    } else {
        CK(cudaMemcpy2D(values_t, row, M->u_val.p + (size_t)t * nt * V, (size_t)T * row, row, S,
                        cudaMemcpyDeviceToHost));
    }
    if (next_body)
        CK(cudaMemcpy2D(next_body, 2 * sizeof(float), M->u_val.p + (size_t)(nt - 1) * V, row, 2 * sizeof(float), B,
                        cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_uni_token_ce(void* h, int S0, float* out) {
    Model* M = (Model*)h;
    const int nt = M->D.ntok, T = M->ce_T;
    if (!M->D.uni || !M->ce_S || S0 < 1 || S0 > M->ce_S) return -1;
    CK(cudaMemcpy2D(out, nt * sizeof(float), M->u_ce.p + (size_t)(T - 1) * nt, (size_t)T * nt * sizeof(float),
                    nt * sizeof(float), S0, cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_uni_candidates(void* h, int Mc, const float* cand, float* hunger) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (!D.uni || !M->B || D.hs > 64 || (int)M->start_host.size() != M->B) return -1;
    const int S = M->S, T = M->T, B = M->B, nt = D.ntok, V = D.V, d = D.d, L = D.L, rows = B * nt, R = S * Mc;
    const float* P = M->P;
    const long* off = M->off;
    // inputs: each candidate's move values, centred; its key range and step position (the last step of its window)
    std::vector<float> in((size_t)R * V, 0.f);
    std::vector<int> lo(R), hi(R), pos(R);
    for (int s = 0; s < S; s++) {
        const int b = s * T + T - 1;   // the window's last frame; its move token is row b * nt + nt - 1
        for (int m = 0; m < Mc; m++) {
            const int r = s * Mc + m;
            const float* a = cand + (size_t)r * 3;
            in[(size_t)r * V + 0] = 0.5f * (fminf(fmaxf(a[0], -1.f), 1.f) + 1.f) - 0.5f;
            in[(size_t)r * V + 1] = 0.5f * (fminf(fmaxf(a[1], -1.f), 1.f) + 1.f) - 0.5f;
            in[(size_t)r * V + 2] = (a[2] > 0.5f ? 1.f : 0.f) - 0.5f;
            lo[r] = M->start_host[b] * nt;
            hi[r] = b * nt + nt - 1;
            pos[r] = M->pos_host[b];
        }
    }
    float* cin = M->c_in.get((size_t)R * V);
    CK(cudaMemcpy(cin, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
    int* ip = (int*)M->c_val.get((size_t)R * 3 + (size_t)R * V);   // the int arrays, then the values
    CK(cudaMemcpy(ip, lo.data(), R * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ip + R, hi.data(), R * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ip + 2 * R, pos.data(), R * sizeof(int), cudaMemcpyHostToDevice));
    float* val = (float*)(ip + 3 * R);
    float* x = M->c_x.get((size_t)R * d);
    float* ln = M->c_ln.get((size_t)R * d);
    float* lnm = M->c_lnm.get(R);
    float* lnr = M->c_lnr.get(R);
    float* qc = M->c_qkv.get((size_t)R * 3 * d);
    float* att = M->c_att.get((size_t)R * d);
    float* res = M->c_res.get((size_t)R * d);
    float* fch = M->c_fch.get((size_t)R * D.dff);
    float* fchg = M->c_fchg.get((size_t)R * D.dff);
    const float* cuin = cin;
    if (D.ib) {
        float* hot = M->c_hot.get((size_t)R * D.VI);
        k_uni_twohot<<<GRID((long)R * V)>>>(cin, hot, R, nt, V, D.nbin, 1, M->uslot);
        cuin = hot;
    }
    linear(M, x, cuin, P + off[UIW], P + off[UIB], R, D.VI, d, 0);
    k_cand_embed<<<GRID((long)R * d)>>>(x, P + off[USL] + (long)(nt - 1) * d, P + off[WPE], ip + 2 * R, R, d);
    const int lnb = (R + 7) / 8;
    for (int l = 0; l < L; l++) {   // as transformer_fwd, for the candidate rows, against the cached keys and values
        k_ln_fwd<<<lnb, 256>>>(ln, lnm, lnr, x, P + off[LN1W] + l * d, P + off[LN1B] + l * d, R, d);
        linear(M, qc, ln, P + off[QKVW] + (size_t)l * 3 * d * d, P + off[QKVB] + l * 3 * d, R, d, 3 * d, 0);
        k_attn_decode<<<(R * D.nh * 32 + 127) / 128, 128>>>(qc, M->qkv.p + (size_t)l * rows * 3 * d, ip, ip + R, att, R, d,
                                                            D.nh);
        cudaMemcpyAsync(res, x, sizeof(float) * R * d, cudaMemcpyDeviceToDevice);
        gemm(M, false, true, R, d, d, att, d, P + off[APW] + (size_t)l * d * d, d, 1.f, res, d);
        k_bias_act<<<GRID((long)R * d)>>>(res, P + off[APB] + l * d, R, d, 0);
        k_ln_fwd<<<lnb, 256>>>(ln, lnm, lnr, res, P + off[LN2W] + l * d, P + off[LN2B] + l * d, R, d);
        linear(M, fch, ln, P + off[FCW] + (size_t)l * D.dff * d, P + off[FCB] + l * D.dff, R, d, D.dff, 0);
        k_gelu_fwd<<<GRID((long)R * D.dff)>>>(fchg, fch, (long)R * D.dff);
        cudaMemcpyAsync(x, res, sizeof(float) * R * d, cudaMemcpyDeviceToDevice);
        gemm(M, false, true, R, d, D.dff, fchg, D.dff, P + off[FPW] + (size_t)l * d * D.dff, D.dff, 1.f, x, d);
        k_bias_act<<<GRID((long)R * d)>>>(x, P + off[FPB] + l * d, R, d, 0);
    }
    k_ln_fwd<<<lnb, 256>>>(ln, lnm, lnr, x, P + off[LNFW], P + off[LNFB], R, d);
    float* lg = M->c_lg.get((size_t)R * D.OV);
    linear(M, lg, ln, P + off[UOW], P + off[UOB], R, d, D.OV, 0);
    k_uni_probs<<<GRID((long)R * V)>>>(lg, val, (long)R * V, D.nbin);
    CK(cudaMemcpy2D(hunger, sizeof(float), val, (size_t)V * sizeof(float), sizeof(float), R, cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_encode(void* h, int n, const uint8_t* images, float* vf_out) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (n < 1 || D.patch) return -1;
    alloc_batch(M, n);
    CK(cudaMemcpy(M->img, images, (size_t)n * D.img * D.img * 3, cudaMemcpyHostToDevice));
    M->B = 0;   // the uploaded batch is gone
    encoder_fwd(M, n);
    CK(cudaMemcpy(vf_out, M->vf.p, (size_t)n * D.feat * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_imagine(void* h, int n, int H, const float* s0, const float* h0, const float* actions, float* hunger,
                float* latents, float* frames) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (!D.dh || n < 1 || H < 1) return -1;
    const int F = D.feat, Hd = D.dh, R = D.rpx, UT = F + 3, UH = F + 4;
    const float* P = M->P;
    const long* off = M->off;
    float* sv = M->im_s.get((size_t)n * F); float* s2 = M->im_s2.get((size_t)n * F);
    float* hh = M->im_h.get((size_t)n * (H + 1));
    float* act = M->im_act.get((size_t)n * H * 3);
    float* uT = M->im_uT.get((size_t)n * UT); float* uH = M->im_uH.get((size_t)n * UH);
    float* hT = M->im_hT.get((size_t)n * Hd); float* hHb = M->im_hH.get((size_t)n * Hd);
    float* hX = M->im_hX.get((size_t)n * Hd);
    float* lat = latents ? M->y_s.get((size_t)n * H * F) : nullptr;     // reuse training buffers as outputs
    float* fr = frames ? M->y_x.get((size_t)n * H * R) : nullptr;
    float* xx = frames ? M->im_x.get((size_t)n * R) : nullptr;
    float* hout = M->y_h.get((size_t)n * H);
    CK(cudaMemcpy(sv, s0, (size_t)n * F * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(hh, h0, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(act, actions, (size_t)n * H * 3 * sizeof(float), cudaMemcpyHostToDevice));
    for (int k = 0; k < H; k++) {
        k_im_in<<<GRID((long)n * UH)>>>(sv, hh, act, uT, uH, n, H, k, F);
        linear(M, hHb, uH, P + off[H1W], P + off[H1B], n, UH, Hd, 1);
        linear(M, hh, hHb, P + off[H2W], P + off[H2B], n, Hd, 1, 2);
        linear(M, hT, uT, P + off[T1W], P + off[T1B], n, UT, Hd, 1);
        gemm(M, false, true, n, F, Hd, hT, Hd, P + off[T2W], Hd, 0.f, s2, F);
        k_dyn_res<<<GRID((long)n * F)>>>(s2, P + off[T2B], uT, n, F);
        std::swap(sv, s2);
        k_store_step<<<GRID((long)n)>>>(hout, hh, n, H, k, 1);
        if (lat) k_store_step<<<GRID((long)n * F)>>>(lat, sv, n, H, k, F);
        if (fr) {
            linear(M, hX, sv, P + off[X1W], P + off[X1B], n, F, Hd, 1);
            linear(M, xx, hX, P + off[X2W], P + off[X2B], n, Hd, R, 0);
            k_store_step<<<GRID((long)n * R)>>>(fr, xx, n, H, k, R);
        }
    }
    CK(cudaMemcpy(hunger, hout, (size_t)n * H * sizeof(float), cudaMemcpyDeviceToHost));
    if (lat) CK(cudaMemcpy(latents, lat, (size_t)n * H * F * sizeof(float), cudaMemcpyDeviceToHost));
    if (fr) CK(cudaMemcpy(frames, fr, (size_t)n * H * R * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int vmc_decode(void* h, int n, const float* latents, float* frames) {
    Model* M = (Model*)h;
    const Dims& D = M->D;
    if (!D.dh || n < 1) return -1;
    float* sv = M->im_s.get((size_t)n * D.feat);
    float* hX = M->im_hX.get((size_t)n * D.dh);
    float* xx = M->im_x.get((size_t)n * D.rpx);
    CK(cudaMemcpy(sv, latents, (size_t)n * D.feat * sizeof(float), cudaMemcpyHostToDevice));
    linear(M, hX, sv, M->P + M->off[X1W], M->P + M->off[X1B], n, D.feat, D.dh, 1);
    linear(M, xx, hX, M->P + M->off[X2W], M->P + M->off[X2B], n, D.dh, D.rpx, 0);
    CK(cudaMemcpy(frames, xx, (size_t)n * D.rpx * sizeof(float), cudaMemcpyDeviceToHost));
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
    k_sqnorm<<<256, 256>>>(M->G, M->n_params, h0, h1, p0, p1, M->off[AX1W], M->acc + 5);
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
    double a[ACC_N];
    CK(cudaMemcpy(a, M->acc, sizeof(a), cudaMemcpyDeviceToHost));
    const double n = M->B > 0 ? M->B : 1;
    memset(out, 0, sizeof(VMCStats));
    out->loss = (float)((M->coef[0] * a[0] + M->coef[1] * a[19] + M->coef[2] * a[3] + M->coef[3] * a[10] +
                         M->coef[4] * a[13] + M->coef[5] * a[20] + M->coef[6] * a[17] +
                         M->coef[7] * (a[21] + a[22])) / n);   // weighted drive terms
    out->dyn_mse = a[14] > 0 ? (float)(a[13] / a[14]) : 0.f;
    out->dyn_hunger_mse = a[16] > 0 ? (float)(a[15] / a[16]) : 0.f;
    out->dyn_recon_mse = a[18] > 0 ? (float)(a[17] / a[18]) : 0.f;
    out->inv_mse = a[12] > 0 ? (float)(a[10] / a[12]) : 0.f;
    out->inv_r2 = a[11] > 0 ? (float)(1.0 - a[10] / a[11]) : 0.f;
    out->pred_mse = (float)(a[1] / n);
    out->drive_mse = (float)(a[2] / n);
    out->move_nll = (float)(a[3] / n);
    out->click_p = (float)(a[4] / n);
    out->pred_changed_mse = a[9] > 0 ? (float)(a[8] / a[9]) : 0.f;
    out->grad_norm = (float)sqrt(a[5] + a[6] + a[7]);
    out->gn_backbone = (float)sqrt(a[5]); out->gn_policy = (float)sqrt(a[6]); out->gn_pred = (float)sqrt(a[7]);
    out->act_drive_mse = (float)(a[23] / n);
    if (M->D.uni) {   // the universal model: the mean token cross-entropy, and by predicted kind
        out->loss = (float)(a[33] / ((double)n * (M->D.ntok - M->D.nnot)));
        float* ce[3] = {&out->uni_ce_body, &out->uni_ce_patch, &out->uni_ce_move};
        float* se[3] = {&out->uni_mse_body, &out->uni_mse_patch, &out->uni_mse_move};
        for (int k = 0; k < 3; k++) if (a[30 + k] > 0) { *ce[k] = (float)(a[24 + k] / a[30 + k]); *se[k] = (float)(a[27 + k] / a[30 + k]); }
    }
    out->act_frame_mse = (float)(a[22] / n);
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
