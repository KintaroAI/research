// vm_agent.cpp — Visual–motor agent: model, acting, rollout memory, PPO + AdamW (CPU, OpenMP, SSE).
//
// Architecture (sizes from VMConfig):
//   image u8 [img,img,3] -> /255 -> conv k4s2p1 (3->c1) ReLU -> conv (c1->c2) ReLU -> conv (c2->c3) ReLU
//     -> flatten (c3 x img/8 x img/8) -> linear -> feat, ReLU                       = visual feature v_t
//   token input u_t = [v_t, drives_t (2), prev_action_t (3), prev_reward_t (1)]
//   x_t = W_in u_t + b_in + wpe[pos_t]            pos_t = steps since the context started
//   n_layer pre-LN Transformer blocks (causal, segment-masked attention, GELU MLP) -> LN_f -> z_t
//   heads on z_t:
//     policy : mu = tanh(W_pi z)[0:2], state-independent log std (2), click logit = (W_pi z)[2]
//     value  : V = w_v . z + b_v
//     predict: h = ReLU(W_p1 [z_t, a_t] + b) -> ReLU(W_p2 h + b) -> (pc2, p/4, p/4) -> deconv k4s2p1 ReLU
//              -> deconv k4s2p1 -> sigmoid = predicted next frame (3, p, p); sigmoid(W_dr h + b) = next drives
//   pred_residual=1: predicted next frame = downsampled current frame + decoder output (no sigmoid),
//              i.e. the decoder predicts the change caused by the action.
//   pred_skip=1: U-Net-style skips from the encoder of the current frame plus the action tiled as 3
//              channels: conv-3 map (c3 x img/8^2) + action into deconv 1 (followed by ReLU, so action and
//              local features interact nonlinearly), conv-2 map (c2 x img/4^2) + action into deconv 2.
//              The effect of a move can then be computed locally. Skip gradients flow into the encoder.
//   inv_hidden>0: inverse-dynamics head on the visual features of consecutive frames of one episode:
//              MLP([v_t, v_{t+1}]) -> executed (dx, dy) of a_t. Uses the encoder output (before the token
//              mixes in the previous action, which would leak a_t), so it teaches the encoder where the
//              controllable cursor is. Target = the agent's own executed action (no world state).
//   coord=1 (CoordConv / spatial broadcast): the encoder input gets 2 constant coordinate channels; the
//              first deconv input gets [MLP map, ReLU(W_b h) tiled over the grid, x, y], the second
//              [features, x, y]. Coordinate maps are fixed positional encodings (no world information).
//
// Context handling: a context starts at every episode start and at every block boundary (every `ctx`
// steps). Training sequences are exactly these blocks, so the KV-cached acting path and the batched
// training path compute the same function (identical accumulation order => bitwise-equal outputs).
//
// Every matrix product is an NN GEMM (gemm_nn). Transposed copies of weight matrices (wt) are refreshed
// after every optimizer step. Training parallelism: one OpenMP thread per 32-step sequence, thread-local
// gradient buffers, deterministic static schedule + ordered reduction.
//
// Tests: src/test_vm_agent.cpp does `#define TESTING` + `#include "vm_agent.cpp"` (llm.c pattern).
#include "vm_agent.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <omp.h>
#include <immintrin.h>

#define CK(x)                                                                              \
    do {                                                                                   \
        if (!(x)) {                                                                        \
            fprintf(stderr, "vm_agent: check failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
            abort();                                                                       \
        }                                                                                  \
    } while (0)

static const int VM_MAGIC = 20261023;
static const int VM_VERSION = 1;
static const float LS_MIN = -5.0f, LS_MAX = 1.0f;   // log-std clamp range
static const float LOG_2PI = 1.8378770664093453f;

// =====================================================================================
// RNG: xoshiro256** seeded by splitmix64
// =====================================================================================
struct Rng { uint64_t s[4]; };
static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void rng_seed(Rng& r, uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) r.s[i] = splitmix64(x);
}
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rng_next(Rng& r) {
    uint64_t* s = r.s;
    uint64_t res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static inline float rng_uniform(Rng& r) { return (float)(rng_next(r) >> 40) * (1.0f / 16777216.0f); }  // [0,1)
static inline float rng_normal(Rng& r) {
    float u1 = rng_uniform(r), u2 = rng_uniform(r);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

// =====================================================================================
// GEMM: C[M,N] = (acc ? C : 0) + A[M,K] . B[K,N], row-major. 4x8 SSE register blocks.
// Per-element accumulation order is k = 0..K-1 in every code path (vector, column tail, row tail),
// so results do not depend on M/N blocking (needed for acting == training bitwise).
// =====================================================================================
static void gemm_nn(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                    float* C, int ldc, bool acc) {
    int i = 0;
    for (; i + 4 <= M; i += 4) {
        const float* a0 = A + (size_t)(i + 0) * lda;
        const float* a1 = A + (size_t)(i + 1) * lda;
        const float* a2 = A + (size_t)(i + 2) * lda;
        const float* a3 = A + (size_t)(i + 3) * lda;
        float* c0 = C + (size_t)(i + 0) * ldc;
        float* c1 = C + (size_t)(i + 1) * ldc;
        float* c2 = C + (size_t)(i + 2) * ldc;
        float* c3 = C + (size_t)(i + 3) * ldc;
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            __m128 s00, s01, s10, s11, s20, s21, s30, s31;
            if (acc) {
                s00 = _mm_loadu_ps(c0 + j); s01 = _mm_loadu_ps(c0 + j + 4);
                s10 = _mm_loadu_ps(c1 + j); s11 = _mm_loadu_ps(c1 + j + 4);
                s20 = _mm_loadu_ps(c2 + j); s21 = _mm_loadu_ps(c2 + j + 4);
                s30 = _mm_loadu_ps(c3 + j); s31 = _mm_loadu_ps(c3 + j + 4);
            } else {
                s00 = s01 = s10 = s11 = s20 = s21 = s30 = s31 = _mm_setzero_ps();
            }
            const float* bp = B + j;
            for (int k = 0; k < K; k++, bp += ldb) {
                __m128 b0 = _mm_loadu_ps(bp), b1 = _mm_loadu_ps(bp + 4), x;
                x = _mm_set1_ps(a0[k]); s00 = _mm_add_ps(s00, _mm_mul_ps(x, b0)); s01 = _mm_add_ps(s01, _mm_mul_ps(x, b1));
                x = _mm_set1_ps(a1[k]); s10 = _mm_add_ps(s10, _mm_mul_ps(x, b0)); s11 = _mm_add_ps(s11, _mm_mul_ps(x, b1));
                x = _mm_set1_ps(a2[k]); s20 = _mm_add_ps(s20, _mm_mul_ps(x, b0)); s21 = _mm_add_ps(s21, _mm_mul_ps(x, b1));
                x = _mm_set1_ps(a3[k]); s30 = _mm_add_ps(s30, _mm_mul_ps(x, b0)); s31 = _mm_add_ps(s31, _mm_mul_ps(x, b1));
            }
            _mm_storeu_ps(c0 + j, s00); _mm_storeu_ps(c0 + j + 4, s01);
            _mm_storeu_ps(c1 + j, s10); _mm_storeu_ps(c1 + j + 4, s11);
            _mm_storeu_ps(c2 + j, s20); _mm_storeu_ps(c2 + j + 4, s21);
            _mm_storeu_ps(c3 + j, s30); _mm_storeu_ps(c3 + j + 4, s31);
        }
        for (; j < N; j++) {
            float s0 = acc ? c0[j] : 0.f, s1 = acc ? c1[j] : 0.f, s2 = acc ? c2[j] : 0.f, s3 = acc ? c3[j] : 0.f;
            for (int k = 0; k < K; k++) {
                float b = B[(size_t)k * ldb + j];
                s0 += a0[k] * b; s1 += a1[k] * b; s2 += a2[k] * b; s3 += a3[k] * b;
            }
            c0[j] = s0; c1[j] = s1; c2[j] = s2; c3[j] = s3;
        }
    }
    for (; i < M; i++) {
        const float* a = A + (size_t)i * lda;
        float* c = C + (size_t)i * ldc;
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            __m128 s0 = acc ? _mm_loadu_ps(c + j) : _mm_setzero_ps();
            __m128 s1 = acc ? _mm_loadu_ps(c + j + 4) : _mm_setzero_ps();
            const float* bp = B + j;
            for (int k = 0; k < K; k++, bp += ldb) {
                __m128 x = _mm_set1_ps(a[k]);
                s0 = _mm_add_ps(s0, _mm_mul_ps(x, _mm_loadu_ps(bp)));
                s1 = _mm_add_ps(s1, _mm_mul_ps(x, _mm_loadu_ps(bp + 4)));
            }
            _mm_storeu_ps(c + j, s0); _mm_storeu_ps(c + j + 4, s1);
        }
        for (; j < N; j++) {
            float s = acc ? c[j] : 0.f;
            for (int k = 0; k < K; k++) s += a[k] * B[(size_t)k * ldb + j];
            c[j] = s;
        }
    }
}

static void transpose(const float* src, int rows, int cols, float* dst) {  // dst[cols, rows]
    const int BS = 32;
    for (int i0 = 0; i0 < rows; i0 += BS)
        for (int j0 = 0; j0 < cols; j0 += BS) {
            int i1 = std::min(rows, i0 + BS), j1 = std::min(cols, j0 + BS);
            for (int i = i0; i < i1; i++)
                for (int j = j0; j < j1; j++) dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
        }
}

// =====================================================================================
// Layer ops
// =====================================================================================
// Convolution geometry is fixed to kernel 4, stride 2, padding 1 (exact halving / doubling).
static const int KS = 4, KK = 16, ST = 2, PD = 1;

// cols[(c*4+ky)*4+kx][oy*Wo+ox] = x[c][oy*2-1+ky][ox*2-1+kx]   (x: [C,H,W])
static void im2col(const float* x, int C, int H, int W, int Ho, int Wo, float* cols) {
    const int P = Ho * Wo;
    for (int c = 0; c < C; c++)
        for (int ky = 0; ky < KS; ky++)
            for (int kx = 0; kx < KS; kx++) {
                float* dst = cols + (size_t)((c * KS + ky) * KS + kx) * P;
                for (int oy = 0; oy < Ho; oy++) {
                    int iy = oy * ST - PD + ky;
                    float* dr = dst + oy * Wo;
                    if (iy < 0 || iy >= H) { memset(dr, 0, sizeof(float) * Wo); continue; }
                    const float* xr = x + ((size_t)c * H + iy) * W;
                    for (int ox = 0; ox < Wo; ox++) {
                        int ix = ox * ST - PD + kx;
                        dr[ox] = (ix >= 0 && ix < W) ? xr[ix] : 0.f;
                    }
                }
            }
}
// rows[oy*Wo+ox][(c*4+ky)*4+kx] (transpose of im2col)
static void im2row(const float* x, int C, int H, int W, int Ho, int Wo, float* rows) {
    const int CKK = C * KK;
    for (int oy = 0; oy < Ho; oy++)
        for (int ox = 0; ox < Wo; ox++) {
            float* dr = rows + (size_t)(oy * Wo + ox) * CKK;
            for (int c = 0; c < C; c++)
                for (int ky = 0; ky < KS; ky++) {
                    int iy = oy * ST - PD + ky;
                    for (int kx = 0; kx < KS; kx++) {
                        int ix = ox * ST - PD + kx;
                        dr[(c * KS + ky) * KS + kx] =
                            (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[((size_t)c * H + iy) * W + ix] : 0.f;
                    }
                }
        }
}
// x += adjoint(im2col)(cols)
static void col2im(const float* cols, int C, int H, int W, int Ho, int Wo, float* x) {
    const int P = Ho * Wo;
    for (int c = 0; c < C; c++)
        for (int ky = 0; ky < KS; ky++)
            for (int kx = 0; kx < KS; kx++) {
                const float* src = cols + (size_t)((c * KS + ky) * KS + kx) * P;
                for (int oy = 0; oy < Ho; oy++) {
                    int iy = oy * ST - PD + ky;
                    if (iy < 0 || iy >= H) continue;
                    float* xr = x + ((size_t)c * H + iy) * W;
                    const float* sr = src + oy * Wo;
                    for (int ox = 0; ox < Wo; ox++) {
                        int ix = ox * ST - PD + kx;
                        if (ix >= 0 && ix < W) xr[ix] += sr[ox];
                    }
                }
            }
}

// y [Co, H/2, H/2] = ReLU(conv(x [Ci, H, H]; Wm [Co, Ci*16]) + b)
static void conv_fwd(const float* x, int Ci, int H, const float* Wm, const float* b, int Co, float* y, float* scr) {
    const int Ho = H / 2, P = Ho * Ho, CKK = Ci * KK;
    im2col(x, Ci, H, H, Ho, Ho, scr);
    for (int co = 0; co < Co; co++)
        for (int p = 0; p < P; p++) y[(size_t)co * P + p] = b[co];
    gemm_nn(Co, P, CKK, Wm, CKK, scr, P, y, P, true);
    for (int i = 0; i < Co * P; i++) y[i] = y[i] > 0.f ? y[i] : 0.f;
}
// dy: grad wrt the conv output (already through ReLU'). dW [Co, Ci*16] += ; db += ; dx [Ci,H,H] = (if non-null)
static void conv_bwd(const float* x, int Ci, int H, const float* WT, int Co, const float* dy, float* dW, float* db,
                     float* dx, float* scr) {
    const int Ho = H / 2, P = Ho * Ho, CKK = Ci * KK;
    im2row(x, Ci, H, H, Ho, Ho, scr);                       // [P, CKK]
    gemm_nn(Co, CKK, P, dy, P, scr, CKK, dW, CKK, true);
    for (int co = 0; co < Co; co++) {
        float s = 0.f;
        for (int p = 0; p < P; p++) s += dy[(size_t)co * P + p];
        db[co] += s;
    }
    if (dx) {
        gemm_nn(CKK, P, Co, WT, Co, dy, P, scr, P, false);  // dcols [CKK, P] = W^T dy
        memset(dx, 0, sizeof(float) * (size_t)Ci * H * H);
        col2im(scr, Ci, H, H, Ho, Ho, dx);
    }
}
// Transposed conv (k4 s2 p1): y [Co, 2Hi, 2Hi] = col2im(W^T x) + b, Wd layout [Ci, Co*16] (PyTorch ConvTranspose2d)
static void deconv_fwd(const float* x, int Ci, int Hi, const float* WdT, const float* b, int Co, float* y, float* scr) {
    const int Ho = 2 * Hi, Pi = Hi * Hi, Po = Ho * Ho;
    gemm_nn(Co * KK, Pi, Ci, WdT, Ci, x, Pi, scr, Pi, false);  // cols [Co*16, Pi]
    for (int co = 0; co < Co; co++)
        for (int p = 0; p < Po; p++) y[(size_t)co * Po + p] = b[co];
    col2im(scr, Co, Ho, Ho, Hi, Hi, y);
}
static void deconv_bwd(const float* x, int Ci, int Hi, const float* Wd, int Co, const float* dy, float* dW, float* db,
                       float* dx, float* scr) {
    const int Ho = 2 * Hi, Pi = Hi * Hi, Po = Ho * Ho, CoKK = Co * KK;
    for (int co = 0; co < Co; co++) {
        float s = 0.f;
        for (int p = 0; p < Po; p++) s += dy[(size_t)co * Po + p];
        db[co] += s;
    }
    im2row(dy, Co, Ho, Ho, Hi, Hi, scr);                        // [Pi, CoKK]
    gemm_nn(Ci, CoKK, Pi, x, Pi, scr, CoKK, dW, CoKK, true);
    if (dx) {
        im2col(dy, Co, Ho, Ho, Hi, Hi, scr);                    // [CoKK, Pi]
        gemm_nn(Ci, Pi, CoKK, Wd, CoKK, scr, Pi, dx, Pi, false);
    }
}

// out [T, OC] = inp [T, C] . W^T + b, using WT [C, OC]
static void linear_fwd(float* out, const float* inp, const float* WT, const float* b, int T, int C, int OC) {
    for (int t = 0; t < T; t++)
        for (int o = 0; o < OC; o++) out[(size_t)t * OC + o] = b[o];
    gemm_nn(T, OC, C, inp, C, WT, OC, out, OC, true);
}
// dinp [T, C] (= or +=) dout . W ; dW [OC, C] += dout^T . inp ; db += sum_t dout
static void linear_bwd(const float* dout, const float* inp, const float* W, int T, int C, int OC, float* dinp,
                       bool dinp_acc, float* dW, float* db, float* scr) {
    if (dinp) gemm_nn(T, C, OC, dout, OC, W, C, dinp, C, dinp_acc);
    transpose(dout, T, OC, scr);                             // [OC, T]
    gemm_nn(OC, C, T, scr, T, inp, C, dW, C, true);
    for (int t = 0; t < T; t++)
        for (int o = 0; o < OC; o++) db[o] += dout[(size_t)t * OC + o];
}

static void layernorm_fwd(float* out, float* mean, float* rstd, const float* inp, const float* w, const float* b,
                          int T, int C) {
    for (int t = 0; t < T; t++) {
        const float* x = inp + (size_t)t * C;
        float m = 0.f;
        for (int i = 0; i < C; i++) m += x[i];
        m /= C;
        float v = 0.f;
        for (int i = 0; i < C; i++) { float xs = x[i] - m; v += xs * xs; }
        v /= C;
        float s = 1.0f / sqrtf(v + 1e-5f);
        float* o = out + (size_t)t * C;
        for (int i = 0; i < C; i++) o[i] = (s * (x[i] - m)) * w[i] + b[i];
        mean[t] = m;
        rstd[t] = s;
    }
}
// dinp += ; dw += ; db +=
static void layernorm_bwd(float* dinp, float* dw, float* db, const float* dout, const float* inp, const float* w,
                          const float* mean, const float* rstd, int T, int C) {
    for (int t = 0; t < T; t++) {
        const float* dy = dout + (size_t)t * C;
        const float* x = inp + (size_t)t * C;
        float* dx = dinp + (size_t)t * C;
        float m = mean[t], s = rstd[t];
        float dm = 0.f, dnm = 0.f;
        for (int i = 0; i < C; i++) {
            float nrm = (x[i] - m) * s, dn = w[i] * dy[i];
            dm += dn;
            dnm += dn * nrm;
        }
        dm /= C;
        dnm /= C;
        for (int i = 0; i < C; i++) {
            float nrm = (x[i] - m) * s, dn = w[i] * dy[i];
            db[i] += dy[i];
            dw[i] += nrm * dy[i];
            dx[i] += (dn - dm - nrm * dnm) * s;
        }
    }
}

// Softmax attention of one query over nk keys/values (rows at kb + j*ks, vb + j*vs); weights -> w[0..nk)
// al (attention regions, task 00044): key j may be attended to only if al[(s0 + j) % nt]; else its weight is 0
static void attend_one(const float* q, const float* kb, int ks, const float* vb, int vs, int nk, int hs, float scale,
                       float* w, float* out, const unsigned char* al = nullptr, int nt = 1, int s0 = 0) {
    float mx = -FLT_MAX;
    for (int j = 0; j < nk; j++) {
        if (al && !al[(s0 + j) % nt]) continue;
        const float* k = kb + (size_t)j * ks;
        float s = 0.f;
        for (int i = 0; i < hs; i++) s += q[i] * k[i];
        s *= scale;
        w[j] = s;
        if (s > mx) mx = s;
    }
    float sum = 0.f;
    for (int j = 0; j < nk; j++) {
        if (al && !al[(s0 + j) % nt]) { w[j] = 0.f; continue; }
        float e = expf(w[j] - mx); w[j] = e; sum += e;
    }
    float inv = 1.0f / sum;
    for (int j = 0; j < nk; j++) w[j] *= inv;
    for (int i = 0; i < hs; i++) out[i] = 0.f;
    for (int j = 0; j < nk; j++) {
        const float* v = vb + (size_t)j * vs;
        float a = w[j];
        for (int i = 0; i < hs; i++) out[i] += a * v[i];
    }
}
// qkv [T, 3d] -> out [T, d]; att [nh, Kmax, Kmax] (row t: weights at columns ss[t]..se[t], se = NULL: ..t).
// allow (attention regions, task 00044; NULL = none): token t attends to t2 only if allow[(t % nt) * nt + t2 % nt]
static void attention_fwd(float* out, float* att, const float* qkv, const int* ss, const int* se, int T, int Kmax,
                          int d, int nh, const unsigned char* allow = nullptr, int nt = 1) {
    const int hs = d / nh, d3 = 3 * d;
    const float scale = 1.0f / sqrtf((float)hs);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < nh; h++) {
            int lo = ss[t], hi = se ? se[t] : t;
            float* arow = att + ((size_t)h * Kmax + t) * Kmax;
            attend_one(qkv + (size_t)t * d3 + h * hs, qkv + (size_t)lo * d3 + d + h * hs, d3,
                       qkv + (size_t)lo * d3 + 2 * d + h * hs, d3, hi - lo + 1, hs, scale, arow + lo,
                       out + (size_t)t * d + h * hs, allow ? allow + (size_t)(t % nt) * nt : nullptr, nt, lo);
        }
}
// dqkv += (must be zeroed by caller); datt: scratch [Kmax]
static void attention_bwd(float* dqkv, const float* dout, const float* qkv, const float* att, const int* ss,
                          const int* se, int T, int Kmax, int d, int nh, float* datt) {
    const int hs = d / nh, d3 = 3 * d;
    const float scale = 1.0f / sqrtf((float)hs);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < nh; h++) {
            const float* q = qkv + (size_t)t * d3 + h * hs;
            float* dq = dqkv + (size_t)t * d3 + h * hs;
            const float* w = att + ((size_t)h * Kmax + t) * Kmax;
            const float* dy = dout + (size_t)t * d + h * hs;
            const int lo = ss[t], hi = se ? se[t] : t;
            for (int t2 = lo; t2 <= hi; t2++) {
                const float* v = qkv + (size_t)t2 * d3 + 2 * d + h * hs;
                float* dv = dqkv + (size_t)t2 * d3 + 2 * d + h * hs;
                float s = 0.f;
                for (int i = 0; i < hs; i++) s += v[i] * dy[i];
                datt[t2] = s;
                for (int i = 0; i < hs; i++) dv[i] += w[t2] * dy[i];
            }
            float dsum = 0.f;
            for (int t2 = lo; t2 <= hi; t2++) dsum += w[t2] * datt[t2];
            for (int t2 = lo; t2 <= hi; t2++) {
                float dp = w[t2] * (datt[t2] - dsum) * scale;
                const float* k = qkv + (size_t)t2 * d3 + d + h * hs;
                float* dk = dqkv + (size_t)t2 * d3 + d + h * hs;
                for (int i = 0; i < hs; i++) { dq[i] += dp * k[i]; dk[i] += dp * q[i]; }
            }
        }
}

static const float GELU_S = 0.7978845608028654f;  // sqrt(2/pi)
static void gelu_fwd(float* out, const float* inp, int n) {
    for (int i = 0; i < n; i++) {
        float x = inp[i];
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_S * (x + 0.044715f * x * x * x)));
    }
}
static void gelu_bwd(float* dinp, const float* inp, const float* dout, int n) {  // dinp =
    for (int i = 0; i < n; i++) {
        float x = inp[i];
        float ta = GELU_S * (x + 0.044715f * x * x * x);
        float th = tanhf(ta), ch = coshf(ta);
        float sech2 = 1.0f / (ch * ch);
        float g = 0.5f * (1.0f + th) + x * 0.5f * sech2 * GELU_S * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = g * dout[i];
    }
}

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf_(float x) { return (x > 0.f ? x : 0.f) + log1pf(expf(-fabsf(x))); }
static inline float clampf_(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// Box-average downsample of an u8 HWC image by integer factor f (rounding half up).
static void downsample_u8(const uint8_t* img, int S, int f, uint8_t* out) {
    const int So = S / f, ff = f * f;
    for (int y = 0; y < So; y++)
        for (int x = 0; x < So; x++)
            for (int c = 0; c < 3; c++) {
                int s = 0;
                for (int dy = 0; dy < f; dy++)
                    for (int dx = 0; dx < f; dx++) s += img[((size_t)(y * f + dy) * S + (x * f + dx)) * 3 + c];
                out[((size_t)y * So + x) * 3 + c] = (uint8_t)((s + ff / 2) / ff);
            }
}

// =====================================================================================
// Policy distribution: (dx, dy) ~ N(tanh(pi[0:2]), exp(logstd)^2), click ~ Bernoulli(sigmoid(pi[2]))
// =====================================================================================
struct PolicyEval {
    float mu[2], ls[2], sd[2], logit, p, logp, ent;
    float dlogp_dpi[3], dlogp_dls[2], dent_dpi[3], dent_dls[2];
};
static void policy_eval(const float* pi3, const float* logstd, const float* a, PolicyEval& e) {
    e.logp = 0.f;
    e.ent = 0.f;
    for (int i = 0; i < 2; i++) {
        e.mu[i] = tanhf(pi3[i]);
        e.ls[i] = clampf_(logstd[i], LS_MIN, LS_MAX);
        e.sd[i] = expf(e.ls[i]);
        bool inr = logstd[i] >= LS_MIN && logstd[i] <= LS_MAX;
        float z = (a[i] - e.mu[i]) / e.sd[i];
        e.logp += -0.5f * z * z - e.ls[i] - 0.5f * LOG_2PI;
        e.ent += 0.5f + 0.5f * LOG_2PI + e.ls[i];
        e.dlogp_dpi[i] = (z / e.sd[i]) * (1.0f - e.mu[i] * e.mu[i]);
        e.dlogp_dls[i] = inr ? (z * z - 1.0f) : 0.f;
        e.dent_dpi[i] = 0.f;
        e.dent_dls[i] = inr ? 1.0f : 0.f;
    }
    float l = pi3[2];
    e.logit = l;
    e.p = sigmoidf_(l);
    float c = a[2] > 0.5f ? 1.0f : 0.0f;
    e.logp += c > 0.5f ? -softplusf_(-l) : -softplusf_(l);
    e.ent += e.p * softplusf_(-l) + (1.0f - e.p) * softplusf_(l);
    e.dlogp_dpi[2] = c - e.p;
    e.dent_dpi[2] = -l * e.p * (1.0f - e.p);
}
static inline float value_head(const float* z, const float* w, float b, int d) {
    float s = 0.f;
    for (int i = 0; i < d; i++) s += z[i] * w[i];
    return s + b;
}

// =====================================================================================
// Model + per-thread workspace
// =====================================================================================
enum ParamId {
    C1W, C1B, C2W, C2B, C3W, C3B, EFW, EFB, INW, INB, WPE,
    LN1W, LN1B, QKVW, QKVB, APW, APB, LN2W, LN2B, FCW, FCB, FPW, FPB, LNFW, LNFB,
    PIW, PIB, PHW, PHB, LOGSTD, VW, VB,   // PHW/PHB: policy hidden layer (pi_hidden > 0)
    PR1W, PR1B, PR2W, PR2B, PBW, PBB, D1W, D1B, D2W, D2B, DRW, DRB, IV1W, IV1B, IV2W, IV2B,
    T1W, T1B, T2W, T2B, H1W, H1B, H2W, H2B, X1W, X1B, X2W, X2B,   // latent dynamics (dyn_hidden > 0)
    PEW, PEB, POS2D, TYP, DEW, DEB, MEW, MEB, CEW, CEB,       // patch tokens (patch > 0)
    ACTE, AX1W, AX1B, AXDW, AXDB, AXFW, AXFB,                   // act token (act_token = 1) and its auxiliary heads
    UIW, UIB, USL, UOW, UOB, NUM_P                              // the universal model: input map, slots, output head
};
static const char* PNAME[NUM_P] = {
    "enc_c1w", "enc_c1b", "enc_c2w", "enc_c2b", "enc_c3w", "enc_c3b", "enc_fcw", "enc_fcb", "in_w", "in_b", "wpe",
    "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb", "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb",
    "lnfw", "lnfb", "pi_w", "pi_b", "pi_h_w", "pi_h_b", "logstd", "v_w", "v_b", "pr1w", "pr1b", "pr2w", "pr2b", "pbw", "pbb", "dc1w", "dc1b",
    "dc2w", "dc2b", "drw", "drb", "iv1w", "iv1b", "iv2w", "iv2b",
    "dt1w", "dt1b", "dt2w", "dt2b", "dh1w", "dh1b", "dh2w", "dh2b", "dx1w", "dx1b", "dx2w", "dx2b",
    "pt_w", "pt_b", "pos2d", "tok_type", "drv_w", "drv_b", "mv_w", "mv_b", "cmd_w", "cmd_b",
    "act_e", "ax1w", "ax1b", "axdw", "axdb", "axfw", "axfb",
    "uni_in_w", "uni_in_b", "uni_slot", "uni_out_w", "uni_out_b"};

static const int ACT_CG = 8;   // act token: side of the coarse next frame it predicts

struct Dims {
    int img, c1, c2, c3, h1, h2, h3, enc_flat, feat, d, L, nh, hs, dff, K, in_dim;
    int pimg, p0, p1, pc1, pc2, ph, pred_flat, pin_dim, img_px, small_px, ds;
    int coord, in_ch, pb, d1_in, d2_in;   // coordinate channels on/off; encoder input / deconv input channels
    int residual;                         // prediction = current frame + decoder output
    int skip, skip_off, skip1_off;        // encoder skips + tiled action; channel offsets in deconv 2 / deconv 1
    int ih;                               // inverse-dynamics hidden size (0 = no inverse head)
    int dh, rimg, rpx;                    // latent dynamics hidden size (0 = off); decoder frame side / values
    int pih;                              // policy hidden size (0 = linear policy head)
    int nsum = 0, nnot = 0;               // universal: summary tokens per step (task 00047), tokens without a target
    int patch, gp, npch, ntok, pdim;      // patch tokens: patch side (0 = off), grid side, patches, tokens per step,
                                          // values per patch
    int cell, cpx, KT;                    // predicted cell side / values per patch; tokens per context (K * ntok)
    int act, cds, cfx;                    // act token on/off; prediction frame -> coarse frame block side; coarse values
    int uni, V, nbin, OV, ch;             // universal model: on/off; values per token; bins per value; head outputs;
                                          // channels per patch pixel (1 with gray, else 3)
    int ib, VI;                           // bin-embedding input on/off (task 00036); W_in's input width (V or V * bins)
};
static Dims make_dims(const VMConfig& c) {
    Dims D;
    D.img = c.img; D.c1 = c.c1; D.c2 = c.c2; D.c3 = c.c3;
    D.h1 = c.img / 2; D.h2 = c.img / 4; D.h3 = c.img / 8;
    D.enc_flat = c.c3 * D.h3 * D.h3;
    D.feat = c.feat; D.d = c.d_model; D.L = c.n_layer; D.nh = c.n_head; D.hs = c.d_model / c.n_head;
    D.dff = c.d_ff; D.K = c.ctx; D.in_dim = c.feat + 6;
    D.pimg = c.pred_img; D.p0 = c.pred_img / 4; D.p1 = c.pred_img / 2;
    D.pc1 = c.pc1; D.pc2 = c.pc2; D.ph = c.pred_hidden;
    D.pred_flat = c.pc2 * D.p0 * D.p0; D.pin_dim = c.d_model + 3;
    D.img_px = c.img * c.img * 3; D.small_px = c.pred_img * c.pred_img * 3;
    D.ds = c.img / c.pred_img;
    D.coord = c.coord ? 1 : 0;
    D.in_ch = 3 + 2 * D.coord;
    D.pb = D.coord ? c.pb : 0;
    D.d1_in = D.pc2 + D.pb + 2 * D.coord + (c.pred_skip ? D.c3 + 3 : 0);
    D.skip1_off = D.pc2 + D.pb + 2 * D.coord;
    D.ih = c.inv_hidden > 0 ? c.inv_hidden : 0;
    D.d2_in = D.pc1 + 2 * D.coord;
    D.residual = c.pred_residual ? 1 : 0;
    D.skip = c.pred_skip ? 1 : 0;
    D.skip_off = D.pc1 + 2 * D.coord;
    D.d2_in = D.pc1 + 2 * D.coord + D.skip * (D.c2 + 3);
    D.dh = c.dyn_hidden > 0 ? c.dyn_hidden : 0;
    D.rimg = c.pred_img / 2;
    D.rpx = D.rimg * D.rimg * 3;
    D.pih = c.pi_hidden > 0 ? c.pi_hidden : 0;
    D.patch = c.patch > 0 ? c.patch : 0;
    D.gp = D.patch ? c.img / D.patch : 0;
    D.npch = D.gp * D.gp;
    D.act = (D.patch && c.act_token) ? 1 : 0;
    D.ntok = D.patch ? D.npch + 3 + D.act : 1;
    D.uni = (D.patch && c.universal) ? 1 : 0;   // tokens: body, the patches, the move
    if (D.uni) D.ntok = vm_uni_nt(&c);          // + the summary tokens (task 00047)
    D.nsum = D.uni ? vm_uni_nsum(&c) : 0;
    D.nnot = D.uni ? vm_uni_untargeted(&c) : 0;
    D.cds = c.pred_img / ACT_CG;
    D.cfx = ACT_CG * ACT_CG * 3;
    D.pdim = D.patch * D.patch * 3;
    D.ch = (D.uni && c.gray) ? 1 : 3;
    D.V = D.uni ? D.patch * D.patch * D.ch : 0;
    D.nbin = D.uni ? c.bins : 0;
    D.OV = D.V * D.nbin;
    D.ib = D.uni ? c.in_bins : 0;
    D.VI = D.ib ? D.V * D.nbin : D.V;
    D.cell = D.patch ? c.pred_img / D.gp : 0;
    D.cpx = D.cell * D.cell * 3;
    D.KT = D.K * D.ntok;
    if (D.patch) { D.coord = 0; D.pb = 0; D.skip = 0; }   // CNN encoder/decoder options do not apply
    return D;
}

struct Workspace {
    int K = 0;
    std::vector<float> x0, a1, a2, a3, vf, tin, xin;
    std::vector<float> ln1, ln1m, ln1r, qkv, att, atty, res1, ln2, ln2m, ln2r, fch, fchg, res2;
    std::vector<float> lnf, lnfm, lnfr, pi, val, dpi, dval, pih, dpih;
    std::vector<float> pin, ph1, ph2, pd1, pout, pdrv, dpout, dpdrv, ub, d1in, d2in, dd1in, dd2in, dub;
    std::vector<int> ss, pos;
    std::vector<float> scr, proj, arow;
    std::vector<float> dz, dres, dln, datty, dqkv, dfch, dfchg, dtin, dvf, da3, da2, da1;
    std::vector<float> dph2, dph1, dpin, dpd1, dskip, dskip3;
    std::vector<float> iv_in, iv_h, iv_out, div_out, div_h, div_in, dvf_extra;
    std::vector<uint8_t> cur_small, cur_small_seq;
    // latent dynamics: chain steps (pairs) stored level by level; y_* forward, g_* gradients
    std::vector<int> y_t, y_j, y_par, y_lev;
    std::vector<float> y_uT, y_hT, y_s, y_uH, y_hH, y_h, y_hX, y_x, y_hX0, y_x0;
    std::vector<float> g_s, g_hT, g_uT, g_hH, g_uH, g_h, g_hX, g_x, g_hX0, g_x0;
    std::vector<uint8_t> y_small, y_tgt;
    // patch tokens: per token the first / last token it attends to; patch pixels and embeddings; the move tokens'
    // outputs; per-patch predicted cell changes
    std::vector<int> tss, tse;
    std::vector<float> tpx, pemb, dpemb, zm, dzm, pdel, dpdel;
    // act token: rows [its output, executed move]; hidden; predicted drives and coarse frame; gradients; coarse frames
    std::vector<float> ax_in, ax_h, ax_drv, ax_frm, dax_in, dax_h, dax_drv, dax_frm;
    std::vector<uint8_t> ax_cur;
    // universal model: per token the centred input values, the next token's values (targets) and how many of them are
    // used; one row of head logits and its probabilities
    std::vector<float> u_in, u_tgt, u_row, u_prob;
    std::vector<int> u_use;
};
static Workspace* ws_create(const Dims& D) {
    Workspace* W = new Workspace();
    const size_t K = D.K, d = D.d, L = D.L, KT = D.KT;           // KT: tokens per context (= K without patches)
    const size_t PR = D.patch ? K * (D.npch + 1) : K;               // prediction-head rows
    W->K = D.K;
    W->x0.resize(K * D.in_ch * D.img * D.img);
    W->a1.resize(K * D.c1 * D.h1 * D.h1);
    W->a2.resize(K * D.c2 * D.h2 * D.h2);
    W->a3.resize(K * D.enc_flat);
    W->vf.resize(K * D.feat);
    W->tin.resize(K * D.in_dim);
    W->xin.resize(KT * d);
    for (auto* v : {&W->ln1, &W->atty, &W->res1, &W->ln2, &W->res2}) v->resize(L * KT * d);
    for (auto* v : {&W->ln1m, &W->ln1r, &W->ln2m, &W->ln2r}) v->resize(L * KT);
    W->qkv.resize(L * KT * 3 * d);
    W->att.resize(L * D.nh * KT * KT);
    W->fch.resize(L * KT * D.dff);
    W->fchg.resize(L * KT * D.dff);
    W->lnf.resize(KT * d); W->lnfm.resize(KT); W->lnfr.resize(KT);
    W->pi.resize(K * 3); W->val.resize(K); W->dpi.resize(K * 3); W->dval.resize(K);
    if (D.pih) { W->pih.resize(K * D.pih); W->dpih.resize(K * D.pih); }
    W->pin.resize(PR * D.pin_dim); W->ph1.resize(PR * D.ph); W->ph2.resize(K * D.pred_flat);
    W->pd1.resize(K * D.pc1 * D.p1 * D.p1); W->pout.resize(K * D.small_px); W->pdrv.resize(K * 2);
    W->dpout.resize(K * D.small_px); W->dpdrv.resize(K * 2);
    W->ub.resize(K * std::max(1, D.pb)); W->dub.resize(K * std::max(1, D.pb));
    W->d1in.resize(K * D.d1_in * D.p0 * D.p0); W->d2in.resize(K * D.d2_in * D.p1 * D.p1);
    W->dd1in.resize((size_t)D.d1_in * D.p0 * D.p0); W->dd2in.resize((size_t)D.d2_in * D.p1 * D.p1);
    W->ss.resize(K); W->pos.resize(K);
    size_t scr = 0;
    scr = std::max(scr, (size_t)D.h1 * D.h1 * D.in_ch * KK);
    scr = std::max(scr, (size_t)D.h2 * D.h2 * D.c1 * KK);
    scr = std::max(scr, (size_t)D.h3 * D.h3 * D.c2 * KK);
    scr = std::max(scr, (size_t)D.p0 * D.p0 * D.pc1 * KK);
    scr = std::max(scr, (size_t)D.p1 * D.p1 * 3 * KK);
    int maxoc = std::max(std::max(3 * D.d, D.dff), std::max(std::max(D.feat, D.d), std::max(D.ph, D.pred_flat)));
    maxoc = std::max(maxoc, std::max(D.pb, std::max(D.ih, std::max(D.pih, D.cpx))));
    if (D.act) maxoc = std::max(maxoc, std::max(D.cfx, D.pin_dim));
    scr = std::max(scr, KT * (size_t)maxoc);
    W->scr.resize(scr);
    W->proj.resize(KT * std::max(d, (size_t)D.dff));
    W->arow.resize(KT);
    W->dz.resize(KT * d); W->dres.resize(KT * d); W->dln.resize(KT * d); W->datty.resize(KT * d);
    W->dqkv.resize(KT * 3 * d); W->dfch.resize(KT * D.dff); W->dfchg.resize(KT * D.dff);
    W->dtin.resize(K * D.in_dim); W->dvf.resize(K * D.feat); W->da3.resize(K * D.enc_flat);
    W->da2.resize((size_t)D.c2 * D.h2 * D.h2); W->da1.resize((size_t)D.c1 * D.h1 * D.h1);
    W->dph2.resize(K * D.pred_flat); W->dph1.resize(PR * D.ph); W->dpin.resize(PR * D.pin_dim);
    if (D.patch) {
        W->tss.resize(KT); W->tse.resize(KT);
        W->tpx.resize(K * D.npch * (size_t)D.pdim);
        W->pemb.resize(K * D.npch * d); W->dpemb.resize(K * D.npch * d);
        W->zm.resize(K * d); W->dzm.resize(K * d);
        W->pdel.resize(K * D.npch * (size_t)D.cpx); W->dpdel.resize(K * D.npch * (size_t)D.cpx);
    }
    if (D.act) {
        for (auto* v : {&W->ax_in, &W->dax_in}) v->resize(K * (size_t)D.pin_dim);
        for (auto* v : {&W->ax_h, &W->dax_h}) v->resize(K * (size_t)D.ph);
        for (auto* v : {&W->ax_drv, &W->dax_drv}) v->resize(K * 2);
        for (auto* v : {&W->ax_frm, &W->dax_frm}) v->resize(K * (size_t)D.cfx);
        W->ax_cur.resize(K * (size_t)D.cfx);
    }
    if (D.uni) {
        W->u_in.resize(KT * (size_t)D.VI); W->u_tgt.resize(KT * (size_t)D.V); W->u_use.resize(KT);
        W->u_row.resize(D.OV); W->u_prob.resize(D.OV);
    }
    W->dpd1.resize((size_t)D.pc1 * D.p1 * D.p1);
    W->cur_small.resize(D.small_px);
    W->cur_small_seq.resize(K * D.small_px);
    if (D.skip) W->dskip.resize(K * (size_t)D.c2 * D.h2 * D.h2);
    if (D.skip) W->dskip3.resize(K * (size_t)D.enc_flat);
    if (D.ih) {
        W->iv_in.resize(K * 2 * (size_t)D.feat); W->div_in.resize(K * 2 * (size_t)D.feat);
        W->iv_h.resize(K * (size_t)D.ih); W->div_h.resize(K * (size_t)D.ih);
        W->iv_out.resize(K * 2); W->div_out.resize(K * 2);
    }
    W->dvf_extra.resize(K * (size_t)D.feat);
    if (D.dh) {
        const size_t F = D.feat, Hd = D.dh, R = D.rpx, NP = K * (K + 1) / 2;   // chain steps per sequence <= K(K+1)/2
        W->y_t.resize(NP); W->y_j.resize(NP); W->y_par.resize(NP); W->y_lev.resize(K + 1);
        for (auto* v : {&W->y_uT, &W->g_uT}) v->resize(NP * (F + 3));
        for (auto* v : {&W->y_uH, &W->g_uH}) v->resize(NP * (F + 4));
        for (auto* v : {&W->y_hT, &W->y_hH, &W->y_hX, &W->g_hT, &W->g_hH, &W->g_hX}) v->resize(NP * Hd);
        for (auto* v : {&W->y_s, &W->g_s}) v->resize(NP * F);
        for (auto* v : {&W->y_h, &W->g_h}) v->resize(NP);
        for (auto* v : {&W->y_x, &W->g_x}) v->resize(NP * R);
        for (auto* v : {&W->y_hX0, &W->g_hX0}) v->resize(K * Hd);
        for (auto* v : {&W->y_x0, &W->g_x0}) v->resize(K * R);
        W->y_small.resize(D.small_px);
        W->y_tgt.resize(R);
        W->scr.resize(std::max(W->scr.size(), NP * std::max(std::max(R, Hd), F + 4)));
    }
    return W;
}

struct TEntry { int pid, rows, cols, count; };
struct Model {
    VMConfig cfg;
    Dims D;
    std::vector<unsigned char> allow;   // attention regions (task 00044): [ntok * ntok], empty = none
    std::vector<VMSlot> uslot;          // the universal model's slot layout (tasks 00047, 00048): [ntok]
    long off[NUM_P], sz[NUM_P], n_params = 0;
    std::vector<float> params, grads, wt, m, v;
    std::vector<TEntry> tlist;
    long step = 0;
    Rng rng;
    std::vector<Workspace*> ws;
    std::vector<std::vector<float>> tg;
    std::vector<char> tdirty, tused;
};

static void model_layout(Model* M) {
    const Dims& D = M->D;
    const long d = D.d, L = D.L;
    long s[NUM_P];
    s[C1W] = (long)D.c1 * D.in_ch * KK; s[C1B] = D.c1;
    s[C2W] = (long)D.c2 * D.c1 * KK; s[C2B] = D.c2;
    s[C3W] = (long)D.c3 * D.c2 * KK; s[C3B] = D.c3;
    s[EFW] = (long)D.feat * D.enc_flat; s[EFB] = D.feat;
    s[INW] = d * D.in_dim; s[INB] = d; s[WPE] = (long)D.K * d;
    s[LN1W] = L * d; s[LN1B] = L * d; s[QKVW] = L * 3 * d * d; s[QKVB] = L * 3 * d;
    s[APW] = L * d * d; s[APB] = L * d; s[LN2W] = L * d; s[LN2B] = L * d;
    s[FCW] = L * D.dff * d; s[FCB] = L * D.dff; s[FPW] = L * d * D.dff; s[FPB] = L * d;
    s[LNFW] = d; s[LNFB] = d;
    s[PIW] = 3L * (D.pih ? D.pih : d); s[PIB] = 3; s[PHW] = (long)D.pih * d; s[PHB] = D.pih;
    s[LOGSTD] = 2; s[VW] = d; s[VB] = 1;
    s[PR1W] = (long)D.ph * D.pin_dim; s[PR1B] = D.ph;
    s[PR2W] = (long)D.pred_flat * D.ph; s[PR2B] = D.pred_flat;
    s[PBW] = (long)D.pb * D.ph; s[PBB] = D.pb;
    s[D1W] = (long)D.d1_in * D.pc1 * KK; s[D1B] = D.pc1;
    s[D2W] = (long)D.d2_in * 3 * KK; s[D2B] = 3;
    s[DRW] = 2L * D.ph; s[DRB] = 2;
    s[IV1W] = (long)D.ih * 2 * D.feat; s[IV1B] = D.ih; s[IV2W] = 2L * D.ih; s[IV2B] = D.ih ? 2 : 0;
    const long dh = D.dh, F = D.feat;
    s[T1W] = dh * (F + 3); s[T1B] = dh; s[T2W] = F * dh; s[T2B] = dh ? F : 0;
    s[H1W] = dh * (F + 4); s[H1B] = dh; s[H2W] = dh; s[H2B] = dh ? 1 : 0;
    s[X1W] = dh * F; s[X1B] = dh; s[X2W] = (long)D.rpx * dh; s[X2B] = dh ? D.rpx : 0;
    const bool pm = D.patch > 0;   // patch tokens: no CNN encoder or decoder; PR2W outputs one patch's cell
    if (pm) {
        for (int p : {C1W, C1B, C2W, C2B, C3W, C3B, EFW, EFB, INW, INB, PBW, PBB, D1W, D1B, D2W, D2B}) s[p] = 0;
        s[PR2W] = (long)D.cpx * D.ph; s[PR2B] = D.cpx;
    }
    s[PEW] = pm ? d * D.pdim : 0; s[PEB] = pm ? d : 0; s[POS2D] = pm ? (long)D.npch * d : 0; s[TYP] = pm ? 4 * d : 0;
    s[DEW] = pm ? 2 * d : 0; s[DEB] = pm ? d : 0; s[MEW] = pm ? 3 * d : 0; s[MEB] = pm ? d : 0;
    s[CEW] = pm ? d : 0; s[CEB] = pm ? d : 0;
    const long ac = D.act;
    s[TYP] = pm ? (4 + ac) * d : 0;
    s[ACTE] = ac * d; s[AX1W] = ac * D.ph * D.pin_dim; s[AX1B] = ac * D.ph; s[AXDW] = ac * 2 * D.ph; s[AXDB] = ac * 2;
    s[AXFW] = ac * D.cfx * D.ph; s[AXFB] = ac * D.cfx;
    const long un = D.uni;
    if (un)   // the universal model: one input map, slot embeddings, one output head; no per-type heads
        for (int p : {PIW, PIB, PHW, PHB, VW, VB, PR1W, PR1B, PR2W, PR2B, DRW, DRB, PEW, PEB, POS2D, TYP, DEW, DEB,
                      MEW, MEB, CEW, CEB}) s[p] = 0;
    s[UIW] = un * d * D.VI; s[UIB] = un * d; s[USL] = un * D.ntok * d; s[UOW] = un * D.OV * d; s[UOB] = un * D.OV;
    long o = 0;
    for (int i = 0; i < NUM_P; i++) { M->off[i] = o; M->sz[i] = s[i]; o += s[i]; }
    M->n_params = o;
    if (D.uni) {
        M->tlist = {{QKVW, 3 * D.d, D.d, D.L}, {APW, D.d, D.d, D.L}, {FCW, D.dff, D.d, D.L}, {FPW, D.d, D.dff, D.L},
                    {UIW, D.d, D.VI, 1}, {UOW, D.OV, D.d, 1}};
        return;
    }
    if (pm) {
        M->tlist = {
            {QKVW, 3 * D.d, D.d, D.L}, {APW, D.d, D.d, D.L}, {FCW, D.dff, D.d, D.L}, {FPW, D.d, D.dff, D.L},
            {PIW, 3, D.pih ? D.pih : D.d, 1}, {PHW, D.pih, D.d, 1}, {PR1W, D.ph, D.pin_dim, 1}, {PR2W, D.cpx, D.ph, 1},
            {DRW, 2, D.ph, 1}, {PEW, D.d, D.pdim, 1}};
        if (D.act) M->tlist.insert(M->tlist.end(), {{AX1W, D.ph, D.pin_dim, 1}, {AXDW, 2, D.ph, 1}, {AXFW, D.cfx, D.ph, 1}});
        return;
    }
    M->tlist = {
        {EFW, D.feat, D.enc_flat, 1}, {INW, D.d, D.in_dim, 1}, {QKVW, 3 * D.d, D.d, D.L}, {APW, D.d, D.d, D.L},
        {FCW, D.dff, D.d, D.L}, {FPW, D.d, D.dff, D.L}, {PIW, 3, D.pih ? D.pih : D.d, 1}, {PHW, D.pih, D.d, 1},
        {PR1W, D.ph, D.pin_dim, 1},
        {PR2W, D.pred_flat, D.ph, 1}, {DRW, 2, D.ph, 1}, {C2W, D.c2, D.c1 * KK, 1}, {C3W, D.c3, D.c2 * KK, 1},
        {D1W, D.d1_in, D.pc1 * KK, 1}, {D2W, D.d2_in, 3 * KK, 1}, {PBW, D.pb, D.ph, 1},
        {IV1W, D.ih, 2 * D.feat, 1}, {IV2W, D.ih ? 2 : 0, D.ih, 1},
        {T1W, D.dh, D.feat + 3, 1}, {T2W, D.dh ? D.feat : 0, D.dh, 1}, {H1W, D.dh, D.feat + 4, 1},
        {H2W, D.dh ? 1 : 0, D.dh, 1}, {X1W, D.dh, D.feat, 1}, {X2W, D.dh ? D.rpx : 0, D.dh, 1}};
}

static void refresh_wt(Model* M) {
    struct Job { const float* src; float* dst; int rows, cols; };
    std::vector<Job> jobs;
    for (const TEntry& e : M->tlist)
        for (int c = 0; c < e.count; c++) {
            long o = M->off[e.pid] + (long)c * e.rows * e.cols;
            jobs.push_back({M->params.data() + o, M->wt.data() + o, e.rows, e.cols});
        }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < (int)jobs.size(); j++) transpose(jobs[j].src, jobs[j].rows, jobs[j].cols, jobs[j].dst);
}

static void ensure_threads(Model* M) {
    int nt = omp_get_max_threads();
    while ((int)M->ws.size() < nt) {
        M->ws.push_back(ws_create(M->D));
        M->tg.emplace_back((size_t)M->n_params, 0.f);
        M->tdirty.push_back(0);
        M->tused.push_back(0);
    }
}

static void fill_normal(Rng& r, float* p, long n, float std) {
    for (long i = 0; i < n; i++) p[i] = rng_normal(r) * std;
}
static void model_init_params(Model* M, uint64_t seed) {
    const Dims& D = M->D;
    Rng r;
    rng_seed(r, seed);
    float* P = M->params.data();
    std::fill(M->params.begin(), M->params.end(), 0.f);
    const long d = D.d, L = D.L;
    fill_normal(r, P + M->off[C1W], M->sz[C1W], sqrtf(2.0f / (D.in_ch * KK)));
    fill_normal(r, P + M->off[C2W], M->sz[C2W], sqrtf(2.0f / (D.c1 * KK)));
    fill_normal(r, P + M->off[C3W], M->sz[C3W], sqrtf(2.0f / (D.c2 * KK)));
    fill_normal(r, P + M->off[EFW], M->sz[EFW], sqrtf(2.0f / D.enc_flat));
    fill_normal(r, P + M->off[INW], M->sz[INW], sqrtf(1.0f / D.in_dim));
    fill_normal(r, P + M->off[WPE], M->sz[WPE], 0.02f);
    for (long i = 0; i < L * d; i++) { P[M->off[LN1W] + i] = 1.f; P[M->off[LN2W] + i] = 1.f; }
    for (long i = 0; i < d; i++) P[M->off[LNFW] + i] = 1.f;
    fill_normal(r, P + M->off[QKVW], M->sz[QKVW], 0.02f);
    fill_normal(r, P + M->off[APW], M->sz[APW], 0.02f / sqrtf(2.0f * L));
    fill_normal(r, P + M->off[FCW], M->sz[FCW], 0.02f);
    fill_normal(r, P + M->off[FPW], M->sz[FPW], 0.02f / sqrtf(2.0f * L));
    fill_normal(r, P + M->off[PHW], M->sz[PHW], sqrtf(2.0f / d));   // no draws when pi_hidden = 0
    fill_normal(r, P + M->off[PIW], M->sz[PIW], 0.01f / sqrtf((float)(D.pih ? D.pih : d)));
    P[M->off[LOGSTD] + 0] = M->cfg.logstd_init;
    P[M->off[LOGSTD] + 1] = M->cfg.logstd_init;
    fill_normal(r, P + M->off[VW], M->sz[VW], 1.0f / sqrtf((float)d));
    fill_normal(r, P + M->off[PR1W], M->sz[PR1W], sqrtf(2.0f / D.pin_dim));
    // patch tokens: PR2W outputs a cell's change directly, so it starts small (near "nothing changes")
    fill_normal(r, P + M->off[PR2W], M->sz[PR2W], D.patch ? 0.1f * sqrtf(1.0f / D.ph) : sqrtf(2.0f / D.ph));
    fill_normal(r, P + M->off[PBW], M->sz[PBW], sqrtf(2.0f / D.ph));
    fill_normal(r, P + M->off[D1W], M->sz[D1W], sqrtf(2.0f / (D.d1_in * 4)));
    fill_normal(r, P + M->off[D2W], M->sz[D2W], sqrtf(1.0f / (D.d2_in * 4)));
    fill_normal(r, P + M->off[DRW], M->sz[DRW], sqrtf(1.0f / D.ph));
    if (D.ih) {
        fill_normal(r, P + M->off[IV1W], M->sz[IV1W], sqrtf(2.0f / (2 * D.feat)));
        fill_normal(r, P + M->off[IV2W], M->sz[IV2W], sqrtf(1.0f / D.ih));
    }
    if (D.dh) {
        fill_normal(r, P + M->off[T1W], M->sz[T1W], sqrtf(2.0f / (D.feat + 3)));
        fill_normal(r, P + M->off[T2W], M->sz[T2W], 0.1f * sqrtf(1.0f / D.dh));   // near-identity transition
        fill_normal(r, P + M->off[H1W], M->sz[H1W], sqrtf(2.0f / (D.feat + 4)));
        fill_normal(r, P + M->off[H2W], M->sz[H2W], sqrtf(1.0f / D.dh));
        fill_normal(r, P + M->off[X1W], M->sz[X1W], sqrtf(2.0f / D.feat));
        fill_normal(r, P + M->off[X2W], M->sz[X2W], sqrtf(1.0f / D.dh));
    }
    if (D.patch && !D.uni) {
        fill_normal(r, P + M->off[PEW], M->sz[PEW], sqrtf(1.0f / D.pdim));
        fill_normal(r, P + M->off[TYP], M->sz[TYP], 0.1f);
        fill_normal(r, P + M->off[DEW], M->sz[DEW], sqrtf(1.0f / 2));
        fill_normal(r, P + M->off[MEW], M->sz[MEW], sqrtf(1.0f / 3));
        fill_normal(r, P + M->off[CEW], M->sz[CEW], 1.0f);
        // pos2d starts as a smooth 2D sine-cosine code of the cell's grid position (as in MAE), amplitude 0.1:
        // x in dims [0, d/2), y in [d/2, d); learned from there
        const int q = (int)(d / 4);
        for (int k = 0; k < D.npch; k++) {
            const float gx = (float)(k % D.gp), gy = (float)(k / D.gp);
            float* pe = P + M->off[POS2D] + (size_t)k * d;
            for (int i = 0; i < q; i++) {
                const float w = powf(10000.f, -(float)i / q);
                pe[i] = 0.1f * sinf(gx * w); pe[q + i] = 0.1f * cosf(gx * w);
                pe[2 * q + i] = 0.1f * sinf(gy * w); pe[3 * q + i] = 0.1f * cosf(gy * w);
            }
        }
    }
    if (D.act) {   // drawn after everything else: models without the act token are unchanged
        fill_normal(r, P + M->off[ACTE], M->sz[ACTE], 0.1f);
        fill_normal(r, P + M->off[AX1W], M->sz[AX1W], sqrtf(2.0f / D.pin_dim));
        fill_normal(r, P + M->off[AXDW], M->sz[AXDW], sqrtf(1.0f / D.ph));
        fill_normal(r, P + M->off[AXFW], M->sz[AXFW], 0.1f * sqrtf(1.0f / D.ph));   // residual: near "no change"
    }
    if (D.uni) {   // drawn after everything else: the other models are unchanged
        // inputs in [-0.5, 0.5]: about unit scale; bin embeddings: GPT-2's 0.02, as the slots (vm-00029)
        fill_normal(r, P + M->off[UIW], M->sz[UIW], D.ib ? 0.02f : sqrtf(3.0f / D.V));
        fill_normal(r, P + M->off[USL], M->sz[USL], 0.02f);
        fill_normal(r, P + M->off[UOW], M->sz[UOW], 0.02f);                // near-uniform bins at birth
        // patch and summary slots add the 2D sine-cosine code of their centre, as pos2d (task 00052: by the slot
        // table; before, slot k + 1 got cell k's code, which summaries shifted)
        const int q = (int)(d / 4);
        for (int k = 0; k < D.ntok; k++) {
            float gx, gy;
            if (!vm_uni_center(&M->cfg, &M->uslot[k], &gy, &gx)) continue;
            float* pe = P + M->off[USL] + (size_t)k * d;
            for (int i = 0; i < q; i++) {
                const float w = powf(10000.f, -(float)i / q);
                pe[i] += 0.1f * sinf(gx * w); pe[q + i] += 0.1f * cosf(gx * w);
                pe[2 * q + i] += 0.1f * sinf(gy * w); pe[3 * q + i] += 0.1f * cosf(gy * w);
            }
        }
    }
}

static bool config_valid(const VMConfig& c) {
    if (c.img <= 0 || c.img % 8) return false;
    if (c.pred_img <= 0 || c.pred_img % 4 || c.img % c.pred_img) return false;
    if (c.d_model <= 0 || c.n_head <= 0 || c.d_model % c.n_head) return false;
    if (c.ctx <= 0 || c.n_layer <= 0 || c.c1 <= 0 || c.c2 <= 0 || c.c3 <= 0 || c.feat <= 0) return false;
    if (c.d_ff <= 0 || c.pc1 <= 0 || c.pc2 <= 0 || c.pred_hidden <= 0) return false;
    if (c.coord && c.pb < 0) return false;
    if (c.dyn_hidden < 0 || c.one_frame < 0 || c.one_frame > 1) return false;
    if (c.pred_skip && (c.img / 4 != c.pred_img / 2 || c.img / 8 != c.pred_img / 4)) return false;  // map sizes
    if (c.patch < 0 || c.pi_hidden < 0) return false;
    if (c.patch > 0) {   // each patch predicts its own cell of the next frame; no inverse or latent dynamics heads
        if (c.img % c.patch || c.pred_img % (c.img / c.patch)) return false;
        if (c.inv_hidden > 0 || c.dyn_hidden > 0) return false;
    }
    if (c.act_token < 0 || c.act_token > 1) return false;
    if (c.act_token && (c.patch <= 0 || !c.pred_residual || c.pred_img % ACT_CG)) return false;
    if (c.universal < 0 || c.universal > 1) return false;
    if (c.universal && (c.patch <= 0 || c.act_token || c.bins < 2 || c.patch * c.patch * (c.gray ? 1 : 3) < 3)) return false;
    if (c.gray < 0 || c.gray > 1 || (c.gray && !c.universal)) return false;
    if (c.in_bins < 0 || c.in_bins > 1 || (c.in_bins && !c.universal)) return false;
    if (!vm_attn_valid(&c)) return false;
    return true;
}
static Model* model_new(const VMConfig& cfg) {
    if (!config_valid(cfg)) { fprintf(stderr, "vm_agent: invalid config\n"); return nullptr; }
    Model* M = new Model();
    M->cfg = cfg;
    M->D = make_dims(cfg);
    if (const int nt = vm_attn_allow(&cfg, nullptr)) {   // attention regions (task 00044)
        M->allow.resize((size_t)nt * nt);
        vm_attn_allow(&cfg, M->allow.data());
    }
    if (M->D.uni) {
        M->uslot.resize(M->D.ntok);
        vm_uni_layout(&cfg, M->uslot.data());
    }
    model_layout(M);
    M->params.assign(M->n_params, 0.f);
    M->grads.assign(M->n_params, 0.f);
    M->wt.assign(M->n_params, 0.f);
    M->m.assign(M->n_params, 0.f);
    M->v.assign(M->n_params, 0.f);
    rng_seed(M->rng, 0x5EED);
    return M;
}

// =====================================================================================
// Sequence forward / loss / backward (training + analysis)
// =====================================================================================
// A view of one sequence of T tokens with per-token strides (elements of the pointed-to type).
struct SeqView {
    int T = 0;
    const uint8_t* img = nullptr; long img_st = 0;
    const float* drives = nullptr; long drv_st = 0;
    const float* prev_action = nullptr; long pa_st = 0;
    const float* prev_reward = nullptr; long pr_st = 0;
    const uint8_t* ep_start = nullptr; long es_st = 0;
    const float* action = nullptr; long act_st = 0;         // raw sampled actions (training)
    const float* pred_action = nullptr; long pact_st = 0;   // analysis: actions conditioning the prediction
    const float* old_logp = nullptr; const float* old_value = nullptr;
    const float* adv = nullptr; const float* ret = nullptr; long sc_st = 0;
    const uint8_t* next_small = nullptr; long ns_st = 0;
    const float* next_drives = nullptr; long nd_st = 0;
};

// Two constant channels: x and y pixel-center coordinates mapped to (-1, 1). dst: [2, S, S]
static void fill_coords(float* dst, int S) {
    for (int y = 0; y < S; y++)
        for (int x = 0; x < S; x++) {
            dst[(size_t)y * S + x] = (x + 0.5f) / S * 2.0f - 1.0f;
            dst[(size_t)S * S + (size_t)y * S + x] = (y + 0.5f) / S * 2.0f - 1.0f;
        }
}

static void encode_frame(const Model* M, Workspace* W, float* scr, const uint8_t* img, int t) {
    const Dims& D = M->D;
    const float* P = M->params.data();
    const int S = D.img;
    float* x0 = W->x0.data() + (size_t)t * D.in_ch * S * S;
    for (int y = 0; y < S; y++)
        for (int x = 0; x < S; x++)
            for (int c = 0; c < 3; c++) x0[((size_t)c * S + y) * S + x] = img[((size_t)y * S + x) * 3 + c] / 255.0f;
    if (D.coord) fill_coords(x0 + (size_t)3 * S * S, S);
    float* a1 = W->a1.data() + (size_t)t * D.c1 * D.h1 * D.h1;
    float* a2 = W->a2.data() + (size_t)t * D.c2 * D.h2 * D.h2;
    float* a3 = W->a3.data() + (size_t)t * D.enc_flat;
    conv_fwd(x0, D.in_ch, S, P + M->off[C1W], P + M->off[C1B], D.c1, a1, scr);
    conv_fwd(a1, D.c1, D.h1, P + M->off[C2W], P + M->off[C2B], D.c2, a2, scr);
    conv_fwd(a2, D.c2, D.h2, P + M->off[C3W], P + M->off[C3B], D.c3, a3, scr);
}

static void build_token_input(const Model* M, Workspace* W, int t, const float* drv, const float* pa, float pr) {
    const Dims& D = M->D;
    float* u = W->tin.data() + (size_t)t * D.in_dim;
    const float* vf = W->vf.data() + (size_t)t * D.feat;
    for (int i = 0; i < D.feat; i++) u[i] = vf[i];
    u[D.feat + 0] = drv[0];
    u[D.feat + 1] = drv[1];
    u[D.feat + 2] = pa[0];
    u[D.feat + 3] = pa[1];
    u[D.feat + 4] = pa[2];
    u[D.feat + 5] = M->cfg.use_prev_reward ? pr : 0.f;
}

// The forward and backward passes of one sequence are split into per-frame parts (the convolutional encoder and
// frame decoder: the bulk of the work) and per-sequence parts (the Transformer, heads and losses). A per-frame part
// writes only frame t's slots of the sequence's workspace W and uses the scratch of the calling thread's workspace X
// (X == W when run serially), so the frames of one sequence can run on different threads (loss_and_grads_frames).
// Policy head: pi [T,3] = W2 . relu(W1 . z + b1) + b2 with the hidden layer h [T,pih] (pi_hidden > 0),
// else pi = W . z + b.
static void policy_fwd(const Model* M, const float* z, float* h, float* pi, int T) {
    const Dims& D = M->D;
    const float* WT = M->wt.data();
    const float* P = M->params.data();
    if (!D.pih) { linear_fwd(pi, z, WT + M->off[PIW], P + M->off[PIB], T, D.d, 3); return; }
    linear_fwd(h, z, WT + M->off[PHW], P + M->off[PHB], T, D.d, D.pih);
    for (int i = 0; i < T * D.pih; i++) h[i] = h[i] > 0.f ? h[i] : 0.f;
    linear_fwd(pi, h, WT + M->off[PIW], P + M->off[PIB], T, D.pih, 3);
}

// The Transformer layers and final LayerNorm on T token inputs W->xin -> W->lnf. Token t attends to tokens
// ss[t]..se[t] (se = NULL: ..t). Buffers are laid out for KT tokens (patch tokens: K * ntok, else K).
static void transformer_fwd(const Model* M, Workspace* W, int T, const int* ss, const int* se) {
    const Dims& D = M->D;
    const int K = D.KT, d = D.d;
    const float* P = M->params.data();
    const float* WT = M->wt.data();
    const float* x = W->xin.data();
    for (int l = 0; l < D.L; l++) {
        float* ln1 = W->ln1.data() + (size_t)l * K * d;
        float* ln1m = W->ln1m.data() + (size_t)l * K;
        float* ln1r = W->ln1r.data() + (size_t)l * K;
        float* qkv = W->qkv.data() + (size_t)l * K * 3 * d;
        float* att = W->att.data() + (size_t)l * D.nh * K * K;
        float* atty = W->atty.data() + (size_t)l * K * d;
        float* res1 = W->res1.data() + (size_t)l * K * d;
        float* ln2 = W->ln2.data() + (size_t)l * K * d;
        float* ln2m = W->ln2m.data() + (size_t)l * K;
        float* ln2r = W->ln2r.data() + (size_t)l * K;
        float* fch = W->fch.data() + (size_t)l * K * D.dff;
        float* fchg = W->fchg.data() + (size_t)l * K * D.dff;
        float* res2 = W->res2.data() + (size_t)l * K * d;
        float* proj = W->proj.data();
        layernorm_fwd(ln1, ln1m, ln1r, x, P + M->off[LN1W] + l * d, P + M->off[LN1B] + l * d, T, d);
        linear_fwd(qkv, ln1, WT + M->off[QKVW] + (size_t)l * 3 * d * d, P + M->off[QKVB] + l * 3 * d, T, d, 3 * d);
        attention_fwd(atty, att, qkv, ss, se, T, K, d, D.nh, M->allow.empty() ? nullptr : M->allow.data(), D.ntok);
        linear_fwd(proj, atty, WT + M->off[APW] + (size_t)l * d * d, P + M->off[APB] + l * d, T, d, d);
        for (int i = 0; i < T * d; i++) res1[i] = x[i] + proj[i];
        layernorm_fwd(ln2, ln2m, ln2r, res1, P + M->off[LN2W] + l * d, P + M->off[LN2B] + l * d, T, d);
        linear_fwd(fch, ln2, WT + M->off[FCW] + (size_t)l * D.dff * d, P + M->off[FCB] + l * D.dff, T, d, D.dff);
        gelu_fwd(fchg, fch, T * D.dff);
        linear_fwd(proj, fchg, WT + M->off[FPW] + (size_t)l * d * D.dff, P + M->off[FPB] + l * d, T, D.dff, d);
        for (int i = 0; i < T * d; i++) res2[i] = res1[i] + proj[i];
        x = res2;
    }
    layernorm_fwd(W->lnf.data(), W->lnfm.data(), W->lnfr.data(), x, P + M->off[LNFW], P + M->off[LNFB], T, d);
}

static void seq_fwd_setup(const Model* M, Workspace* W, const SeqView& sv) {
    for (int t = 0; t < sv.T; t++) {
        bool st = (t == 0) || M->cfg.one_frame || (sv.ep_start && sv.ep_start[(size_t)t * sv.es_st]);
        W->ss[t] = st ? t : W->ss[t - 1];
        W->pos[t] = t - W->ss[t];
    }
}

static void seq_fwd_encode(const Model* M, Workspace* W, Workspace* X, const SeqView& sv, int t) {
    encode_frame(M, W, X->scr.data(), sv.img + (size_t)t * sv.img_st, t);
}

// From the encoder's conv-3 maps to the heads and the decoder MLP (everything but the per-frame decoder convs).
static void seq_fwd_core(const Model* M, Workspace* W, const SeqView& sv, bool want_pred) {
    const Dims& D = M->D;
    const int T = sv.T, d = D.d;
    const float* P = M->params.data();
    const float* WT = M->wt.data();
    linear_fwd(W->vf.data(), W->a3.data(), WT + M->off[EFW], P + M->off[EFB], T, D.enc_flat, D.feat);
    for (int i = 0; i < T * D.feat; i++) W->vf[i] = W->vf[i] > 0.f ? W->vf[i] : 0.f;
    for (int t = 0; t < T; t++)
        build_token_input(M, W, t, sv.drives + (size_t)t * sv.drv_st, sv.prev_action + (size_t)t * sv.pa_st,
                          sv.prev_reward[(size_t)t * sv.pr_st]);
    linear_fwd(W->xin.data(), W->tin.data(), WT + M->off[INW], P + M->off[INB], T, D.in_dim, d);
    for (int t = 0; t < T; t++) {
        const float* pe = P + M->off[WPE] + (size_t)W->pos[t] * d;
        float* x = W->xin.data() + (size_t)t * d;
        for (int i = 0; i < d; i++) x[i] += pe[i];
    }
    transformer_fwd(M, W, T, W->ss.data(), nullptr);
    policy_fwd(M, W->lnf.data(), W->pih.data(), W->pi.data(), T);
    for (int t = 0; t < T; t++)
        W->val[t] = value_head(W->lnf.data() + (size_t)t * d, P + M->off[VW], P[M->off[VB]], d);
    if (!want_pred) return;
    // prediction head: conditioned on the executed action (raw action clipped like the world does)
    for (int t = 0; t < T; t++) {
        float* pin = W->pin.data() + (size_t)t * D.pin_dim;
        memcpy(pin, W->lnf.data() + (size_t)t * d, sizeof(float) * d);
        const float* a = sv.pred_action ? sv.pred_action + (size_t)t * sv.pact_st : sv.action + (size_t)t * sv.act_st;
        pin[d + 0] = clampf_(a[0], -1.f, 1.f);
        pin[d + 1] = clampf_(a[1], -1.f, 1.f);
        pin[d + 2] = a[2] > 0.5f ? 1.f : 0.f;
    }
    linear_fwd(W->ph1.data(), W->pin.data(), WT + M->off[PR1W], P + M->off[PR1B], T, D.pin_dim, D.ph);
    for (int i = 0; i < T * D.ph; i++) W->ph1[i] = W->ph1[i] > 0.f ? W->ph1[i] : 0.f;
    linear_fwd(W->ph2.data(), W->ph1.data(), WT + M->off[PR2W], P + M->off[PR2B], T, D.ph, D.pred_flat);
    for (int i = 0; i < T * D.pred_flat; i++) W->ph2[i] = W->ph2[i] > 0.f ? W->ph2[i] : 0.f;
    if (D.coord && D.pb > 0) {
        linear_fwd(W->ub.data(), W->ph1.data(), WT + M->off[PBW], P + M->off[PBB], T, D.ph, D.pb);
        for (int i = 0; i < T * D.pb; i++) W->ub[i] = W->ub[i] > 0.f ? W->ub[i] : 0.f;
    }
    linear_fwd(W->pdrv.data(), W->ph1.data(), WT + M->off[DRW], P + M->off[DRB], T, D.ph, 2);
    for (int i = 0; i < T * 2; i++) W->pdrv[i] = sigmoidf_(W->pdrv[i]);
}

// The decoder convolutions of frame t (needs seq_fwd_core with want_pred).
static void seq_fwd_decode(const Model* M, Workspace* W, Workspace* X, const SeqView& sv, int t) {
    const Dims& D = M->D;
    const float* P = M->params.data();
    const float* WT = M->wt.data();
    const int p1n = D.pc1 * D.p1 * D.p1, q0 = D.p0 * D.p0, q1 = D.p1 * D.p1;
    {
        float* pd1 = W->pd1.data() + (size_t)t * p1n;
        float* pout = W->pout.data() + (size_t)t * D.small_px;
        const float* in1 = W->ph2.data() + (size_t)t * D.pred_flat;
        const float* in2 = pd1;
        const float* ae = W->pin.data() + (size_t)t * D.pin_dim + D.d;   // executed action (3)
        if (D.coord || D.skip) {
            float* x1 = W->d1in.data() + (size_t)t * D.d1_in * q0;
            memcpy(x1, in1, sizeof(float) * D.pred_flat);
            for (int k = 0; k < D.pb; k++) {
                const float u = W->ub[(size_t)t * D.pb + k];
                for (int i = 0; i < q0; i++) x1[(size_t)(D.pc2 + k) * q0 + i] = u;
            }
            if (D.coord) fill_coords(x1 + (size_t)(D.pc2 + D.pb) * q0, D.p0);
            if (D.skip) {
                memcpy(x1 + (size_t)D.skip1_off * q0, W->a3.data() + (size_t)t * D.enc_flat, sizeof(float) * D.enc_flat);
                for (int k = 0; k < 3; k++) {
                    float* dst = x1 + (size_t)(D.skip1_off + D.c3 + k) * q0;
                    for (int i = 0; i < q0; i++) dst[i] = ae[k];
                }
            }
            in1 = x1;
        }
        deconv_fwd(in1, D.d1_in, D.p0, WT + M->off[D1W], P + M->off[D1B], D.pc1, pd1, X->scr.data());
        for (int i = 0; i < p1n; i++) pd1[i] = pd1[i] > 0.f ? pd1[i] : 0.f;
        if (D.coord || D.skip) {
            float* x2 = W->d2in.data() + (size_t)t * D.d2_in * q1;
            memcpy(x2, pd1, sizeof(float) * p1n);
            if (D.coord) fill_coords(x2 + (size_t)D.pc1 * q1, D.p1);
            if (D.skip) {
                const float* a2 = W->a2.data() + (size_t)t * D.c2 * D.h2 * D.h2;
                memcpy(x2 + (size_t)D.skip_off * q1, a2, sizeof(float) * D.c2 * q1);
                for (int k = 0; k < 3; k++) {
                    float* dst = x2 + (size_t)(D.skip_off + D.c2 + k) * q1;
                    for (int i = 0; i < q1; i++) dst[i] = ae[k];
                }
            }
            in2 = x2;
        }
        deconv_fwd(in2, D.d2_in, D.p1, WT + M->off[D2W], P + M->off[D2B], 3, pout, X->scr.data());
        if (D.residual) {
            uint8_t* cs = W->cur_small_seq.data() + (size_t)t * D.small_px;   // HWC u8
            downsample_u8(sv.img + (size_t)t * sv.img_st, D.img, D.ds, cs);
            const int pp = D.pimg * D.pimg;
            for (int y = 0; y < D.pimg; y++)
                for (int x = 0; x < D.pimg; x++)
                    for (int c = 0; c < 3; c++)
                        pout[c * pp + y * D.pimg + x] += cs[(y * D.pimg + x) * 3 + c] / 255.0f;
        } else {
            for (int i = 0; i < D.small_px; i++) pout[i] = sigmoidf_(pout[i]);
        }
    }
}

// Patch tokens (patch > 0). Each step is ntok tokens: npch patch tokens, then the drives, previous-move and
// command tokens. A token attends to every token of its own step and of the earlier steps of its context block.
// Each token predicts its own part of the next state: a patch token its cell of the next frame, the drives token
// the next drives (both from [token output, executed action]), the move token the next move (the policy).
// Fills the same outputs as the CNN model (pi, val, pout, pdrv), so the losses are shared.
static void seq_fwd_patch(const Model* M, Workspace* W, const SeqView& sv, bool want_pred) {
    const Dims& D = M->D;
    const int T = sv.T, d = D.d, nt = D.ntok, np = D.npch, ps = D.patch, S = D.img;
    const float* P = M->params.data();
    const float* WT = M->wt.data();
    for (int t = 0; t < T; t++)
        for (int k = 0; k < nt; k++) {
            W->tss[t * nt + k] = W->ss[t] * nt;
            W->tse[t * nt + k] = t * nt + nt - 1;
        }
    for (int t = 0; t < T; t++) {   // patch pixels in [0, 1], (y, x, c) within the patch
        const uint8_t* img = sv.img + (size_t)t * sv.img_st;
        for (int p = 0; p < np; p++) {
            const int py = (p / D.gp) * ps, px = (p % D.gp) * ps;
            float* dst = W->tpx.data() + ((size_t)t * np + p) * D.pdim;
            for (int y = 0; y < ps; y++)
                for (int x = 0; x < ps; x++)
                    for (int c = 0; c < 3; c++)
                        dst[(y * ps + x) * 3 + c] = img[((size_t)(py + y) * S + px + x) * 3 + c] / 255.0f;
        }
    }
    linear_fwd(W->pemb.data(), W->tpx.data(), WT + M->off[PEW], P + M->off[PEB], T * np, D.pdim, d);
    const float* typ = P + M->off[TYP];
    const float *dw = P + M->off[DEW], *db = P + M->off[DEB], *mw = P + M->off[MEW], *mb = P + M->off[MEB];
    const float *cw = P + M->off[CEW], *cb = P + M->off[CEB];
    for (int t = 0; t < T; t++) {
        const float* pe = P + M->off[WPE] + (size_t)W->pos[t] * d;
        float* xs = W->xin.data() + (size_t)t * nt * d;
        for (int p = 0; p < np; p++) {
            const float* e = W->pemb.data() + ((size_t)t * np + p) * d;
            const float* p2 = P + M->off[POS2D] + (size_t)p * d;
            float* x = xs + (size_t)p * d;
            for (int i = 0; i < d; i++) x[i] = e[i] + p2[i] + typ[i] + pe[i];
        }
        const float* drv = sv.drives + (size_t)t * sv.drv_st;
        const float* pa = sv.prev_action + (size_t)t * sv.pa_st;
        const float cmd = M->cfg.use_prev_reward ? sv.prev_reward[(size_t)t * sv.pr_st] : 0.f;
        float* xd = xs + (size_t)np * d;
        float* xm = xd + d;
        float* xc = xm + d;
        for (int i = 0; i < d; i++) {
            xd[i] = dw[i * 2] * drv[0] + dw[i * 2 + 1] * drv[1] + db[i] + typ[d + i] + pe[i];
            xm[i] = mw[i * 3] * pa[0] + mw[i * 3 + 1] * pa[1] + mw[i * 3 + 2] * pa[2] + mb[i] + typ[2 * d + i] + pe[i];
            xc[i] = cw[i] * cmd + cb[i] + typ[3 * d + i] + pe[i];
        }
        if (D.act) {   // the act token: a learned query
            const float* ae = P + M->off[ACTE];
            float* xa = xc + d;
            for (int i = 0; i < d; i++) xa[i] = ae[i] + typ[4 * d + i] + pe[i];
        }
    }
    transformer_fwd(M, W, T * nt, W->tss.data(), W->tse.data());
    const int zi = D.act ? np + 3 : np + 1;   // the token the policy reads: the act token, else the move token
    for (int t = 0; t < T; t++)
        memcpy(W->zm.data() + (size_t)t * d, W->lnf.data() + ((size_t)t * nt + zi) * d, sizeof(float) * d);
    policy_fwd(M, W->zm.data(), W->pih.data(), W->pi.data(), T);
    for (int t = 0; t < T; t++) W->val[t] = value_head(W->zm.data() + (size_t)t * d, P + M->off[VW], P[M->off[VB]], d);
    if (!want_pred) return;
    // prediction rows: T * np patch rows (t, p), then T drives rows; each [token output, executed action]
    const int R = T * (np + 1);
    for (int r = 0; r < R; r++) {
        const int t = r < T * np ? r / np : r - T * np, k = r < T * np ? r % np : np;
        float* pin = W->pin.data() + (size_t)r * D.pin_dim;
        memcpy(pin, W->lnf.data() + ((size_t)t * nt + k) * d, sizeof(float) * d);
        const float* a = sv.pred_action ? sv.pred_action + (size_t)t * sv.pact_st : sv.action + (size_t)t * sv.act_st;
        pin[d + 0] = clampf_(a[0], -1.f, 1.f);
        pin[d + 1] = clampf_(a[1], -1.f, 1.f);
        pin[d + 2] = a[2] > 0.5f ? 1.f : 0.f;
    }
    linear_fwd(W->ph1.data(), W->pin.data(), WT + M->off[PR1W], P + M->off[PR1B], R, D.pin_dim, D.ph);
    for (int i = 0; i < R * D.ph; i++) W->ph1[i] = W->ph1[i] > 0.f ? W->ph1[i] : 0.f;
    linear_fwd(W->pdel.data(), W->ph1.data(), WT + M->off[PR2W], P + M->off[PR2B], T * np, D.ph, D.cpx);
    linear_fwd(W->pdrv.data(), W->ph1.data() + (size_t)T * np * D.ph, WT + M->off[DRW], P + M->off[DRB], T, D.ph, 2);
    for (int i = 0; i < T * 2; i++) W->pdrv[i] = sigmoidf_(W->pdrv[i]);
    const int pp = D.pimg * D.pimg, cs = D.cell;
    for (int t = 0; t < T; t++) {   // the predicted frame (CHW), as the CNN decoder's
        float* pout = W->pout.data() + (size_t)t * D.small_px;
        for (int p = 0; p < np; p++) {
            const float* dl = W->pdel.data() + ((size_t)t * np + p) * D.cpx;
            const int oy = (p / D.gp) * cs, ox = (p % D.gp) * cs;
            for (int y = 0; y < cs; y++)
                for (int x = 0; x < cs; x++)
                    for (int c = 0; c < 3; c++) pout[c * pp + (oy + y) * D.pimg + ox + x] = dl[(y * cs + x) * 3 + c];
        }
        if (D.residual) {
            uint8_t* cur = W->cur_small_seq.data() + (size_t)t * D.small_px;   // HWC u8
            downsample_u8(sv.img + (size_t)t * sv.img_st, D.img, D.ds, cur);
            for (int y = 0; y < D.pimg; y++)
                for (int x = 0; x < D.pimg; x++)
                    for (int c = 0; c < 3; c++) pout[c * pp + y * D.pimg + x] += cur[(y * D.pimg + x) * 3 + c] / 255.0f;
        } else {
            for (int i = 0; i < D.small_px; i++) pout[i] = sigmoidf_(pout[i]);
        }
    }
    if (!D.act) return;
    // the act token's auxiliary heads on [its output, executed move]: next drives, coarse next frame (residual)
    for (int t = 0; t < T; t++) {
        float* in = W->ax_in.data() + (size_t)t * D.pin_dim;
        memcpy(in, W->zm.data() + (size_t)t * d, sizeof(float) * d);
        memcpy(in + d, W->pin.data() + (size_t)t * np * D.pin_dim + d, sizeof(float) * 3);   // a_t, as the patch rows'
        downsample_u8(W->cur_small_seq.data() + (size_t)t * D.small_px, D.pimg, D.cds, W->ax_cur.data() + (size_t)t * D.cfx);
    }
    linear_fwd(W->ax_h.data(), W->ax_in.data(), WT + M->off[AX1W], P + M->off[AX1B], T, D.pin_dim, D.ph);
    for (int i = 0; i < T * D.ph; i++) W->ax_h[i] = W->ax_h[i] > 0.f ? W->ax_h[i] : 0.f;
    linear_fwd(W->ax_drv.data(), W->ax_h.data(), WT + M->off[AXDW], P + M->off[AXDB], T, D.ph, 2);
    for (int i = 0; i < T * 2; i++) W->ax_drv[i] = sigmoidf_(W->ax_drv[i]);
    linear_fwd(W->ax_frm.data(), W->ax_h.data(), WT + M->off[AXFW], P + M->off[AXFB], T, D.ph, D.cfx);
    for (int i = 0; i < T * D.cfx; i++) W->ax_frm[i] += W->ax_cur[i] / 255.0f;
}

// ---- the universal model (universal = 1, task 00023) ----
// Each step is ntok = npch + 2 (+ nsum) tokens: the body (hunger, restlessness), the patches row by row (pixels / 255, (y, x, c)
// within the patch), the move made at this step ((dx + 1) / 2, (dy + 1) / 2, click). Every token is a value vector of V
// numbers in [0, 1] (body 2 used, move 3, patch V), centred to [-0.5, 0.5] at the input, unused entries 0. Fully causal
// within the context; every token predicts the next token's values: the move token the drives after its step.
// Summary tokens (task 00047, vm_uni_slot) have no values: the token before one has no target.

// The token values of step t, slot k, into v [V] (raw, in [0, 1]); returns the number used.
// Bin-embedding input (task 00036): the first u of V values in [0, 1], each split two-hot over K bins (centres
// k / (K - 1)), into out [V * K]; unused values and bins 0.
static void uni_twohot(const float* v, int u, int V, int K, float* out) {
    for (int i = 0; i < V * K; i++) out[i] = 0.f;
    for (int j = 0; j < u; j++) {
        const float p = fminf(fmaxf(v[j], 0.f), 1.f) * (K - 1);
        const int lo = std::min((int)floorf(p), K - 2);
        const float w = p - lo;
        out[(size_t)j * K + lo] = 1.f - w;
        out[(size_t)j * K + lo + 1] = w;
    }
}

static int uni_values(const Model* M, const SeqView& sv, int t, int k, float* v) {
    const Dims& D = M->D;
    const int V = D.V, ps = D.patch, S = D.img;
    for (int i = 0; i < V; i++) v[i] = 0.f;
    const int kind = M->uslot[k].kind, p = M->uslot[k].idx;
    if (kind == 2) {   // a summary token: no values, or its block box-averaged to one patch (task 00049)
        if (!M->cfg.sum_thumb) return 0;
        int by0, bx0, fy, fx;
        vm_uni_block(&M->cfg, &M->uslot[k], &by0, &bx0, &fy, &fx);
        const uint8_t* img = sv.img + (size_t)t * sv.img_st;
        const int ch = D.ch;
        const float sc = 1.0f / (255.0f * fy * fx);
        for (int y = 0; y < ps; y++)
            for (int x = 0; x < ps; x++)
                for (int c = 0; c < ch; c++) {
                    float a = 0.f;
                    for (int i = 0; i < fy; i++)
                        for (int j = 0; j < fx; j++)
                            a += img[((size_t)(by0 + y * fy + i) * S + bx0 + x * fx + j) * 3 + c];
                    v[(y * ps + x) * ch + c] = a * sc;
                }
        return V;
    }
    if (k == 0) {
        const float* dr = sv.drives + (size_t)t * sv.drv_st;
        v[0] = dr[0]; v[1] = dr[1];
        return 2;
    }
    if (k == D.ntok - 1) {
        const float* a = sv.action + (size_t)t * sv.act_st;
        v[0] = 0.5f * (clampf_(a[0], -1.f, 1.f) + 1.f);
        v[1] = 0.5f * (clampf_(a[1], -1.f, 1.f) + 1.f);
        v[2] = a[2] > 0.5f ? 1.f : 0.f;
        return 3;
    }
    const int py = (p / D.gp) * ps, px = (p % D.gp) * ps, ch = D.ch;   // gray: the first channel only
    const uint8_t* img = sv.img + (size_t)t * sv.img_st;
    for (int y = 0; y < ps; y++)
        for (int x = 0; x < ps; x++)
            for (int c = 0; c < ch; c++) v[(y * ps + x) * ch + c] = img[((size_t)(py + y) * S + px + x) * 3 + c] / 255.0f;
    return V;
}

// Tokens, targets and the Transformer (after seq_fwd_setup). Targets need sv.next_drives (training) for the move
// tokens; without it their targets are left 0 (forward passes for acting / analysis).
static void seq_fwd_uni(const Model* M, Workspace* W, const SeqView& sv) {
    const Dims& D = M->D;
    const int T = sv.T, nt = D.ntok, V = D.V, d = D.d;
    const float* P = M->params.data();
    for (int t = 0; t < T; t++)
        for (int k = 0; k < nt; k++) {
            const int i = t * nt + k;
            W->tss[i] = W->ss[t] * nt;
            float* in = W->u_in.data() + (size_t)i * D.VI;
            float* val = D.ib ? W->u_row.data() : in;
            const int u = uni_values(M, sv, t, k, val);
            if (k > 0) {   // this token is the target of the one before it (in the same step)
                memcpy(W->u_tgt.data() + (size_t)(i - 1) * V, val, sizeof(float) * V);
                W->u_use[i - 1] = u;
            }
            if (D.ib) uni_twohot(val, u, V, D.nbin, in);
            else for (int j = 0; j < u; j++) in[j] -= 0.5f;
            if (k == nt - 1) {   // the move token predicts the drives after its step
                float* tg = W->u_tgt.data() + (size_t)i * V;
                for (int j = 0; j < V; j++) tg[j] = 0.f;
                if (sv.next_drives) { tg[0] = sv.next_drives[(size_t)t * sv.nd_st]; tg[1] = sv.next_drives[(size_t)t * sv.nd_st + 1]; }
                W->u_use[i] = 2;
            }
        }
    linear_fwd(W->xin.data(), W->u_in.data(), M->wt.data() + M->off[UIW], P + M->off[UIB], T * nt, D.VI, d);
    for (int t = 0; t < T; t++) {
        const float* pe = P + M->off[WPE] + (size_t)W->pos[t] * d;
        for (int k = 0; k < nt; k++) {
            float* x = W->xin.data() + ((size_t)t * nt + k) * d;
            const float* sl = P + M->off[USL] + (size_t)k * d;
            for (int j = 0; j < d; j++) x[j] += sl[j] + pe[j];
        }
    }
    transformer_fwd(M, W, T * nt, W->tss.data(), nullptr);
}

// One token's head: logits [OV] for its final output z, then the per-value softmax into prob [OV].
static void uni_head_row(const Model* M, const float* z, float* row, float* prob) {
    const Dims& D = M->D;
    linear_fwd(row, z, M->wt.data() + M->off[UOW], M->params.data() + M->off[UOB], 1, D.d, D.OV);
    const int K = D.nbin;
    for (int u = 0; u < D.V; u++) {
        const float* l = row + (size_t)u * K;
        float* p = prob + (size_t)u * K;
        float mx = l[0];
        for (int k = 1; k < K; k++) mx = l[k] > mx ? l[k] : mx;
        float sum = 0.f;
        for (int k = 0; k < K; k++) { p[k] = expf(l[k] - mx); sum += p[k]; }
        const float inv = 1.0f / sum;
        for (int k = 0; k < K; k++) p[k] *= inv;
    }
}

// Two-hot target of a value in [0, 1] over K bins with centres k / (K - 1): the bins below and above it.
static inline void two_hot(float v, int K, int& lo, float& wlo, float& whi) {
    const float pos = clampf_(v, 0.f, 1.f) * (K - 1);
    lo = (int)pos;
    if (lo > K - 2) lo = K - 2;
    whi = pos - lo;
    wlo = 1.0f - whi;
}

static void seq_forward(const Model* M, Workspace* W, const SeqView& sv, bool want_pred) {
    seq_fwd_setup(M, W, sv);
    if (M->D.uni) { seq_fwd_uni(M, W, sv); return; }
    if (M->D.patch) { seq_fwd_patch(M, W, sv, want_pred); return; }
    for (int t = 0; t < sv.T; t++) seq_fwd_encode(M, W, W, sv, t);
    seq_fwd_core(M, W, sv, want_pred);
    if (want_pred)
        for (int t = 0; t < sv.T; t++) seq_fwd_decode(M, W, W, sv, t);
}

struct LossAcc {
    double pg = 0, v = 0, ent = 0, kl = 0, clip = 0, n = 0, clickp = 0;
    double pred_se = 0, pred_n = 0, pred_ch_se = 0, copy_se = 0, copy_ch_se = 0, ch_n = 0, drive_se = 0;
    double inv_se = 0, inv_n = 0, inv_var = 0;
    double dyn_se = 0, dyn_n = 0, dynh_se = 0, dynh_n = 0, dynx_se = 0, dynx_n = 0;
    double move = 0;
    double act_dse = 0, act_fse = 0;   // act token: next-drives and coarse-next-frame squared errors (per pred row)
    double uce[3] = {0, 0, 0}, use_[3] = {0, 0, 0}, un[3] = {0, 0, 0};   // universal: by predicted kind (body, patch, move)
    double total = 0;
    void add(const LossAcc& o) {
        pg += o.pg; v += o.v; ent += o.ent; kl += o.kl; clip += o.clip; n += o.n; clickp += o.clickp;
        pred_se += o.pred_se; pred_n += o.pred_n; pred_ch_se += o.pred_ch_se; copy_se += o.copy_se;
        copy_ch_se += o.copy_ch_se; ch_n += o.ch_n; drive_se += o.drive_se; total += o.total;
        act_dse += o.act_dse; act_fse += o.act_fse;
        for (int k = 0; k < 3; k++) { uce[k] += o.uce[k]; use_[k] += o.use_[k]; un[k] += o.un[k]; }
        inv_se += o.inv_se; inv_n += o.inv_n; inv_var += o.inv_var;
        dyn_se += o.dyn_se; dyn_n += o.dyn_n; dynh_se += o.dynh_se; dynh_n += o.dynh_n;
        dynx_se += o.dynx_se; dynx_n += o.dynx_n;
        move += o.move;
    }
};

// The drives-loss weight of a step: 1 + drive_changed_weight if hunger changes by more than drive_change_eps over it.
static inline float drive_weight(const PPOConfig& pc, const float* drv, const float* next_drv) {
    return (pc.drive_changed_weight > 0.f && fabsf(next_drv[0] - drv[0]) > pc.drive_change_eps)
               ? 1.0f + pc.drive_changed_weight : 1.0f;
}

// Computes per-token losses for a forwarded sequence; fills W->dpi, W->dval, W->dpout, W->dpdrv and dls[2].
static void seq_loss(const Model* M, Workspace* W, const SeqView& sv, const PPOConfig& pc, float inv_B,
                     float adv_mean, float adv_std, bool have_pred, float* dls, LossAcc& acc) {
    const Dims& D = M->D;
    const float* logstd = M->params.data() + M->off[LOGSTD];
    const int T = sv.T;
    for (int t = 0; t < T; t++) {
        float* dpi = W->dpi.data() + (size_t)t * 3;
        dpi[0] = dpi[1] = dpi[2] = 0.f;
        W->dval[t] = 0.f;
        if (pc.use_ppo) {
            PolicyEval e;
            policy_eval(W->pi.data() + (size_t)t * 3, logstd, sv.action + (size_t)t * sv.act_st, e);
            float A = sv.adv[(size_t)t * sv.sc_st];
            if (pc.norm_adv) A = (A - adv_mean) / (adv_std + 1e-8f);
            float lr = clampf_(e.logp - sv.old_logp[(size_t)t * sv.sc_st], -20.f, 20.f);
            float ratio = expf(lr);
            float s1 = ratio * A, s2 = clampf_(ratio, 1.f - pc.clip_eps, 1.f + pc.clip_eps) * A;
            float pg = -(s1 < s2 ? s1 : s2);
            float g = (s1 <= s2) ? -A * ratio : 0.f;  // d pg / d logp
            for (int k = 0; k < 3; k++) dpi[k] = inv_B * (g * e.dlogp_dpi[k] - pc.ent_coef * e.dent_dpi[k]);
            for (int k = 0; k < 2; k++) dls[k] += inv_B * (g * e.dlogp_dls[k] - pc.ent_coef * e.dent_dls[k]);
            float V = W->val[t], R = sv.ret[(size_t)t * sv.sc_st];
            float vloss, dV;
            if (pc.clip_vloss) {
                float Vo = sv.old_value[(size_t)t * sv.sc_st];
                float Vc = Vo + clampf_(V - Vo, -pc.vclip_eps, pc.vclip_eps);
                float lu = (V - R) * (V - R), lc = (Vc - R) * (Vc - R);
                if (lu >= lc) { vloss = 0.5f * lu; dV = V - R; }
                else { vloss = 0.5f * lc; dV = (fabsf(V - Vo) < pc.vclip_eps) ? (Vc - R) : 0.f; }
            } else {
                vloss = 0.5f * (V - R) * (V - R);
                dV = V - R;
            }
            W->dval[t] = inv_B * pc.vf_coef * dV;
            acc.pg += pg; acc.v += vloss; acc.ent += e.ent; acc.clickp += e.p;
            acc.kl += (ratio - 1.f) - lr;
            acc.clip += fabsf(ratio - 1.f) > pc.clip_eps ? 1 : 0;
            acc.total += pg + pc.vf_coef * vloss - pc.ent_coef * e.ent;
        }
        if (pc.move_coef > 0.f) {   // next-move prediction: -log pi(a_t | context) of the move actually made
            PolicyEval e;
            policy_eval(W->pi.data() + (size_t)t * 3, logstd, sv.action + (size_t)t * sv.act_st, e);
            for (int k = 0; k < 3; k++) dpi[k] -= inv_B * pc.move_coef * e.dlogp_dpi[k];
            for (int k = 0; k < 2; k++) dls[k] -= inv_B * pc.move_coef * e.dlogp_dls[k];
            acc.move -= e.logp;
            acc.total -= pc.move_coef * e.logp;
            if (!pc.use_ppo) acc.clickp += e.p;
        }
        acc.n += 1;
        if (!have_pred) continue;
        const uint8_t* ns = sv.next_small + (size_t)t * sv.ns_st;
        float* pout = W->pout.data() + (size_t)t * D.small_px;
        float* dpout = W->dpout.data() + (size_t)t * D.small_px;
        downsample_u8(sv.img + (size_t)t * sv.img_st, D.img, D.ds, W->cur_small.data());
        const uint8_t* cs = W->cur_small.data();
        const int pp = D.pimg * D.pimg;
        const float gscale = inv_B * pc.pred_coef * 2.0f / D.small_px;
        const float wch = 1.0f + pc.pred_changed_weight;  // loss weight of pixels that change between frames
        double se = 0, se_w = 0, ch_se = 0, cp_se = 0, cp_ch = 0;
        int nch = 0;
        for (int y = 0; y < D.pimg; y++)
            for (int x = 0; x < D.pimg; x++) {
                const int hw = (y * D.pimg + x) * 3;
                bool changed = ns[hw] != cs[hw] || ns[hw + 1] != cs[hw + 1] || ns[hw + 2] != cs[hw + 2];
                for (int c = 0; c < 3; c++) {
                    const int ci = c * pp + y * D.pimg + x;
                    float p = pout[ci], g = ns[hw + c] / 255.0f, e = p - g;
                    float ec = cs[hw + c] / 255.0f - g;
                    const float wgt = changed ? wch : 1.0f;
                    se += (double)e * e;
                    se_w += (double)wgt * e * e;
                    cp_se += (double)ec * ec;
                    if (changed) { ch_se += (double)e * e; cp_ch += (double)ec * ec; }
                    dpout[ci] = D.residual ? gscale * wgt * e : gscale * wgt * e * p * (1.0f - p);
                }
                nch += changed ? 1 : 0;
            }
        acc.pred_se += se / D.small_px;
        acc.copy_se += cp_se / D.small_px;
        acc.pred_ch_se += ch_se;
        acc.copy_ch_se += cp_ch;
        acc.ch_n += 3.0 * nch;
        acc.pred_n += 1;
        acc.total += pc.pred_coef * se_w / D.small_px;
        const float* nd = sv.next_drives + (size_t)t * sv.nd_st;
        const float* pdv = W->pdrv.data() + (size_t)t * 2;
        float* dpd = W->dpdrv.data() + (size_t)t * 2;
        const float wd = drive_weight(pc, sv.drives + (size_t)t * sv.drv_st, nd);   // unusual hunger change: 1 + w
        double dse = 0;
        for (int j = 0; j < 2; j++) {
            float p = pdv[j], e = p - nd[j];
            dse += (double)e * e;
            dpd[j] = wd * inv_B * pc.drive_coef * e * p * (1.0f - p);  // d/dp of coef * mean_j e^2 = coef * e (2/2)
        }
        acc.drive_se += dse / 2.0;
        acc.total += pc.drive_coef * wd * dse / 2.0;
        if (!D.act) continue;
        // the act token's auxiliary losses: next drives (as the drives head's), coarse next frame (mean squared error)
        const float* ad = W->ax_drv.data() + (size_t)t * 2;
        float* dad = W->dax_drv.data() + (size_t)t * 2;
        double ase = 0;
        for (int j = 0; j < 2; j++) {
            float p = ad[j], e = p - nd[j];
            ase += (double)e * e;
            dad[j] = wd * inv_B * pc.act_coef * e * p * (1.0f - p);
        }
        uint8_t tgt[ACT_CG * ACT_CG * 3];
        downsample_u8(ns, D.pimg, D.cds, tgt);
        const float* af = W->ax_frm.data() + (size_t)t * D.cfx;
        float* daf = W->dax_frm.data() + (size_t)t * D.cfx;
        double fse = 0;
        for (int i = 0; i < D.cfx; i++) {
            const float e = af[i] - tgt[i] / 255.0f;
            fse += (double)e * e;
            daf[i] = inv_B * pc.act_coef * 2.0f * e / D.cfx;
        }
        acc.act_dse += ase / 2.0;
        acc.act_fse += fse / D.cfx;
        acc.total += pc.act_coef * (wd * ase / 2.0 + fse / D.cfx);
    }
}

// Inverse dynamics on consecutive visual features within an episode; fills W->div_out (zero where invalid).
static void seq_inverse(const Model* M, Workspace* W, const SeqView& sv, const PPOConfig& pc, float inv_B,
                        LossAcc& acc) {
    const Dims& D = M->D;
    const int T = sv.T, F = D.feat;
    const float* P = M->params.data();
    const float* WT = M->wt.data();
    for (int t = 0; t < T; t++) {
        float* in = W->iv_in.data() + (size_t)t * 2 * F;
        const int t1 = t + 1 < T ? t + 1 : t;
        memcpy(in, W->vf.data() + (size_t)t * F, sizeof(float) * F);
        memcpy(in + F, W->vf.data() + (size_t)t1 * F, sizeof(float) * F);
    }
    linear_fwd(W->iv_h.data(), W->iv_in.data(), WT + M->off[IV1W], P + M->off[IV1B], T, 2 * F, D.ih);
    for (int i = 0; i < T * D.ih; i++) W->iv_h[i] = W->iv_h[i] > 0.f ? W->iv_h[i] : 0.f;
    linear_fwd(W->iv_out.data(), W->iv_h.data(), WT + M->off[IV2W], P + M->off[IV2B], T, D.ih, 2);
    for (int t = 0; t < T; t++) {
        float* g = W->div_out.data() + (size_t)t * 2;
        g[0] = g[1] = 0.f;
        const bool valid = t + 1 < T && !(sv.ep_start && sv.ep_start[(size_t)(t + 1) * sv.es_st]);
        if (!valid) continue;
        const float* a = sv.pred_action ? sv.pred_action + (size_t)t * sv.pact_st : sv.action + (size_t)t * sv.act_st;
        double se = 0;
        for (int k = 0; k < 2; k++) {
            const float tgt = clampf_(a[k], -1.f, 1.f);
            const float e = W->iv_out[(size_t)t * 2 + k] - tgt;
            se += 0.5 * (double)e * e;
            g[k] = inv_B * pc.inv_coef * e;
            acc.inv_var += 0.5 * (double)tgt * tgt;
        }
        acc.inv_se += se;
        acc.inv_n += 1;
        acc.total += pc.inv_coef * se;
    }
}

// Executed action, as the world applies it: clipped dx, dy and click in {0, 1}.
static inline void executed3(const float* a, float* e) {
    e[0] = clampf_(a[0], -1.f, 1.f);
    e[1] = clampf_(a[1], -1.f, 1.f);
    e[2] = a[2] > 0.5f ? 1.f : 0.f;
}

// One batched step of the latent dynamics for n rows: hunger and transition from inputs uT = [s, a] and
// uH = [s, h, a] (s = uT[:feat]). Writes hT, hH (post-ReLU), h (post-sigmoid) and s_out = s + T2 hT.
static void dyn_step(const Model* M, int n, const float* uT, const float* uH, float* hT, float* hH, float* h,
                     float* s_out) {
    const Dims& D = M->D;
    const int F = D.feat, Hd = D.dh;
    const float* P = M->params.data();
    const float* WT = M->wt.data();
    linear_fwd(hH, uH, WT + M->off[H1W], P + M->off[H1B], n, F + 4, Hd);
    for (int i = 0; i < n * Hd; i++) hH[i] = hH[i] > 0.f ? hH[i] : 0.f;
    linear_fwd(h, hH, WT + M->off[H2W], P + M->off[H2B], n, Hd, 1);
    for (int i = 0; i < n; i++) h[i] = sigmoidf_(h[i]);
    linear_fwd(hT, uT, WT + M->off[T1W], P + M->off[T1B], n, F + 3, Hd);
    for (int i = 0; i < n * Hd; i++) hT[i] = hT[i] > 0.f ? hT[i] : 0.f;
    linear_fwd(s_out, hT, WT + M->off[T2W], P + M->off[T2B], n, Hd, F);
    for (int r = 0; r < n; r++)
        for (int i = 0; i < F; i++) s_out[(size_t)r * F + i] += uT[(size_t)r * (F + 3) + i];
}

// Coarse decoder for n latents: hX (post-ReLU) and x = X2 relu(X1 s) [n, rpx] (HWC order).
static void dyn_decode(const Model* M, int n, const float* s, float* hX, float* x) {
    const Dims& D = M->D;
    const float* P = M->params.data();
    const float* WT = M->wt.data();
    linear_fwd(hX, s, WT + M->off[X1W], P + M->off[X1B], n, D.feat, D.dh);
    for (int i = 0; i < n * D.dh; i++) hX[i] = hX[i] > 0.f ? hX[i] : 0.f;
    linear_fwd(x, hX, WT + M->off[X2W], P + M->off[X2B], n, D.dh, D.rpx);
}

// Latent dynamics losses (dyn_hidden > 0). Chains of imagined steps start at every token's visual feature v_t.
// Step k of the chain from t ("pair" (t, k)) uses the action of token j = t + k - 1; chains stop at the horizon,
// at the end of the sequence and after a terminal step. Losses, scaled by inv_B like the other per-token losses:
//   latent  dyn_coef   * mean_i (s_hat - v_{j+1})^2    when token j+1 continues the episode (no gradient to v_{j+1})
//   hunger  dyn_h_coef * (h_hat - next_drives[j][0])^2 input [s_hat_{k-1}, true hunger at j, action j]
//   decoder dyn_x_coef * mean (x(s_hat) - next_small[j] / 2 / 255)^2, and x(v_t) vs token t's own frame
// The heads are trained on top of the encoder and never change it (v is detached everywhere). With gradients into
// the encoder the latent loss collapsed the features, and the decoder on real latents alone still shrank them
// (exp vm-00002: 32 and 61 of 128 dims active, inverse-dynamics R^2 stuck at 0). Accumulates the head gradients
// into G.
static void seq_dyn(const Model* M, Workspace* W, const SeqView& sv, const PPOConfig& pc, float inv_B, float* G,
                    LossAcc& acc) {
    const Dims& D = M->D;
    const int T = sv.T, F = D.feat, Hd = D.dh, R = D.rpx, UT = F + 3, UH = F + 4;
    const float* P = M->params.data();
    float* scr = W->scr.data();
    const int Hmax = std::min(pc.dyn_horizon, T);
    // ---- chain layout: level k holds the pairs (t, k) at indices [y_lev[k-1], y_lev[k])
    int np = 0, nl = 0;
    W->y_lev[0] = 0;
    for (int k = 1; k <= Hmax; k++) {
        const int begin = np;
        if (k == 1) {
            for (int t = 0; t < T; t++) { W->y_t[np] = t; W->y_j[np] = t; W->y_par[np] = -1; np++; }
        } else {
            for (int q = W->y_lev[k - 2]; q < W->y_lev[k - 1]; q++) {
                const int t = W->y_t[q], j = t + k - 1;
                if (j > T - 1) continue;
                if (sv.ep_start && sv.ep_start[(size_t)j * sv.es_st]) continue;   // step j-1 ended the episode
                W->y_t[np] = t; W->y_j[np] = j; W->y_par[np] = q; np++;
            }
        }
        if (np == begin) break;
        W->y_lev[k] = np;
        nl = k;
    }
    // ---- forward, level by level (each level depends on the previous one)
    for (int k = 1; k <= nl; k++) {
        const int p0 = W->y_lev[k - 1], n = W->y_lev[k] - p0;
        for (int p = p0; p < p0 + n; p++) {
            const int t = W->y_t[p], j = W->y_j[p];
            const float* sp = W->y_par[p] < 0 ? W->vf.data() + (size_t)t * F : W->y_s.data() + (size_t)W->y_par[p] * F;
            float e[3];
            executed3(sv.action + (size_t)j * sv.act_st, e);
            float* ut = W->y_uT.data() + (size_t)p * UT;
            float* uh = W->y_uH.data() + (size_t)p * UH;
            memcpy(ut, sp, sizeof(float) * F);
            memcpy(uh, sp, sizeof(float) * F);
            ut[F] = e[0]; ut[F + 1] = e[1]; ut[F + 2] = e[2];
            uh[F] = sv.drives[(size_t)j * sv.drv_st]; uh[F + 1] = e[0]; uh[F + 2] = e[1]; uh[F + 3] = e[2];
        }
        dyn_step(M, n, W->y_uT.data() + (size_t)p0 * UT, W->y_uH.data() + (size_t)p0 * UH,
                 W->y_hT.data() + (size_t)p0 * Hd, W->y_hH.data() + (size_t)p0 * Hd, W->y_h.data() + p0,
                 W->y_s.data() + (size_t)p0 * F);
    }
    dyn_decode(M, np, W->y_s.data(), W->y_hX.data(), W->y_x.data());
    dyn_decode(M, T, W->vf.data(), W->y_hX0.data(), W->y_x0.data());
    // ---- losses and output gradients
    const float cs = inv_B * pc.dyn_coef * 2.0f / F, ch = inv_B * pc.dyn_h_coef * 2.0f, cx = inv_B * pc.dyn_x_coef * 2.0f / R;
    std::fill(W->g_s.begin(), W->g_s.begin() + (size_t)np * F, 0.f);
    for (int p = 0; p < np; p++) {
        const int j = W->y_j[p];
        const float* sh = W->y_s.data() + (size_t)p * F;
        if (j + 1 <= T - 1 && !(sv.ep_start && sv.ep_start[(size_t)(j + 1) * sv.es_st])) {
            const float* tg = W->vf.data() + (size_t)(j + 1) * F;
            float* gs = W->g_s.data() + (size_t)p * F;
            double se = 0;
            for (int i = 0; i < F; i++) { const float e = sh[i] - tg[i]; se += (double)e * e; gs[i] = cs * e; }
            acc.dyn_se += se / F; acc.dyn_n += 1; acc.total += pc.dyn_coef * se / F;
        }
        W->g_h[p] = 0.f;
        if (pc.dyn_h_horizon <= 0 || j - W->y_t[p] + 1 <= pc.dyn_h_horizon) {   // chain step k = j - t + 1
            const float hh = W->y_h[p], eh = hh - sv.next_drives[(size_t)j * sv.nd_st];
            const float wj = drive_weight(pc, sv.drives + (size_t)j * sv.drv_st, sv.next_drives + (size_t)j * sv.nd_st);
            W->g_h[p] = wj * ch * eh * hh * (1.0f - hh);
            acc.dynh_se += (double)eh * eh; acc.dynh_n += 1; acc.total += pc.dyn_h_coef * wj * (double)eh * eh;
        }
        downsample_u8(sv.next_small + (size_t)j * sv.ns_st, D.pimg, 2, W->y_tgt.data());
        const float* x = W->y_x.data() + (size_t)p * R;
        float* gx = W->g_x.data() + (size_t)p * R;
        double se = 0;
        for (int i = 0; i < R; i++) { const float e = x[i] - W->y_tgt[i] / 255.0f; se += (double)e * e; gx[i] = cx * e; }
        acc.dynx_se += se / R; acc.dynx_n += 1; acc.total += pc.dyn_x_coef * se / R;
    }
    for (int t = 0; t < T; t++) {
        downsample_u8(sv.img + (size_t)t * sv.img_st, D.img, D.ds, W->y_small.data());
        downsample_u8(W->y_small.data(), D.pimg, 2, W->y_tgt.data());
        const float* x = W->y_x0.data() + (size_t)t * R;
        float* gx = W->g_x0.data() + (size_t)t * R;
        double se = 0;
        for (int i = 0; i < R; i++) { const float e = x[i] - W->y_tgt[i] / 255.0f; se += (double)e * e; gx[i] = cx * e; }
        acc.dynx_se += se / R; acc.dynx_n += 1; acc.total += pc.dyn_x_coef * se / R;
    }
    // ---- backward: decoders first (no dependencies), then the chains from the last level down
    linear_bwd(W->g_x.data(), W->y_hX.data(), P + M->off[X2W], np, Hd, R, W->g_hX.data(), false, G + M->off[X2W],
               G + M->off[X2B], scr);
    for (int i = 0; i < np * Hd; i++) if (W->y_hX[i] <= 0.f) W->g_hX[i] = 0.f;
    linear_bwd(W->g_hX.data(), W->y_s.data(), P + M->off[X1W], np, F, Hd, W->g_s.data(), true, G + M->off[X1W],
               G + M->off[X1B], scr);
    linear_bwd(W->g_x0.data(), W->y_hX0.data(), P + M->off[X2W], T, Hd, R, W->g_hX0.data(), false, G + M->off[X2W],
               G + M->off[X2B], scr);
    for (int i = 0; i < T * Hd; i++) if (W->y_hX0[i] <= 0.f) W->g_hX0[i] = 0.f;
    linear_bwd(W->g_hX0.data(), W->vf.data(), P + M->off[X1W], T, F, Hd, nullptr, false, G + M->off[X1W],
               G + M->off[X1B], scr);
    for (int k = nl; k >= 1; k--) {
        const int p0 = W->y_lev[k - 1], n = W->y_lev[k] - p0;
        float* ghH = W->g_hH.data() + (size_t)p0 * Hd;
        float* ghT = W->g_hT.data() + (size_t)p0 * Hd;
        linear_bwd(W->g_h.data() + p0, W->y_hH.data() + (size_t)p0 * Hd, P + M->off[H2W], n, Hd, 1, ghH, false,
                   G + M->off[H2W], G + M->off[H2B], scr);
        for (int i = 0; i < n * Hd; i++) if (W->y_hH[(size_t)p0 * Hd + i] <= 0.f) ghH[i] = 0.f;
        linear_bwd(ghH, W->y_uH.data() + (size_t)p0 * UH, P + M->off[H1W], n, UH, Hd, W->g_uH.data() + (size_t)p0 * UH,
                   false, G + M->off[H1W], G + M->off[H1B], scr);
        linear_bwd(W->g_s.data() + (size_t)p0 * F, W->y_hT.data() + (size_t)p0 * Hd, P + M->off[T2W], n, Hd, F, ghT,
                   false, G + M->off[T2W], G + M->off[T2B], scr);
        for (int i = 0; i < n * Hd; i++) if (W->y_hT[(size_t)p0 * Hd + i] <= 0.f) ghT[i] = 0.f;
        linear_bwd(ghT, W->y_uT.data() + (size_t)p0 * UT, P + M->off[T1W], n, UT, Hd, W->g_uT.data() + (size_t)p0 * UT,
                   false, G + M->off[T1W], G + M->off[T1B], scr);
        for (int p = p0; p < p0 + n; p++) {   // gradient wrt the previous latent: residual + transition + hunger inputs
            if (W->y_par[p] < 0) continue;     // chain start: v_t is detached
            float* dst = W->g_s.data() + (size_t)W->y_par[p] * F;
            const float* gs = W->g_s.data() + (size_t)p * F;
            const float* gut = W->g_uT.data() + (size_t)p * UT;
            const float* guh = W->g_uH.data() + (size_t)p * UH;
            for (int i = 0; i < F; i++) dst[i] += gs[i] + gut[i] + guh[i];
        }
    }
}

// Decoder backward of frame t: parameter gradients of the decoder convolutions, and the gradients of its inputs
// (decoder MLP output dph2, tiled latent dub, skip maps dskip / dskip3) in frame t's slots of W.
static void seq_bwd_decode(const Model* M, Workspace* W, Workspace* X, int t, float* G) {
    const Dims& D = M->D;
    const float* P = M->params.data();
    float* scr = X->scr.data();
    const int p1n = D.pc1 * D.p1 * D.p1, q0 = D.p0 * D.p0, q1 = D.p1 * D.p1;
    float* dph2 = W->dph2.data();
    const float* pd1 = W->pd1.data() + (size_t)t * p1n;
    float* dpd1 = X->dpd1.data();
    const bool ext1 = D.coord || D.skip;
    const float* in1 = ext1 ? W->d1in.data() + (size_t)t * D.d1_in * q0 : W->ph2.data() + (size_t)t * D.pred_flat;
    const bool ext2 = D.coord || D.skip;
    const float* in2 = ext2 ? W->d2in.data() + (size_t)t * D.d2_in * q1 : pd1;
    float* dx2 = ext2 ? X->dd2in.data() : dpd1;
    deconv_bwd(in2, D.d2_in, D.p1, P + M->off[D2W], 3, W->dpout.data() + (size_t)t * D.small_px,
               G + M->off[D2W], G + M->off[D2B], dx2, scr);
    if (ext2) memcpy(dpd1, dx2, sizeof(float) * p1n);   // coordinate / action channels get no gradient
    if (D.skip)
        memcpy(W->dskip.data() + (size_t)t * D.c2 * q1, dx2 + (size_t)D.skip_off * q1, sizeof(float) * D.c2 * q1);
    for (int i = 0; i < p1n; i++) if (pd1[i] <= 0.f) dpd1[i] = 0.f;
    float* dx1 = ext1 ? X->dd1in.data() : dph2 + (size_t)t * D.pred_flat;
    deconv_bwd(in1, D.d1_in, D.p0, P + M->off[D1W], D.pc1, dpd1, G + M->off[D1W], G + M->off[D1B], dx1, scr);
    if (D.skip)
        memcpy(W->dskip3.data() + (size_t)t * D.enc_flat, dx1 + (size_t)D.skip1_off * q0, sizeof(float) * D.enc_flat);
    if (ext1) {
        memcpy(dph2 + (size_t)t * D.pred_flat, dx1, sizeof(float) * D.pred_flat);
        for (int k = 0; k < D.pb; k++) {
            float sum = 0.f;
            for (int i = 0; i < q0; i++) sum += dx1[(size_t)(D.pc2 + k) * q0 + i];
            W->dub[(size_t)t * D.pb + k] = W->ub[(size_t)t * D.pb + k] > 0.f ? sum : 0.f;
        }
    }
}

// Backward of transformer_fwd: from W->dz (the gradient at W->lnf, [T, d]) to W->dres (the gradient at W->xin);
// parameter gradients are added to G.
static void transformer_bwd(const Model* M, Workspace* W, int T, const int* ss, const int* se, float* G) {
    const Dims& D = M->D;
    const int K = D.KT, d = D.d;
    const float* P = M->params.data();
    float* dres = W->dres.data();
    float* scr = W->scr.data();
    const float* dz = W->dz.data();
    memset(dres, 0, sizeof(float) * T * d);
    const float* xlast = W->res2.data() + (size_t)(D.L - 1) * K * d;
    layernorm_bwd(dres, G + M->off[LNFW], G + M->off[LNFB], dz, xlast, P + M->off[LNFW], W->lnfm.data(),
                  W->lnfr.data(), T, d);
    for (int l = D.L - 1; l >= 0; l--) {
        const float* xl = (l == 0) ? W->xin.data() : W->res2.data() + (size_t)(l - 1) * K * d;
        const float* ln1 = W->ln1.data() + (size_t)l * K * d;
        const float* qkv = W->qkv.data() + (size_t)l * K * 3 * d;
        const float* att = W->att.data() + (size_t)l * D.nh * K * K;
        const float* atty = W->atty.data() + (size_t)l * K * d;
        const float* res1 = W->res1.data() + (size_t)l * K * d;
        const float* ln2 = W->ln2.data() + (size_t)l * K * d;
        const float* fch = W->fch.data() + (size_t)l * K * D.dff;
        const float* fchg = W->fchg.data() + (size_t)l * K * D.dff;
        linear_bwd(dres, fchg, P + M->off[FPW] + (size_t)l * d * D.dff, T, D.dff, d, W->dfchg.data(), false,
                   G + M->off[FPW] + (size_t)l * d * D.dff, G + M->off[FPB] + l * d, scr);
        gelu_bwd(W->dfch.data(), fch, W->dfchg.data(), T * D.dff);
        linear_bwd(W->dfch.data(), ln2, P + M->off[FCW] + (size_t)l * D.dff * d, T, d, D.dff, W->dln.data(), false,
                   G + M->off[FCW] + (size_t)l * D.dff * d, G + M->off[FCB] + l * D.dff, scr);
        layernorm_bwd(dres, G + M->off[LN2W] + l * d, G + M->off[LN2B] + l * d, W->dln.data(), res1,
                      P + M->off[LN2W] + l * d, W->ln2m.data() + (size_t)l * K, W->ln2r.data() + (size_t)l * K, T, d);
        linear_bwd(dres, atty, P + M->off[APW] + (size_t)l * d * d, T, d, d, W->datty.data(), false,
                   G + M->off[APW] + (size_t)l * d * d, G + M->off[APB] + l * d, scr);
        memset(W->dqkv.data(), 0, sizeof(float) * T * 3 * d);
        attention_bwd(W->dqkv.data(), W->datty.data(), qkv, att, ss, se, T, K, d, D.nh, W->arow.data());
        linear_bwd(W->dqkv.data(), ln1, P + M->off[QKVW] + (size_t)l * 3 * d * d, T, d, 3 * d, W->dln.data(), false,
                   G + M->off[QKVW] + (size_t)l * 3 * d * d, G + M->off[QKVB] + l * 3 * d, scr);
        layernorm_bwd(dres, G + M->off[LN1W] + l * d, G + M->off[LN1B] + l * d, W->dln.data(), xl,
                      P + M->off[LN1W] + l * d, W->ln1m.data() + (size_t)l * K, W->ln1r.data() + (size_t)l * K, T, d);
    }
}

// Backward from the heads (policy, value, decoder MLP from dph2 / dub of every frame) through the Transformer to
// the encoder's conv-3 maps (da3 of every frame). pi_detach: the policy head's gradient stops at z.
static void seq_bwd_core(const Model* M, Workspace* W, const SeqView& sv, float* G, bool have_pred, float pred_scale,
                         bool have_inv, bool pi_detach) {
    const Dims& D = M->D;
    const int T = sv.T, d = D.d;
    const float* P = M->params.data();
    float* scr = W->scr.data();
    float* dz = W->dz.data();
    memset(dz, 0, sizeof(float) * T * d);
    float* dz_pi = pi_detach ? nullptr : dz;
    if (D.pih) {
        float* dh = W->dpih.data();
        linear_bwd(W->dpi.data(), W->pih.data(), P + M->off[PIW], T, D.pih, 3, dh, false, G + M->off[PIW],
                   G + M->off[PIB], scr);
        for (int i = 0; i < T * D.pih; i++) if (W->pih[i] <= 0.f) dh[i] = 0.f;
        linear_bwd(dh, W->lnf.data(), P + M->off[PHW], T, d, D.pih, dz_pi, true, G + M->off[PHW], G + M->off[PHB],
                   scr);
    } else {
        linear_bwd(W->dpi.data(), W->lnf.data(), P + M->off[PIW], T, d, 3, dz_pi, true, G + M->off[PIW],
                   G + M->off[PIB], scr);
    }
    for (int t = 0; t < T; t++) {
        const float dv = W->dval[t];
        if (dv == 0.f) continue;
        const float* z = W->lnf.data() + (size_t)t * d;
        const float* vw = P + M->off[VW];
        float* gz = dz + (size_t)t * d;
        float* gw = G + M->off[VW];
        for (int i = 0; i < d; i++) { gz[i] += dv * vw[i]; gw[i] += dv * z[i]; }
        G[M->off[VB]] += dv;
    }
    if (have_pred) {
        float* dph2 = W->dph2.data();
        for (int i = 0; i < T * D.pred_flat; i++) if (W->ph2[i] <= 0.f) dph2[i] = 0.f;
        float* dph1 = W->dph1.data();
        linear_bwd(W->dpdrv.data(), W->ph1.data(), P + M->off[DRW], T, D.ph, 2, dph1, false, G + M->off[DRW],
                   G + M->off[DRB], scr);
        if (D.coord && D.pb > 0)
            linear_bwd(W->dub.data(), W->ph1.data(), P + M->off[PBW], T, D.ph, D.pb, dph1, true, G + M->off[PBW],
                       G + M->off[PBB], scr);
        linear_bwd(dph2, W->ph1.data(), P + M->off[PR2W], T, D.ph, D.pred_flat, dph1, true, G + M->off[PR2W],
                   G + M->off[PR2B], scr);
        for (int i = 0; i < T * D.ph; i++) if (W->ph1[i] <= 0.f) dph1[i] = 0.f;
        float* dpin = W->dpin.data();
        linear_bwd(dph1, W->pin.data(), P + M->off[PR1W], T, D.pin_dim, D.ph, dpin, false, G + M->off[PR1W],
                   G + M->off[PR1B], scr);
        if (pred_scale != 0.f)
            for (int t = 0; t < T; t++)
                for (int i = 0; i < d; i++) dz[(size_t)t * d + i] += pred_scale * dpin[(size_t)t * D.pin_dim + i];
    }
    transformer_bwd(M, W, T, W->ss.data(), nullptr, G);
    float* dres = W->dres.data();
    for (int t = 0; t < T; t++) {
        float* gpe = G + M->off[WPE] + (size_t)W->pos[t] * d;
        const float* dr = dres + (size_t)t * d;
        for (int i = 0; i < d; i++) gpe[i] += dr[i];
    }
    linear_bwd(dres, W->tin.data(), P + M->off[INW], T, D.in_dim, d, W->dtin.data(), false, G + M->off[INW],
               G + M->off[INB], scr);
    float* dvx = W->dvf_extra.data();
    memset(dvx, 0, sizeof(float) * T * D.feat);
    if (have_inv) {
        const int F = D.feat;
        linear_bwd(W->div_out.data(), W->iv_h.data(), P + M->off[IV2W], T, D.ih, 2, W->div_h.data(), false,
                   G + M->off[IV2W], G + M->off[IV2B], scr);
        for (int i = 0; i < T * D.ih; i++) if (W->iv_h[i] <= 0.f) W->div_h[i] = 0.f;
        linear_bwd(W->div_h.data(), W->iv_in.data(), P + M->off[IV1W], T, 2 * F, D.ih, W->div_in.data(), false,
                   G + M->off[IV1W], G + M->off[IV1B], scr);
        for (int t = 0; t < T; t++) {
            const float* gi = W->div_in.data() + (size_t)t * 2 * F;
            const int t1 = t + 1 < T ? t + 1 : t;
            for (int i = 0; i < F; i++) { dvx[(size_t)t * F + i] += gi[i]; dvx[(size_t)t1 * F + i] += gi[F + i]; }
        }
    }
    for (int t = 0; t < T; t++)
        for (int i = 0; i < D.feat; i++) {
            const size_t k = (size_t)t * D.feat + i;
            W->dvf[k] = W->vf[k] > 0.f ? W->dtin[(size_t)t * D.in_dim + i] + dvx[k] : 0.f;
        }
    linear_bwd(W->dvf.data(), W->a3.data(), P + M->off[EFW], T, D.enc_flat, D.feat, W->da3.data(), false,
               G + M->off[EFW], G + M->off[EFB], scr);
}

// Encoder backward of frame t, from da3 (plus the decoder's skip gradients) to the convolution parameters.
static void seq_bwd_encode(const Model* M, Workspace* W, Workspace* X, int t, float* G, bool have_pred,
                           float pred_scale) {
    const Dims& D = M->D;
    const float* WT = M->wt.data();
    float* scr = X->scr.data();
    float* da2 = X->da2.data();
    float* da1 = X->da1.data();
    const float* x0 = W->x0.data() + (size_t)t * D.in_ch * D.img * D.img;
    const float* a1 = W->a1.data() + (size_t)t * D.c1 * D.h1 * D.h1;
    const float* a2 = W->a2.data() + (size_t)t * D.c2 * D.h2 * D.h2;
    const float* a3 = W->a3.data() + (size_t)t * D.enc_flat;
    float* da3 = W->da3.data() + (size_t)t * D.enc_flat;
    if (have_pred && D.skip && pred_scale != 0.f) {
        const float* ds = W->dskip3.data() + (size_t)t * D.enc_flat;
        for (int i = 0; i < D.enc_flat; i++) da3[i] += pred_scale * ds[i];
    }
    for (int i = 0; i < D.enc_flat; i++) if (a3[i] <= 0.f) da3[i] = 0.f;
    conv_bwd(a2, D.c2, D.h2, WT + M->off[C3W], D.c3, da3, G + M->off[C3W], G + M->off[C3B], da2, scr);
    const int n2 = D.c2 * D.h2 * D.h2;
    if (have_pred && D.skip && pred_scale != 0.f) {
        const float* ds = W->dskip.data() + (size_t)t * n2;
        for (int i = 0; i < n2; i++) da2[i] += pred_scale * ds[i];
    }
    for (int i = 0; i < n2; i++) if (a2[i] <= 0.f) da2[i] = 0.f;
    conv_bwd(a1, D.c1, D.h1, WT + M->off[C2W], D.c2, da2, G + M->off[C2W], G + M->off[C2B], da1, scr);
    const int n1 = D.c1 * D.h1 * D.h1;
    for (int i = 0; i < n1; i++) if (a1[i] <= 0.f) da1[i] = 0.f;
    conv_bwd(x0, D.in_ch, D.img, nullptr, D.c1, da1, G + M->off[C1W], G + M->off[C1B], nullptr, scr);
}

// Backward of seq_fwd_patch (after seq_loss): heads -> token outputs -> Transformer -> token embeddings.
static void seq_bwd_patch(const Model* M, Workspace* W, const SeqView& sv, float* G, bool have_pred, float pred_scale,
                          bool pi_detach) {
    const Dims& D = M->D;
    const int T = sv.T, d = D.d, nt = D.ntok, np = D.npch;
    const float* P = M->params.data();
    float* scr = W->scr.data();
    float* dz = W->dz.data();
    memset(dz, 0, sizeof(float) * (size_t)T * nt * d);
    float* dzm = W->dzm.data();
    memset(dzm, 0, sizeof(float) * (size_t)T * d);
    float* dzm_pi = pi_detach ? nullptr : dzm;
    if (D.pih) {
        float* dh = W->dpih.data();
        linear_bwd(W->dpi.data(), W->pih.data(), P + M->off[PIW], T, D.pih, 3, dh, false, G + M->off[PIW],
                   G + M->off[PIB], scr);
        for (int i = 0; i < T * D.pih; i++) if (W->pih[i] <= 0.f) dh[i] = 0.f;
        linear_bwd(dh, W->zm.data(), P + M->off[PHW], T, d, D.pih, dzm_pi, true, G + M->off[PHW], G + M->off[PHB],
                   scr);
    } else {
        linear_bwd(W->dpi.data(), W->zm.data(), P + M->off[PIW], T, d, 3, dzm_pi, true, G + M->off[PIW],
                   G + M->off[PIB], scr);
    }
    for (int t = 0; t < T; t++) {
        const float dv = W->dval[t];
        if (dv == 0.f) continue;
        const float* z = W->zm.data() + (size_t)t * d;
        const float* vw = P + M->off[VW];
        float* gz = dzm + (size_t)t * d;
        float* gw = G + M->off[VW];
        for (int i = 0; i < d; i++) { gz[i] += dv * vw[i]; gw[i] += dv * z[i]; }
        G[M->off[VB]] += dv;
    }
    const int zi = D.act ? np + 3 : np + 1;   // the token the policy reads
    if (have_pred && D.act) {   // the act token's auxiliary heads -> its output (with the prediction scale)
        float* dh = W->dax_h.data();
        linear_bwd(W->dax_drv.data(), W->ax_h.data(), P + M->off[AXDW], T, D.ph, 2, dh, false, G + M->off[AXDW],
                   G + M->off[AXDB], scr);
        linear_bwd(W->dax_frm.data(), W->ax_h.data(), P + M->off[AXFW], T, D.ph, D.cfx, dh, true, G + M->off[AXFW],
                   G + M->off[AXFB], scr);
        for (int i = 0; i < T * D.ph; i++) if (W->ax_h[i] <= 0.f) dh[i] = 0.f;
        linear_bwd(dh, W->ax_in.data(), P + M->off[AX1W], T, D.pin_dim, D.ph, W->dax_in.data(), false,
                   G + M->off[AX1W], G + M->off[AX1B], scr);
        if (pred_scale != 0.f)
            for (int t = 0; t < T; t++)
                for (int i = 0; i < d; i++) dzm[(size_t)t * d + i] += pred_scale * W->dax_in[(size_t)t * D.pin_dim + i];
    }
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++) dz[((size_t)t * nt + zi) * d + i] += dzm[(size_t)t * d + i];
    if (have_pred) {
        const int R = T * (np + 1), pp = D.pimg * D.pimg, cs = D.cell;
        for (int t = 0; t < T; t++) {   // dpout (CHW) -> each patch's cell
            const float* dpo = W->dpout.data() + (size_t)t * D.small_px;
            for (int p = 0; p < np; p++) {
                float* dl = W->dpdel.data() + ((size_t)t * np + p) * D.cpx;
                const int oy = (p / D.gp) * cs, ox = (p % D.gp) * cs;
                for (int y = 0; y < cs; y++)
                    for (int x = 0; x < cs; x++)
                        for (int c = 0; c < 3; c++) dl[(y * cs + x) * 3 + c] = dpo[c * pp + (oy + y) * D.pimg + ox + x];
            }
        }
        float* dph1 = W->dph1.data();
        linear_bwd(W->dpdel.data(), W->ph1.data(), P + M->off[PR2W], T * np, D.ph, D.cpx, dph1, false,
                   G + M->off[PR2W], G + M->off[PR2B], scr);
        linear_bwd(W->dpdrv.data(), W->ph1.data() + (size_t)T * np * D.ph, P + M->off[DRW], T, D.ph, 2,
                   dph1 + (size_t)T * np * D.ph, false, G + M->off[DRW], G + M->off[DRB], scr);
        for (int i = 0; i < R * D.ph; i++) if (W->ph1[i] <= 0.f) dph1[i] = 0.f;
        float* dpin = W->dpin.data();
        linear_bwd(dph1, W->pin.data(), P + M->off[PR1W], R, D.pin_dim, D.ph, dpin, false, G + M->off[PR1W],
                   G + M->off[PR1B], scr);
        if (pred_scale != 0.f)
            for (int r = 0; r < R; r++) {
                const int t = r < T * np ? r / np : r - T * np, k = r < T * np ? r % np : np;
                float* g = dz + ((size_t)t * nt + k) * d;
                const float* gi = dpin + (size_t)r * D.pin_dim;
                for (int i = 0; i < d; i++) g[i] += pred_scale * gi[i];
            }
    }
    transformer_bwd(M, W, T * nt, W->tss.data(), W->tse.data(), G);
    const float* dres = W->dres.data();
    float* gtyp = G + M->off[TYP];
    float *gdw = G + M->off[DEW], *gdb = G + M->off[DEB], *gmw = G + M->off[MEW], *gmb = G + M->off[MEB];
    float *gcw = G + M->off[CEW], *gcb = G + M->off[CEB];
    for (int t = 0; t < T; t++) {
        float* gpe = G + M->off[WPE] + (size_t)W->pos[t] * d;
        for (int k = 0; k < nt; k++) {
            const float* dr = dres + ((size_t)t * nt + k) * d;
            float* gt = gtyp + (size_t)(k < np ? 0 : k - np + 1) * d;
            for (int i = 0; i < d; i++) { gpe[i] += dr[i]; gt[i] += dr[i]; }
            if (k < np) {
                float* g2 = G + M->off[POS2D] + (size_t)k * d;
                float* ge = W->dpemb.data() + ((size_t)t * np + k) * d;
                for (int i = 0; i < d; i++) { g2[i] += dr[i]; ge[i] = dr[i]; }
            }
        }
        const float* drv = sv.drives + (size_t)t * sv.drv_st;
        const float* pa = sv.prev_action + (size_t)t * sv.pa_st;
        const float cmd = M->cfg.use_prev_reward ? sv.prev_reward[(size_t)t * sv.pr_st] : 0.f;
        const float* rd = dres + ((size_t)t * nt + np) * d;
        const float* rm = rd + d;
        const float* rc = rm + d;
        for (int i = 0; i < d; i++) {
            gdw[i * 2] += rd[i] * drv[0]; gdw[i * 2 + 1] += rd[i] * drv[1]; gdb[i] += rd[i];
            gmw[i * 3] += rm[i] * pa[0]; gmw[i * 3 + 1] += rm[i] * pa[1]; gmw[i * 3 + 2] += rm[i] * pa[2];
            gmb[i] += rm[i];
            gcw[i] += rc[i] * cmd; gcb[i] += rc[i];
        }
        if (D.act) {
            const float* ra = rc + d;
            float* gae = G + M->off[ACTE];
            for (int i = 0; i < d; i++) gae[i] += ra[i];
        }
    }
    linear_bwd(W->dpemb.data(), W->tpx.data(), P + M->off[PEW], T * np, D.pdim, d, nullptr, false, G + M->off[PEW],
               G + M->off[PEB], scr);
}

// The loss of every token (cross-entropy of the next token's used values, their mean, times inv_tok), fused with the
// head's backward: the head's gradients into G and each token's output gradient into W->dz.
static void seq_loss_uni(const Model* M, Workspace* W, const SeqView& sv, float inv_tok, float* G, LossAcc& acc) {
    const Dims& D = M->D;
    const int T = sv.T, nt = D.ntok, V = D.V, d = D.d, K = D.nbin;
    const float* P = M->params.data();
    float* row = W->u_row.data();
    float* prob = W->u_prob.data();
    float* gW = G + M->off[UOW];
    float* gb = G + M->off[UOB];
    const float* Wo = P + M->off[UOW];
    for (int i = 0; i < T * nt; i++) {
        const int k = i % nt, u = W->u_use[i];
        const int kind = k == nt - 1 ? 0 : (k == nt - 2 ? 2 : 1);   // predicted: the next body, the move, a patch
        if (u == 0) {   // the next token is a summary (task 00047): no target
            for (int j = 0; j < d; j++) W->dz[(size_t)i * d + j] = 0.f;
            continue;
        }
        const float* z = W->lnf.data() + (size_t)i * d;
        const float* tg = W->u_tgt.data() + (size_t)i * V;
        uni_head_row(M, z, row, prob);
        float* dz = W->dz.data() + (size_t)i * d;
        for (int j = 0; j < d; j++) dz[j] = 0.f;
        const float sc = inv_tok / u;
        double ce = 0, se = 0;
        for (int v = 0; v < u; v++) {
            int lo; float wlo, whi;
            two_hot(tg[v], K, lo, wlo, whi);
            float* p = prob + (size_t)v * K;
            float ev = 0.f;
            for (int b = 0; b < K; b++) ev += p[b] * (float)b / (K - 1);
            ce -= wlo * logf(std::max(p[lo], 1e-30f)) + whi * logf(std::max(p[lo + 1], 1e-30f));
            se += (double)(ev - tg[v]) * (ev - tg[v]);
            p[lo] -= wlo;           // dL/dlogits = (p - target) * sc, in place
            p[lo + 1] -= whi;
            for (int b = 0; b < K; b++) {
                const float g = p[b] * sc;
                const int r = v * K + b;
                gb[r] += g;
                const float* wr = Wo + (size_t)r * d;
                float* gw = gW + (size_t)r * d;
                for (int j = 0; j < d; j++) { gw[j] += g * z[j]; dz[j] += g * wr[j]; }
            }
        }
        ce /= u;
        acc.uce[kind] += ce; acc.use_[kind] += se / u; acc.un[kind] += 1;
        acc.total += ce;
        acc.n += 1;
    }
}

// Backward from W->dz: the Transformer, then the input map, the slot and the step embeddings.
static void seq_bwd_uni(const Model* M, Workspace* W, const SeqView& sv, float* G) {
    const Dims& D = M->D;
    const int T = sv.T, nt = D.ntok, d = D.d;
    transformer_bwd(M, W, T * nt, W->tss.data(), nullptr, G);
    const float* dres = W->dres.data();
    linear_bwd(dres, W->u_in.data(), M->params.data() + M->off[UIW], T * nt, D.VI, d, nullptr, false, G + M->off[UIW],
               G + M->off[UIB], W->scr.data());
    for (int t = 0; t < T; t++) {
        float* gpe = G + M->off[WPE] + (size_t)W->pos[t] * d;
        for (int k = 0; k < nt; k++) {
            const float* dr = dres + ((size_t)t * nt + k) * d;
            float* gs = G + M->off[USL] + (size_t)k * d;
            for (int j = 0; j < d; j++) { gpe[j] += dr[j]; gs[j] += dr[j]; }
        }
    }
}

static void seq_backward(const Model* M, Workspace* W, const SeqView& sv, float* G, bool have_pred, float pred_scale,
                         bool have_inv, bool pi_detach) {
    if (M->D.patch) { seq_bwd_patch(M, W, sv, G, have_pred, pred_scale, pi_detach); return; }
    if (have_pred)
        for (int t = 0; t < sv.T; t++) seq_bwd_decode(M, W, W, t, G);
    seq_bwd_core(M, W, sv, G, have_pred, pred_scale, have_inv, pi_detach);
    for (int t = 0; t < sv.T; t++) seq_bwd_encode(M, W, W, t, G, have_pred, pred_scale);
}

// =====================================================================================
// Acting: one token per env with a KV cache
// =====================================================================================
struct ActCtx {
    Model* M;
    int N;
    long step = 0;              // committed act() calls since creation/reset; context block = step / ctx
    std::vector<int> len;
    std::vector<float> kc, vc;  // [L][N][K][d]
    std::vector<Rng> rng;
};

static void act_one(const Model* M, Workspace* W, ActCtx* C, int n, const uint8_t* img, const float* drv,
                    const float* pa, float pr, int deterministic, float* action, float* logp, float* value, float* mu,
                    float* click_p) {
    const Dims& D = M->D;
    const int K = D.K, d = D.d, N = C->N;
    const float* P = M->params.data();
    const float* WT = M->wt.data();
    const int pos = C->len[n];
    CK(pos < K);
    encode_frame(M, W, W->scr.data(), img, 0);
    linear_fwd(W->vf.data(), W->a3.data(), WT + M->off[EFW], P + M->off[EFB], 1, D.enc_flat, D.feat);
    for (int i = 0; i < D.feat; i++) W->vf[i] = W->vf[i] > 0.f ? W->vf[i] : 0.f;
    build_token_input(M, W, 0, drv, pa, pr);
    linear_fwd(W->xin.data(), W->tin.data(), WT + M->off[INW], P + M->off[INB], 1, D.in_dim, d);
    {
        const float* pe = P + M->off[WPE] + (size_t)pos * d;
        for (int i = 0; i < d; i++) W->xin[i] += pe[i];
    }
    const float scale = 1.0f / sqrtf((float)D.hs);
    float* x = W->xin.data();
    float* ln = W->ln1.data();
    float* qkv = W->qkv.data();
    float* atty = W->atty.data();
    float* proj = W->proj.data();
    float* r1 = W->res1.data();
    float* r2 = W->res2.data();
    float* fch = W->fch.data();
    float* fchg = W->fchg.data();
    float mean, rstd;
    for (int l = 0; l < D.L; l++) {
        layernorm_fwd(ln, &mean, &rstd, x, P + M->off[LN1W] + l * d, P + M->off[LN1B] + l * d, 1, d);
        linear_fwd(qkv, ln, WT + M->off[QKVW] + (size_t)l * 3 * d * d, P + M->off[QKVB] + l * 3 * d, 1, d, 3 * d);
        float* kbase = C->kc.data() + (((size_t)l * N + n) * K) * d;
        float* vbase = C->vc.data() + (((size_t)l * N + n) * K) * d;
        memcpy(kbase + (size_t)pos * d, qkv + d, sizeof(float) * d);
        memcpy(vbase + (size_t)pos * d, qkv + 2 * d, sizeof(float) * d);
        for (int h = 0; h < D.nh; h++)
            attend_one(qkv + h * D.hs, kbase + h * D.hs, d, vbase + h * D.hs, d, pos + 1, D.hs, scale, W->arow.data(),
                       atty + h * D.hs);
        linear_fwd(proj, atty, WT + M->off[APW] + (size_t)l * d * d, P + M->off[APB] + l * d, 1, d, d);
        for (int i = 0; i < d; i++) r1[i] = x[i] + proj[i];
        layernorm_fwd(ln, &mean, &rstd, r1, P + M->off[LN2W] + l * d, P + M->off[LN2B] + l * d, 1, d);
        linear_fwd(fch, ln, WT + M->off[FCW] + (size_t)l * D.dff * d, P + M->off[FCB] + l * D.dff, 1, d, D.dff);
        gelu_fwd(fchg, fch, D.dff);
        linear_fwd(proj, fchg, WT + M->off[FPW] + (size_t)l * d * D.dff, P + M->off[FPB] + l * d, 1, D.dff, d);
        for (int i = 0; i < d; i++) r2[i] = r1[i] + proj[i];
        memcpy(x, r2, sizeof(float) * d);
    }
    float* z = W->lnf.data();
    layernorm_fwd(z, &mean, &rstd, x, P + M->off[LNFW], P + M->off[LNFB], 1, d);
    float pi3[3];
    policy_fwd(M, z, W->pih.data(), pi3, 1);
    float V = value_head(z, P + M->off[VW], P[M->off[VB]], d);
    const float* logstd = P + M->off[LOGSTD];
    Rng& rng = C->rng[n];
    float a[3];
    for (int i = 0; i < 2; i++) {
        float m = tanhf(pi3[i]), sd = expf(clampf_(logstd[i], LS_MIN, LS_MAX));
        a[i] = deterministic ? m : m + sd * rng_normal(rng);
        if (mu) mu[i] = m;
    }
    float p = sigmoidf_(pi3[2]);
    a[2] = deterministic ? (p > 0.5f ? 1.f : 0.f) : (rng_uniform(rng) < p ? 1.f : 0.f);
    PolicyEval e;
    policy_eval(pi3, logstd, a, e);
    action[0] = a[0]; action[1] = a[1]; action[2] = a[2];
    *logp = e.logp;
    *value = V;
    if (click_p) *click_p = p;
    C->len[n] = pos + 1;
}

// =====================================================================================
// Rollout memory, GAE, PPO update
// =====================================================================================
struct Rollout {
    Model* M;
    int T, N;
    std::vector<uint8_t> images, next_small, ep_start, done;
    std::vector<float> drives, prev_action, prev_reward, action, logp, value, reward, next_drives, adv, ret;
};

static SeqView rollout_view(const Rollout* R, int s) {
    const Dims& D = R->M->D;
    const int N = R->N, K = D.K, n = s % N, b = s / N;
    const size_t base = (size_t)b * K * N + n;  // index of (t = b*K, env n)
    SeqView v;
    v.T = K;
    v.img = R->images.data() + base * D.img_px; v.img_st = (long)N * D.img_px;
    v.drives = R->drives.data() + base * 2; v.drv_st = 2L * N;
    v.prev_action = R->prev_action.data() + base * 3; v.pa_st = 3L * N;
    v.prev_reward = R->prev_reward.data() + base; v.pr_st = N;
    v.ep_start = R->ep_start.data() + base; v.es_st = N;
    v.action = R->action.data() + base * 3; v.act_st = 3L * N;
    v.old_logp = R->logp.data() + base; v.old_value = R->value.data() + base;
    v.adv = R->adv.data() + base; v.ret = R->ret.data() + base; v.sc_st = N;
    v.next_small = R->next_small.data() + base * D.small_px; v.ns_st = (long)N * D.small_px;
    v.next_drives = R->next_drives.data() + base * 2; v.nd_st = 2L * N;
    return v;
}

static void compute_gae(Rollout* R, const float* last_value, const PPOConfig& pc) {
    const int T = R->T, N = R->N;
    for (int n = 0; n < N; n++) {
        float gae = 0.f;
        for (int t = T - 1; t >= 0; t--) {
            const size_t i = (size_t)t * N + n;
            float nv = (t == T - 1) ? last_value[n] : R->value[i + N];
            float nonterm = R->done[i] ? 0.f : 1.f;
            float delta = R->reward[i] * pc.reward_scale + pc.gamma * nv * nonterm - R->value[i];
            gae = delta + pc.gamma * pc.lam * nonterm * gae;
            R->adv[i] = gae;
            R->ret[i] = gae + R->value[i];
        }
    }
}

// Loss + gradient (into M->grads) over the given sequences. Returns accumulated loss terms.
// Advantage normalization statistics over the minibatch (PPO only).
static void adv_stats(Rollout* R, const int* ids, int n_seq, const PPOConfig& pc, double& am, double& as) {
    am = 0; as = 0;
    if (!(pc.use_ppo && pc.norm_adv)) return;
    double s = 0, s2 = 0;
    long cnt = 0;
    for (int q = 0; q < n_seq; q++) {
        SeqView sv = rollout_view(R, ids[q]);
        for (int t = 0; t < sv.T; t++) { double a = sv.adv[(size_t)t * sv.sc_st]; s += a; s2 += a * a; cnt++; }
    }
    am = s / cnt;
    double var = cnt > 1 ? (s2 - cnt * am * am) / (cnt - 1) : 0.0;
    as = sqrt(var > 0 ? var : 0);
}

// One thread per sequence (minibatches with at least as many sequences as threads).
static LossAcc loss_and_grads_seqs(Model* M, Rollout* R, const int* ids, int n_seq, const PPOConfig& pc) {
    const Dims& D = M->D;
    const bool have_pred = pc.pred_coef > 0.f || pc.drive_coef > 0.f;
    const bool have_inv = pc.inv_coef > 0.f && D.ih > 0;
    const bool have_dyn = D.dh > 0 && pc.dyn_horizon > 0 && (pc.dyn_coef > 0.f || pc.dyn_h_coef > 0.f || pc.dyn_x_coef > 0.f);
    const float inv_B = 1.0f / (float)((long)n_seq * D.K);
    double am, as;
    adv_stats(R, ids, n_seq, pc, am, as);
    const int nt = omp_get_max_threads();
    std::vector<LossAcc> accs(nt);
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        bool used = false;
        #pragma omp for schedule(static, 1)
        for (int q = 0; q < n_seq; q++) {
            float* G = M->tg[tid].data();
            if (!used) {
                if (M->tdirty[tid]) memset(G, 0, sizeof(float) * M->n_params);
                used = true;
            }
            Workspace* W = M->ws[tid];
            SeqView sv = rollout_view(R, ids[q]);
            if (D.uni) {   // the universal model: one loss over every token, nothing else
                seq_forward(M, W, sv, false);
                seq_loss_uni(M, W, sv, inv_B / (D.ntok - D.nnot), G, accs[tid]);   // the mean over targets
                seq_bwd_uni(M, W, sv, G);
                continue;
            }
            seq_forward(M, W, sv, have_pred);
            float dls[2] = {0.f, 0.f};
            seq_loss(M, W, sv, pc, inv_B, (float)am, (float)as, have_pred, dls, accs[tid]);
            if (have_inv) seq_inverse(M, W, sv, pc, inv_B, accs[tid]);
            if (have_dyn) seq_dyn(M, W, sv, pc, inv_B, G, accs[tid]);
            G[M->off[LOGSTD] + 0] += dls[0];
            G[M->off[LOGSTD] + 1] += dls[1];
            seq_backward(M, W, sv, G, have_pred, pc.pred_backbone_scale, have_inv, pc.pi_detach != 0);
        }
        M->tused[tid] = used;
        if (used) M->tdirty[tid] = 1;
    }
    std::vector<const float*> bufs;
    for (int t = 0; t < nt; t++) if (M->tused[t]) bufs.push_back(M->tg[t].data());
    float* G = M->grads.data();
    const long np = M->n_params;
    const int nu = (int)bufs.size();
    #pragma omp parallel for schedule(static)
    for (long i0 = 0; i0 < np; i0 += 4096) {
        const long i1 = std::min(np, i0 + 4096);
        if (nu == 0) { for (long i = i0; i < i1; i++) G[i] = 0.f; continue; }
        for (long i = i0; i < i1; i++) G[i] = bufs[0][i];
        for (int u = 1; u < nu; u++)
            for (long i = i0; i < i1; i++) G[i] += bufs[u][i];
    }
    LossAcc tot;
    for (int t = 0; t < nt; t++) tot.add(accs[t]);
    return tot;
}

// Few sequences, many threads: the per-frame convolutional work of all sequences is spread over the threads
// (n_seq * T work items, for any context length T); the per-sequence parts run one thread per sequence.
// Sequence q keeps its activations in workspace ws[q]; a thread uses its own workspace's scratch for frame items.
// Gradient buffers: threads that run sequence work zero and use their whole buffer; threads that only run frame
// items zero and use only the encoder / decoder convolution ranges. Static scheduling: deterministic for a given
// thread count.
static LossAcc loss_and_grads_frames(Model* M, Rollout* R, const int* ids, int n_seq, const PPOConfig& pc) {
    const Dims& D = M->D;
    const bool have_pred = pc.pred_coef > 0.f || pc.drive_coef > 0.f;
    const bool have_inv = pc.inv_coef > 0.f && D.ih > 0;
    const bool have_dyn = D.dh > 0 && pc.dyn_horizon > 0 && (pc.dyn_coef > 0.f || pc.dyn_h_coef > 0.f || pc.dyn_x_coef > 0.f);
    const float inv_B = 1.0f / (float)((long)n_seq * D.K);
    const float scale = pc.pred_backbone_scale;
    double am, as;
    adv_stats(R, ids, n_seq, pc, am, as);
    const int nt = omp_get_max_threads();
    const int T = D.K, nitems = n_seq * T;
    const int nfr = std::min(nt, nitems);              // threads with frame items
    const long enc0 = M->off[C1W], enc1 = M->off[C3B] + M->sz[C3B];
    const long dec0 = M->off[D1W], dec1 = M->off[D2B] + M->sz[D2B];
    std::vector<SeqView> sv(n_seq);
    for (int q = 0; q < n_seq; q++) sv[q] = rollout_view(R, ids[q]);
    std::vector<LossAcc> accs(nt);
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* G = M->tg[tid].data();
        if (tid < n_seq) memset(G, 0, sizeof(float) * M->n_params);
        else if (tid < nfr) {
            memset(G + enc0, 0, sizeof(float) * (enc1 - enc0));
            memset(G + dec0, 0, sizeof(float) * (dec1 - dec0));
        }
        Workspace* X = M->ws[tid];
        #pragma omp for schedule(static, 1)
        for (int q = 0; q < n_seq; q++) seq_fwd_setup(M, M->ws[q], sv[q]);
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < nitems; i++) seq_fwd_encode(M, M->ws[i / T], X, sv[i / T], i % T);
        #pragma omp for schedule(static, 1)
        for (int q = 0; q < n_seq; q++) seq_fwd_core(M, M->ws[q], sv[q], have_pred);
        if (have_pred) {
            #pragma omp for schedule(static, 1)
            for (int i = 0; i < nitems; i++) seq_fwd_decode(M, M->ws[i / T], X, sv[i / T], i % T);
        }
        #pragma omp for schedule(static, 1)
        for (int q = 0; q < n_seq; q++) {
            Workspace* W = M->ws[q];
            float dls[2] = {0.f, 0.f};
            seq_loss(M, W, sv[q], pc, inv_B, (float)am, (float)as, have_pred, dls, accs[tid]);
            if (have_inv) seq_inverse(M, W, sv[q], pc, inv_B, accs[tid]);
            if (have_dyn) seq_dyn(M, W, sv[q], pc, inv_B, G, accs[tid]);
            G[M->off[LOGSTD] + 0] += dls[0];
            G[M->off[LOGSTD] + 1] += dls[1];
        }
        if (have_pred) {
            #pragma omp for schedule(static, 1)
            for (int i = 0; i < nitems; i++) seq_bwd_decode(M, M->ws[i / T], X, i % T, G);
        }
        #pragma omp for schedule(static, 1)
        for (int q = 0; q < n_seq; q++)
            seq_bwd_core(M, M->ws[q], sv[q], G, have_pred, scale, have_inv, pc.pi_detach != 0);
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < nitems; i++) seq_bwd_encode(M, M->ws[i / T], X, i % T, G, have_pred, scale);
    }
    for (int t = 0; t < nt; t++) { M->tdirty[t] = 1; M->tused[t] = t < std::max(n_seq, nfr); }
    float* G = M->grads.data();
    const long np = M->n_params;
    const int nall = std::max(n_seq, nfr);
    #pragma omp parallel for schedule(static)
    for (long i0 = 0; i0 < np; i0 += 4096) {
        const long i1 = std::min(np, i0 + 4096);
        for (long i = i0; i < i1; i++) {
            const bool conv = (i >= enc0 && i < enc1) || (i >= dec0 && i < dec1);
            const int nu = conv ? nall : n_seq;
            float g = M->tg[0][i];
            for (int u = 1; u < nu; u++) g += M->tg[u][i];
            G[i] = g;
        }
    }
    LossAcc tot;
    for (int t = 0; t < nt; t++) tot.add(accs[t]);
    return tot;
}

static LossAcc loss_and_grads(Model* M, Rollout* R, const int* ids, int n_seq, const PPOConfig& pc) {
    ensure_threads(M);
    return (n_seq < omp_get_max_threads() && !M->D.patch) ? loss_and_grads_frames(M, R, ids, n_seq, pc)
                                         : loss_and_grads_seqs(M, R, ids, n_seq, pc);
}

// Squared gradient norms of three parameter groups: [0] shared backbone (encoder + Transformer),
// [1] policy/value heads, [2] prediction heads (heads only receive their own loss's gradient), and of all
// parameters. Fixed chunks summed in parallel, then combined in order: deterministic for any thread count.
static double grad_sq_norms(const Model* M, double* group) {
    struct Chunk { long b, e; int g; };
    std::vector<Chunk> chunks;
    for (int p = 0; p < NUM_P; p++) {
        const int g = (p >= PIW && p <= VB) ? 1 : (((p >= PR1W && p < PEW) || (p >= AX1W && p <= AXFB) || p >= UOW) ? 2 : 0);
        for (long b = M->off[p]; b < M->off[p] + M->sz[p]; b += 16384)
            chunks.push_back({b, std::min(M->off[p] + M->sz[p], b + 16384), g});
    }
    std::vector<double> part(chunks.size());
    const float* G = M->grads.data();
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < (int)chunks.size(); c++) {
        double s = 0;
        for (long i = chunks[c].b; i < chunks[c].e; i++) s += (double)G[i] * G[i];
        part[c] = s;
    }
    double all = 0;
    group[0] = group[1] = group[2] = 0;
    for (size_t c = 0; c < chunks.size(); c++) { group[chunks[c].g] += part[c]; all += part[c]; }
    return all;
}

// Global gradient norm; *clip is the factor AdamW applies to the gradient (1 unless the norm exceeds max_norm).
// own_clip: the policy/value heads (group 1) get their own factor *clip_head from their own norm, and *clip comes
// from the norm of the rest; otherwise *clip_head = *clip.
static float grad_norm_and_clip(Model* M, float max_norm, double* group_acc, float* clip, bool own_clip,
                                float* clip_head) {
    double gs[3];
    const float norm = (float)sqrt(grad_sq_norms(M, gs));
    if (group_acc)
        for (int k = 0; k < 3; k++) group_acc[k] += sqrt(gs[k]);
    auto factor = [&](float n) { return (max_norm > 0.f && n > max_norm) ? max_norm / (n + 1e-6f) : 1.0f; };
    *clip = factor(own_clip ? (float)sqrt(gs[0] + gs[2]) : norm);
    *clip_head = own_clip ? factor((float)sqrt(gs[1])) : *clip;
    return norm;
}

static void adamw_step(Model* M, const PPOConfig& pc, float clip, float clip_head) {
    M->step++;
    const float b1 = pc.beta1, b2 = pc.beta2;
    const float bc1 = 1.0f - powf(b1, (float)M->step), bc2 = 1.0f - powf(b2, (float)M->step);
    const long np = M->n_params;
    const long h0 = M->off[PIW], h1 = M->off[VB] + M->sz[VB];   // the policy/value heads: PIW..VB, contiguous
    float *P = M->params.data(), *G = M->grads.data(), *m = M->m.data(), *v = M->v.data();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < np; i++) {
        const float c = (i >= h0 && i < h1) ? clip_head : clip;
        const float g = c == 1.0f ? G[i] : G[i] * c;
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        float mh = m[i] / bc1, vh = v[i] / bc2;
        P[i] -= pc.lr * (mh / (sqrtf(vh) + pc.adam_eps) + pc.weight_decay * P[i]);
    }
    refresh_wt(M);
}

// =====================================================================================
// Checkpoints: int32 header[256] (magic, version, config, n_params, has_opt, step) + float32 params [+ m, v]
// =====================================================================================
static void config_to_header(const VMConfig& c, int* h) {
    int k = 2;
    h[k++] = c.img; h[k++] = c.c1; h[k++] = c.c2; h[k++] = c.c3; h[k++] = c.feat; h[k++] = c.d_model;
    h[k++] = c.n_layer; h[k++] = c.n_head; h[k++] = c.d_ff; h[k++] = c.ctx; h[k++] = c.pred_img; h[k++] = c.pc1;
    h[k++] = c.pc2; h[k++] = c.pred_hidden; h[k++] = c.use_prev_reward;
    memcpy(&h[k++], &c.logstd_init, sizeof(float));
    h[k++] = c.coord; h[k++] = c.pb;   // k == 20 here; slots 20..22 hold n_params / has_opt / step
    h[32] = c.pred_residual;            // extension fields live at 32+ (zero in older checkpoints)
    h[33] = c.pred_skip;
    h[34] = c.inv_hidden;
    h[35] = c.dyn_hidden;
    h[36] = c.one_frame;
    h[37] = c.pi_hidden;
    h[38] = c.patch;
    h[39] = c.act_token;
    h[40] = c.universal;
    h[41] = c.bins;
    h[42] = c.gray;
    h[43] = c.in_bins;
    h[44] = c.attn_split;
    h[45] = c.attn_overlap;
    h[46] = c.attn_pred_global;
    h[47] = c.attn_cols;
    h[48] = c.attn_checker;
    h[49] = c.attn_win;
    h[50] = c.uni_sum;
    h[51] = c.uni_pyr;
    h[52] = c.sum_thumb;
}
static void header_to_config(const int* h, VMConfig& c) {
    int k = 2;
    c.img = h[k++]; c.c1 = h[k++]; c.c2 = h[k++]; c.c3 = h[k++]; c.feat = h[k++]; c.d_model = h[k++];
    c.n_layer = h[k++]; c.n_head = h[k++]; c.d_ff = h[k++]; c.ctx = h[k++]; c.pred_img = h[k++]; c.pc1 = h[k++];
    c.pc2 = h[k++]; c.pred_hidden = h[k++]; c.use_prev_reward = h[k++];
    memcpy(&c.logstd_init, &h[k++], sizeof(float));
    c.coord = h[k++]; c.pb = h[k++];
    c.pred_residual = h[32];
    c.pred_skip = h[33];
    c.inv_hidden = h[34];
    c.dyn_hidden = h[35];
    c.one_frame = h[36];
    c.pi_hidden = h[37];
    c.patch = h[38];
    c.act_token = h[39];
    c.universal = h[40];
    c.bins = h[41];
    c.gray = h[42];
    c.in_bins = h[43];
    c.attn_split = h[44];
    c.attn_overlap = h[45];
    c.attn_pred_global = h[46];
    c.attn_cols = h[47];
    c.attn_checker = h[48];
    c.attn_win = h[49];
    c.uni_sum = h[50];
    c.uni_pyr = h[51];
    c.sum_thumb = h[52];
}

// =====================================================================================
// C API
// =====================================================================================
extern "C" {

void* vm_create(const VMConfig* cfg, uint64_t seed) {
    Model* M = model_new(*cfg);
    if (!M) return nullptr;
    model_init_params(M, seed);
    rng_seed(M->rng, seed ^ 0xA5A5A5A5ULL);
    refresh_wt(M);
    return M;
}

int vm_save(void* a, const char* path, int with_opt) {
    Model* M = (Model*)a;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    int h[256] = {0};
    h[0] = VM_MAGIC; h[1] = VM_VERSION;
    config_to_header(M->cfg, h);
    h[20] = (int)M->n_params; h[21] = with_opt ? 1 : 0; h[22] = (int)M->step;
    bool ok = fwrite(h, sizeof(int), 256, f) == 256;
    ok = ok && fwrite(M->params.data(), sizeof(float), M->n_params, f) == (size_t)M->n_params;
    if (with_opt) {
        ok = ok && fwrite(M->m.data(), sizeof(float), M->n_params, f) == (size_t)M->n_params;
        ok = ok && fwrite(M->v.data(), sizeof(float), M->n_params, f) == (size_t)M->n_params;
    }
    fclose(f);
    return ok ? 0 : -1;
}

void* vm_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    int h[256];
    if (fread(h, sizeof(int), 256, f) != 256 || h[0] != VM_MAGIC || h[1] != VM_VERSION) {
        fprintf(stderr, "vm_agent: bad checkpoint header: %s\n", path);
        fclose(f);
        return nullptr;
    }
    VMConfig c;
    header_to_config(h, c);
    Model* M = model_new(c);
    if (!M || M->n_params != h[20]) {
        fprintf(stderr, "vm_agent: checkpoint/config mismatch: %s\n", path);
        fclose(f);
        delete M;
        return nullptr;
    }
    bool ok = fread(M->params.data(), sizeof(float), M->n_params, f) == (size_t)M->n_params;
    if (ok && h[21]) {
        ok = fread(M->m.data(), sizeof(float), M->n_params, f) == (size_t)M->n_params;
        ok = ok && fread(M->v.data(), sizeof(float), M->n_params, f) == (size_t)M->n_params;
        M->step = h[22];
    }
    fclose(f);
    if (!ok) { delete M; return nullptr; }
    refresh_wt(M);
    return M;
}

void vm_free(void* a) {
    Model* M = (Model*)a;
    if (!M) return;
    for (Workspace* W : M->ws) delete W;
    delete M;
}

long vm_num_params(void* a) { return ((Model*)a)->n_params; }
void vm_get_config(void* a, VMConfig* out) { *out = ((Model*)a)->cfg; }
float* vm_params(void* a) { return ((Model*)a)->params.data(); }
float* vm_grads(void* a) { return ((Model*)a)->grads.data(); }
void vm_params_changed(void* a) { refresh_wt((Model*)a); }
void vm_set_threads(int n) { if (n > 0) omp_set_num_threads(n); }
int vm_get_threads(void) { return omp_get_max_threads(); }

int vm_param_layout(void* a, int idx, char* name, int name_len, long* offset, long* size) {
    Model* M = (Model*)a;
    if (idx < 0 || idx >= NUM_P) return -1;
    if (name && name_len > 0) { strncpy(name, PNAME[idx], name_len - 1); name[name_len - 1] = 0; }
    if (offset) *offset = M->off[idx];
    if (size) *size = M->sz[idx];
    return 0;
}

void* vm_ctx_create(void* a, int n_envs, uint64_t seed) {
    Model* M = (Model*)a;
    ActCtx* C = new ActCtx();
    C->M = M;
    C->N = n_envs;
    C->len.assign(n_envs, 0);
    const size_t sz = (size_t)M->D.L * n_envs * M->D.K * M->D.d;
    C->kc.assign(sz, 0.f);
    C->vc.assign(sz, 0.f);
    C->rng.resize(n_envs);
    uint64_t x = seed;
    for (int n = 0; n < n_envs; n++) rng_seed(C->rng[n], splitmix64(x));
    return C;
}
void vm_ctx_free(void* c) { delete (ActCtx*)c; }
void vm_ctx_reset(void* c, const uint8_t* mask) {
    ActCtx* C = (ActCtx*)c;
    for (int n = 0; n < C->N; n++) if (!mask || mask[n]) C->len[n] = 0;
    if (!mask) C->step = 0;
}

void vm_act(void* a, void* c, const uint8_t* images, const float* drives, const float* prev_action,
            const float* prev_reward, const uint8_t* ep_start, int flags, float* action, float* logp, float* value,
            float* mu, float* click_p, void* rollout, int t) {
    Model* M = (Model*)a;
    CK(!M->D.patch);   // patch tokens act through vm_forward on the context window
    ActCtx* C = (ActCtx*)c;
    ensure_threads(M);
    const Dims& D = M->D;
    const int N = C->N;
    const int deterministic = flags & VM_ACT_DETERMINISTIC;
    const bool peek = (flags & VM_ACT_PEEK) != 0;
    // Context starts: block boundary (every ctx committed steps) and episode starts.
    std::vector<int> saved;
    if (peek) saved = C->len;
    if (C->step % D.K == 0 || M->cfg.one_frame) for (int n = 0; n < N; n++) C->len[n] = 0;
    if (ep_start) for (int n = 0; n < N; n++) if (ep_start[n]) C->len[n] = 0;
    if (rollout) CK(!peek && t % D.K == (int)(C->step % D.K));
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        Workspace* W = M->ws[omp_get_thread_num()];
        act_one(M, W, C, n, images + (size_t)n * D.img_px, drives + 2 * n, prev_action + 3 * n, prev_reward[n],
                deterministic, action + 3 * n, logp + n, value + n, mu ? mu + 2 * n : nullptr,
                click_p ? click_p + n : nullptr);
    }
    if (peek) C->len = saved;   // KV rows written at len are overwritten by the next committed call
    else C->step++;
    if (rollout) {
        Rollout* R = (Rollout*)rollout;
        CK(R->N == N && t >= 0 && t < R->T);
        const size_t b = (size_t)t * N;
        memcpy(R->images.data() + b * D.img_px, images, (size_t)N * D.img_px);
        memcpy(R->drives.data() + b * 2, drives, sizeof(float) * 2 * N);
        memcpy(R->prev_action.data() + b * 3, prev_action, sizeof(float) * 3 * N);
        memcpy(R->prev_reward.data() + b, prev_reward, sizeof(float) * N);
        for (int n = 0; n < N; n++) R->ep_start[b + n] = ep_start ? (ep_start[n] ? 1 : 0) : 0;
        memcpy(R->action.data() + b * 3, action, sizeof(float) * 3 * N);
        memcpy(R->logp.data() + b, logp, sizeof(float) * N);
        memcpy(R->value.data() + b, value, sizeof(float) * N);
    }
}

void* vm_rollout_create(void* a, int T, int N) {
    Model* M = (Model*)a;
    if (T <= 0 || N <= 0 || T % M->D.K) { fprintf(stderr, "vm_agent: rollout T must be a multiple of ctx\n"); return nullptr; }
    Rollout* R = new Rollout();
    R->M = M; R->T = T; R->N = N;
    const size_t TN = (size_t)T * N;
    R->images.assign(TN * M->D.img_px, 0);
    R->next_small.assign(TN * M->D.small_px, 0);
    R->ep_start.assign(TN, 0);
    R->done.assign(TN, 0);
    R->drives.assign(TN * 2, 0.f);
    R->prev_action.assign(TN * 3, 0.f);
    R->prev_reward.assign(TN, 0.f);
    R->action.assign(TN * 3, 0.f);
    R->logp.assign(TN, 0.f);
    R->value.assign(TN, 0.f);
    R->reward.assign(TN, 0.f);
    R->next_drives.assign(TN * 2, 0.f);
    R->adv.assign(TN, 0.f);
    R->ret.assign(TN, 0.f);
    return R;
}
void vm_rollout_free(void* r) { delete (Rollout*)r; }

void vm_rollout_outcome(void* r, int t, const float* reward, const uint8_t* done, const uint8_t* next_images,
                        const float* next_drives) {
    Rollout* R = (Rollout*)r;
    const Dims& D = R->M->D;
    const int N = R->N;
    CK(t >= 0 && t < R->T);
    const size_t b = (size_t)t * N;
    memcpy(R->reward.data() + b, reward, sizeof(float) * N);
    for (int n = 0; n < N; n++) R->done[b + n] = done[n] ? 1 : 0;
    memcpy(R->next_drives.data() + b * 2, next_drives, sizeof(float) * 2 * N);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++)
        downsample_u8(next_images + (size_t)n * D.img_px, D.img, D.ds, R->next_small.data() + (b + n) * D.small_px);
}

void vm_rollout_store(void* r, int t, const uint8_t* images, const float* drives, const float* prev_action,
                      const float* prev_reward, const uint8_t* ep_start, const float* action) {
    Rollout* R = (Rollout*)r;
    const Dims& D = R->M->D;
    const int N = R->N;
    CK(t >= 0 && t < R->T);
    const size_t b = (size_t)t * N;
    memcpy(R->images.data() + b * D.img_px, images, (size_t)N * D.img_px);
    memcpy(R->drives.data() + b * 2, drives, sizeof(float) * 2 * N);
    memcpy(R->prev_action.data() + b * 3, prev_action, sizeof(float) * 3 * N);
    memcpy(R->prev_reward.data() + b, prev_reward, sizeof(float) * N);
    for (int n = 0; n < N; n++) R->ep_start[b + n] = ep_start ? (ep_start[n] ? 1 : 0) : 0;
    memcpy(R->action.data() + b * 3, action, sizeof(float) * 3 * N);
    std::fill(R->logp.begin() + b, R->logp.begin() + b + N, 0.f);
    std::fill(R->value.begin() + b, R->value.begin() + b + N, 0.f);
}

void vm_rollout_set_action(void* r, int t, const float* action) {
    Rollout* R = (Rollout*)r;
    CK(t >= 0 && t < R->T);
    memcpy(R->action.data() + (size_t)t * R->N * 3, action, sizeof(float) * 3 * R->N);
}

static void fill_stats(PPOStats* st, const LossAcc& acc, int n_updates, const Model* M) {
    const double n = acc.n > 0 ? acc.n : 1, pn = acc.pred_n > 0 ? acc.pred_n : 1, cn = acc.ch_n > 0 ? acc.ch_n : 1;
    st->pg_loss = (float)(acc.pg / n);
    st->v_loss = (float)(acc.v / n);
    st->entropy = (float)(acc.ent / n);
    st->approx_kl = (float)(acc.kl / n);
    st->clipfrac = (float)(acc.clip / n);
    st->pred_mse = (float)(acc.pred_se / pn);
    st->copy_mse = (float)(acc.copy_se / pn);
    st->pred_changed_mse = (float)(acc.pred_ch_se / cn);
    st->copy_changed_mse = (float)(acc.copy_ch_se / cn);
    st->changed_frac = (float)(acc.ch_n / (pn * M->D.small_px));
    st->drive_mse = (float)(acc.drive_se / pn);
    st->click_p = (float)(acc.clickp / n);
    st->inv_mse = acc.inv_n > 0 ? (float)(acc.inv_se / acc.inv_n) : 0.f;
    st->inv_r2 = acc.inv_var > 0 ? (float)(1.0 - acc.inv_se / acc.inv_var) : 0.f;
    st->dyn_mse = acc.dyn_n > 0 ? (float)(acc.dyn_se / acc.dyn_n) : 0.f;
    st->dyn_hunger_mse = acc.dynh_n > 0 ? (float)(acc.dynh_se / acc.dynh_n) : 0.f;
    st->dyn_recon_mse = acc.dynx_n > 0 ? (float)(acc.dynx_se / acc.dynx_n) : 0.f;
    st->move_nll = (float)(acc.move / n);
    st->act_drive_mse = (float)(acc.act_dse / pn);
    {
        float* ce[3] = {&st->uni_ce_body, &st->uni_ce_patch, &st->uni_ce_move};
        float* se[3] = {&st->uni_mse_body, &st->uni_mse_patch, &st->uni_mse_move};
        for (int k = 0; k < 3; k++) if (acc.un[k] > 0) { *ce[k] = (float)(acc.uce[k] / acc.un[k]); *se[k] = (float)(acc.use_[k] / acc.un[k]); }
    }
    st->act_frame_mse = (float)(acc.act_fse / pn);
    st->updates_done = n_updates;
    const float* ls = M->params.data() + M->off[LOGSTD];
    st->std_x = expf(clampf_(ls[0], LS_MIN, LS_MAX));
    st->std_y = expf(clampf_(ls[1], LS_MIN, LS_MAX));
}

static void rollout_value_stats(const Rollout* R, PPOStats* st) {
    const size_t TN = (size_t)R->T * R->N;
    double sv = 0, sr = 0, sa = 0, sa2 = 0, sr2 = 0, sd = 0, sd2 = 0;
    for (size_t i = 0; i < TN; i++) {
        double r = R->ret[i], v = R->value[i], a = R->adv[i];
        sv += v; sr += r; sr2 += r * r; sa += a; sa2 += a * a;
        sd += r - v; sd2 += (r - v) * (r - v);
    }
    const double n = (double)TN;
    double var_r = sr2 / n - (sr / n) * (sr / n), var_d = sd2 / n - (sd / n) * (sd / n);
    st->explained_var = var_r > 1e-12 ? (float)(1.0 - var_d / var_r) : 0.f;
    st->value_mean = (float)(sv / n);
    st->return_mean = (float)(sr / n);
    st->adv_mean = (float)(sa / n);
    st->adv_std = (float)sqrt(std::max(0.0, sa2 / n - (sa / n) * (sa / n)));
}

void vm_update(void* a, void* r, const float* last_value, const PPOConfig* pcp, PPOStats* st) {
    Model* M = (Model*)a;
    Rollout* R = (Rollout*)r;
    const PPOConfig pc = *pcp;
    memset(st, 0, sizeof(PPOStats));
    compute_gae(R, last_value, pc);
    rollout_value_stats(R, st);
    const int nseq = R->N * (R->T / M->D.K);
    std::vector<int> ids(nseq);
    for (int i = 0; i < nseq; i++) ids[i] = i;
    const int nmb = std::max(1, std::min(pc.minibatches, nseq));
    const int mb = nseq / nmb;
    LossAcc all;
    double gn_sum = 0, gn_groups[3] = {0, 0, 0};
    int n_upd = 0, epochs_done = 0;
    for (int ep = 0; ep < pc.epochs; ep++) {
        for (int i = nseq - 1; i > 0; i--) {
            int j = (int)(rng_next(M->rng) % (uint64_t)(i + 1));
            std::swap(ids[i], ids[j]);
        }
        LossAcc ep_acc;
        for (int k = 0; k < nmb; k++) {
            LossAcc acc = loss_and_grads(M, R, ids.data() + (size_t)k * mb, mb, pc);
            float clip, clip_head;
            gn_sum += grad_norm_and_clip(M, pc.max_grad_norm, gn_groups, &clip, pc.pi_own_clip != 0, &clip_head);
            adamw_step(M, pc, clip, clip_head);
            ep_acc.add(acc);
            n_upd++;
        }
        all.add(ep_acc);
        epochs_done++;
        if (pc.use_ppo && pc.target_kl > 0.f && ep_acc.n > 0 && ep_acc.kl / ep_acc.n > pc.target_kl) break;
    }
    fill_stats(st, all, n_upd, M);
    st->grad_norm = (float)(gn_sum / std::max(1, n_upd));
    st->gn_backbone = (float)(gn_groups[0] / std::max(1, n_upd));
    st->gn_policy = (float)(gn_groups[1] / std::max(1, n_upd));
    st->gn_pred = (float)(gn_groups[2] / std::max(1, n_upd));
    st->epochs_done = epochs_done;
}

float vm_debug_grads(void* a, void* r, const float* last_value, const PPOConfig* pcp, const int* seq_ids, int n_seq,
                     PPOStats* st) {
    Model* M = (Model*)a;
    Rollout* R = (Rollout*)r;
    compute_gae(R, last_value, *pcp);
    LossAcc acc = loss_and_grads(M, R, seq_ids, n_seq, *pcp);
    if (st) { memset(st, 0, sizeof(PPOStats)); rollout_value_stats(R, st); fill_stats(st, acc, 0, M); }
    return (float)(acc.total / std::max(1.0, acc.n));
}

void vm_forward(void* a, int S, int T, const uint8_t* images, const float* drives, const float* prev_action,
                const float* prev_reward, const uint8_t* ep_start, const float* pred_actions, float* z, float* value,
                float* mu, float* click_p, float* pred_frames, float* pred_drives) {
    Model* M = (Model*)a;
    const Dims& D = M->D;
    CK(T >= 1 && T <= D.K);
    CK(!D.uni);   // the universal model: vm_forward_uni
    ensure_threads(M);
    const bool want_pred = pred_actions != nullptr;
    std::vector<SeqView> svs(S);
    for (int s = 0; s < S; s++) {
        const size_t b = (size_t)s * T;
        SeqView& sv = svs[s];
        sv.T = T;
        sv.img = images + b * D.img_px; sv.img_st = D.img_px;
        sv.drives = drives + b * 2; sv.drv_st = 2;
        sv.prev_action = prev_action + b * 3; sv.pa_st = 3;
        sv.prev_reward = prev_reward + b; sv.pr_st = 1;
        sv.ep_start = ep_start ? ep_start + b : nullptr; sv.es_st = 1;
        sv.pred_action = want_pred ? pred_actions + b * 3 : nullptr; sv.pact_st = 3;
    }
    auto outputs = [&](const Workspace* W, int s) {
        for (int t = 0; t < T; t++) {
            const size_t i = (size_t)s * T + t;
            if (z) memcpy(z + i * D.d, (D.patch ? W->zm.data() : W->lnf.data()) + (size_t)t * D.d, sizeof(float) * D.d);
            if (value) value[i] = W->val[t];
            if (mu) { mu[2 * i] = tanhf(W->pi[3 * t]); mu[2 * i + 1] = tanhf(W->pi[3 * t + 1]); }
            if (click_p) click_p[i] = sigmoidf_(W->pi[3 * t + 2]);
            if (want_pred && pred_frames) {
                const float* po = W->pout.data() + (size_t)t * D.small_px;
                float* dst = pred_frames + i * D.small_px;
                const int pp = D.pimg * D.pimg;
                for (int y = 0; y < D.pimg; y++)
                    for (int x = 0; x < D.pimg; x++)
                        for (int c = 0; c < 3; c++) dst[(y * D.pimg + x) * 3 + c] = po[c * pp + y * D.pimg + x];
            }
            if (want_pred && pred_drives) { pred_drives[2 * i] = W->pdrv[2 * t]; pred_drives[2 * i + 1] = W->pdrv[2 * t + 1]; }
        }
    };
    if (S < omp_get_max_threads() && !D.patch) {   // few sequences: spread the frames over the threads, as in training
        const int n = S * T;
        #pragma omp parallel
        {
            Workspace* X = M->ws[omp_get_thread_num()];
            #pragma omp for schedule(static, 1)
            for (int s = 0; s < S; s++) seq_fwd_setup(M, M->ws[s], svs[s]);
            #pragma omp for schedule(static, 1)
            for (int i = 0; i < n; i++) seq_fwd_encode(M, M->ws[i / T], X, svs[i / T], i % T);
            #pragma omp for schedule(static, 1)
            for (int s = 0; s < S; s++) seq_fwd_core(M, M->ws[s], svs[s], want_pred);
            if (want_pred) {
                #pragma omp for schedule(static, 1)
                for (int i = 0; i < n; i++) seq_fwd_decode(M, M->ws[i / T], X, svs[i / T], i % T);
            }
            #pragma omp for schedule(static, 1)
            for (int s = 0; s < S; s++) outputs(M->ws[s], s);
        }
        return;
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int s = 0; s < S; s++) {
        Workspace* W = M->ws[omp_get_thread_num()];
        seq_forward(M, W, svs[s], want_pred);
        outputs(W, s);
    }
}

void vm_forward_uni(void* a, int S, int T, const uint8_t* images, const float* drives, const float* actions,
                    const uint8_t* ep_start, float* values, float* move_probs, float* next_body) {
    Model* M = (Model*)a;
    const Dims& D = M->D;
    CK(D.uni && T >= 1 && T <= D.K);
    ensure_threads(M);
    const int nt = D.ntok, V = D.V, K = D.nbin;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int s = 0; s < S; s++) {
        Workspace* W = M->ws[omp_get_thread_num()];
        const size_t b = (size_t)s * T;
        SeqView sv;
        sv.T = T;
        sv.img = images + b * D.img_px; sv.img_st = D.img_px;
        sv.drives = drives + b * 2; sv.drv_st = 2;
        sv.action = actions + b * 3; sv.act_st = 3;
        sv.ep_start = ep_start ? ep_start + b : nullptr; sv.es_st = 1;
        seq_forward(M, W, sv, false);
        for (int i = 0; i < T * nt; i++) {
            const bool mv = (i % nt) == nt - 2;   // the last patch: its prediction is the move
            const bool nb = (i % nt) == nt - 1;   // the move token: its prediction is the next body
            if (!values && !(move_probs && mv) && !(next_body && nb)) continue;
            uni_head_row(M, W->lnf.data() + (size_t)i * D.d, W->u_row.data(), W->u_prob.data());
            const float* p = W->u_prob.data();
            if (values) {
                float* out = values + (b * nt + i) * V;
                for (int u = 0; u < V; u++) {
                    float ev = 0.f;
                    for (int k = 0; k < K; k++) ev += p[(size_t)u * K + k] * (float)k / (K - 1);
                    out[u] = ev;
                }
            }
            if (move_probs && mv) memcpy(move_probs + (b + i / nt) * 3 * K, p, sizeof(float) * 3 * K);
            if (next_body && nb)
                for (int u = 0; u < 2; u++) {
                    float ev = 0.f;
                    for (int k = 0; k < K; k++) ev += p[(size_t)u * K + k] * (float)k / (K - 1);
                    next_body[(b + i / nt) * 2 + u] = ev;
                }
        }
    }
}

// Encoder only: conv-3 map a3 [n, c3*(img/8)^2] and visual feature vf [n, feat] (either may be NULL).
void vm_attention(void* a, int T, const uint8_t* images, const float* drives, const float* prev_action,
                  const float* prev_reward, const uint8_t* ep_start, float* att_out) {
    Model* M = (Model*)a;
    const Dims& D = M->D;
    CK(T >= 1 && T <= D.K);
    ensure_threads(M);
    SeqView sv;
    sv.T = T;
    sv.img = images; sv.img_st = D.img_px;
    sv.drives = drives; sv.drv_st = 2;
    sv.prev_action = prev_action; sv.pa_st = 3;
    sv.prev_reward = prev_reward; sv.pr_st = 1;
    sv.ep_start = ep_start; sv.es_st = 1;
    Workspace* W = M->ws[0];
    seq_forward(M, W, sv, false);
    const int Tt = D.patch ? T * D.ntok : T, KT = D.KT;
    const int* ss = D.patch ? W->tss.data() : W->ss.data();
    for (int l = 0; l < D.L; l++)
        for (int h = 0; h < D.nh; h++)
            for (int i = 0; i < Tt; i++) {
                const float* row = W->att.data() + (((size_t)l * D.nh + h) * KT + i) * KT;
                float* dst = att_out + (((size_t)l * D.nh + h) * Tt + i) * Tt;
                const int lo = ss[i], hi = (D.patch && !D.uni) ? W->tse[i] : i;
                for (int j = 0; j < Tt; j++) dst[j] = (j >= lo && j <= hi) ? row[j] : 0.f;
            }
}

void vm_encode(void* a, int n, const uint8_t* images, float* a3_out, float* vf_out) {
    Model* M = (Model*)a;
    const Dims& D = M->D;
    CK(!D.patch);
    ensure_threads(M);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        Workspace* W = M->ws[omp_get_thread_num()];
        encode_frame(M, W, W->scr.data(), images + (size_t)i * D.img_px, 0);
        if (a3_out) memcpy(a3_out + (size_t)i * D.enc_flat, W->a3.data(), sizeof(float) * D.enc_flat);
        if (vf_out) {
            linear_fwd(W->vf.data(), W->a3.data(), M->wt.data() + M->off[EFW], M->params.data() + M->off[EFB], 1,
                       D.enc_flat, D.feat);
            for (int k = 0; k < D.feat; k++) vf_out[(size_t)i * D.feat + k] = W->vf[k] > 0.f ? W->vf[k] : 0.f;
        }
    }
}

// Inverse-dynamics predictions (dx, dy) [n,2] for n frame pairs (requires inv_hidden > 0).
void vm_inverse(void* a, int n, const uint8_t* img0, const uint8_t* img1, float* out) {
    Model* M = (Model*)a;
    const Dims& D = M->D;
    CK(D.ih > 0);
    ensure_threads(M);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        Workspace* W = M->ws[omp_get_thread_num()];
        const float* P = M->params.data();
        const float* WT = M->wt.data();
        for (int f = 0; f < 2; f++) {
            encode_frame(M, W, W->scr.data(), (f ? img1 : img0) + (size_t)i * D.img_px, 0);
            float* v = W->iv_in.data() + (size_t)f * D.feat;
            linear_fwd(v, W->a3.data(), WT + M->off[EFW], P + M->off[EFB], 1, D.enc_flat, D.feat);
            for (int k = 0; k < D.feat; k++) v[k] = v[k] > 0.f ? v[k] : 0.f;
        }
        linear_fwd(W->iv_h.data(), W->iv_in.data(), WT + M->off[IV1W], P + M->off[IV1B], 1, 2 * D.feat, D.ih);
        for (int k = 0; k < D.ih; k++) W->iv_h[k] = W->iv_h[k] > 0.f ? W->iv_h[k] : 0.f;
        linear_fwd(out + 2 * i, W->iv_h.data(), WT + M->off[IV2W], P + M->off[IV2B], 1, D.ih, 2);
    }
}

// Latent-dynamics imagination (see vm_agent.h). Rollouts are processed in chunks of up to 64 per batched step.
void vm_imagine(void* a, int n, int H, const float* s0, const float* h0, const float* actions, float* hunger,
                float* latents, float* frames) {
    Model* M = (Model*)a;
    const Dims& D = M->D;
    CK(D.dh > 0 && H >= 1);
    const int F = D.feat, Hd = D.dh, R = D.rpx, UT = F + 3, UH = F + 4, CH = 64;
    const int nch = (n + CH - 1) / CH;
    #pragma omp parallel
    {
        std::vector<float> s((size_t)CH * F), s2((size_t)CH * F), uT((size_t)CH * UT), uH((size_t)CH * UH);
        std::vector<float> hT((size_t)CH * Hd), hH((size_t)CH * Hd), h(CH), hX((size_t)CH * Hd), x((size_t)CH * R);
        #pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < nch; c++) {
            const int i0 = c * CH, m = std::min(CH, n - i0);
            memcpy(s.data(), s0 + (size_t)i0 * F, sizeof(float) * m * F);
            for (int i = 0; i < m; i++) h[i] = h0[i0 + i];
            for (int k = 0; k < H; k++) {
                for (int i = 0; i < m; i++) {
                    float e[3];
                    executed3(actions + ((size_t)(i0 + i) * H + k) * 3, e);
                    float* ut = uT.data() + (size_t)i * UT;
                    float* uh = uH.data() + (size_t)i * UH;
                    memcpy(ut, s.data() + (size_t)i * F, sizeof(float) * F);
                    memcpy(uh, s.data() + (size_t)i * F, sizeof(float) * F);
                    ut[F] = e[0]; ut[F + 1] = e[1]; ut[F + 2] = e[2];
                    uh[F] = h[i]; uh[F + 1] = e[0]; uh[F + 2] = e[1]; uh[F + 3] = e[2];
                }
                dyn_step(M, m, uT.data(), uH.data(), hT.data(), hH.data(), h.data(), s2.data());
                std::swap(s, s2);
                for (int i = 0; i < m; i++) hunger[(size_t)(i0 + i) * H + k] = h[i];
                if (latents)
                    for (int i = 0; i < m; i++)
                        memcpy(latents + ((size_t)(i0 + i) * H + k) * F, s.data() + (size_t)i * F, sizeof(float) * F);
                if (frames) {
                    dyn_decode(M, m, s.data(), hX.data(), x.data());
                    for (int i = 0; i < m; i++)
                        memcpy(frames + ((size_t)(i0 + i) * H + k) * R, x.data() + (size_t)i * R, sizeof(float) * R);
                }
            }
        }
    }
}

void vm_decode(void* a, int n, const float* latents, float* frames) {
    Model* M = (Model*)a;
    const Dims& D = M->D;
    CK(D.dh > 0);
    const int CH = 64, nch = (n + CH - 1) / CH;
    #pragma omp parallel
    {
        std::vector<float> hX((size_t)CH * D.dh);
        #pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < nch; c++) {
            const int i0 = c * CH, m = std::min(CH, n - i0);
            dyn_decode(M, m, latents + (size_t)i0 * D.feat, hX.data(), frames + (size_t)i0 * D.rpx);
        }
    }
}

}  // extern "C"
