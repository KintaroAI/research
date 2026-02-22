// ============================================================================
// sort_layer_sketch.cu — Correlation-based sorting layer for the residual stream
//
// Concept (from the KintaroAI "natural intelligence" doc):
//   The thalamus receives a mixed stream of signals and sorts them by
//   temporal/statistical correlation — signals that co-occur get grouped
//   together. This layer does the same thing on the residual stream:
//   for each position, it looks at a causal window of neighbors, measures
//   representation similarity, and gently blends similar ones together.
//
// Mechanically this is windowed local attention with:
//   - No Q/K/V projections (operates directly on residual vectors)
//   - A learnable mixing coefficient alpha (controls sorting strength)
//   - A learnable temperature tau (controls sharpness of grouping)
//   - A small causal window W (enforces locality)
//
// Parameters added to model: 2 scalars (alpha_raw, tau_raw)
//   alpha = sigmoid(alpha_raw)  — keeps mixing in [0,1], starts near 0
//   tau   = exp(tau_raw)        — keeps temperature positive
//
// Activations added: sort_out (L,B,T,C), sort_att (L,B,T,W)
//
// Integration: insert between residual3 and next layer's input.
//   See bottom of file for exact integration instructions.
// ============================================================================

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Default sorting window size. 128 = ~1 frame for 100-byte frames.
// Can be overridden at compile time: nvcc -DSORT_WINDOW=64 ...
#ifndef SORT_WINDOW
#define SORT_WINDOW 128
#endif

// ----------------------------------------------------------------------------
// Forward kernel
//
// One warp (32 threads) per sequence position.
// Each warp:
//   1) Computes scaled dot-product similarity to W causal neighbors
//   2) Softmax over the window
//   3) Weighted blend of neighbor representations
//   4) Mixes blend into residual stream with learned alpha
//
// Shared memory: W floats per warp (for similarity scores)
// ----------------------------------------------------------------------------
__global__ void sort_forward_kernel(
    float* out,             // (B, T, C) sorted output
    float* sort_att,        // (B, T, W) attention weights (saved for backward)
    const float* inp,       // (B, T, C) input residual stream
    float alpha,            // sigmoid(alpha_raw), mixing strength in [0,1]
    float inv_tau,          // 1/tau = exp(-tau_raw), inverse temperature
    int B, int T, int C, int W
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // One warp per (batch, time) position
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) { return; }

    int t = idx % T;
    const float* x_i = inp + idx * C;

    // Causal window: positions [w_start, t] inclusive
    int w_start = (t - W + 1) > 0 ? (t - W + 1) : 0;
    int w_len = t - w_start + 1;

    // Shared memory for similarity scores, one buffer per warp in this block
    extern __shared__ float shared[];
    float* sim = shared + warp.meta_group_rank() * W;

    // --- Phase 1: Dot-product similarity to each neighbor ---
    float scale = rsqrtf((float)C) * inv_tau;
    float maxval = -FLT_MAX;

    for (int w = 0; w < w_len; w++) {
        int j = w_start + w;
        const float* x_j = inp + (idx - t + j) * C;  // same batch, position j

        // Distributed dot product across warp threads
        float dot = 0.0f;
        for (int c = warp.thread_rank(); c < C; c += warp.size()) {
            dot += x_i[c] * x_j[c];
        }
        dot = cg::reduce(warp, dot, cg::plus<float>{});

        float s = dot * scale;
        if (warp.thread_rank() == 0) {
            sim[w] = s;
            maxval = fmaxf(maxval, s);
        }
    }
    // Broadcast maxval to all threads in warp
    maxval = __shfl_sync(0xffffffff, maxval, 0);

    // --- Phase 2: Softmax over window ---
    // Thread 0 does softmax (W is small, serial is fine)
    if (warp.thread_rank() == 0) {
        float sumexp = 0.0f;
        for (int w = 0; w < w_len; w++) {
            sim[w] = expf(sim[w] - maxval);
            sumexp += sim[w];
        }
        float inv_sum = 1.0f / (sumexp + 1e-8f);
        for (int w = 0; w < w_len; w++) {
            sim[w] *= inv_sum;
        }
    }
    warp.sync();

    // Save attention weights for backward
    if (warp.thread_rank() == 0) {
        float* att_dst = sort_att + idx * W;
        for (int w = 0; w < W; w++) {
            att_dst[w] = (w < w_len) ? sim[w] : 0.0f;
        }
    }

    // --- Phase 3: Weighted blend + residual mix ---
    // out[i] = (1 - alpha) * x[i] + alpha * sum_j(w[j] * x[j])
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float blend = 0.0f;
        for (int w = 0; w < w_len; w++) {
            int j = w_start + w;
            blend += sim[w] * inp[(idx - t + j) * C + c];
        }
        o[c] = (1.0f - alpha) * x_i[c] + alpha * blend;
    }
}

// Forward launcher
void sort_forward(float* out, float* sort_att, const float* inp,
                  float alpha, float inv_tau,
                  int B, int T, int C, int W) {
    const int block_size = 512;  // 16 warps per block
    const int num_warps_per_block = block_size / 32;
    const int grid_size = CEIL_DIV(B * T, num_warps_per_block);
    size_t shared_mem = num_warps_per_block * W * sizeof(float);
    sort_forward_kernel<<<grid_size, block_size, shared_mem>>>(
        out, sort_att, inp, alpha, inv_tau, B, T, C, W);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Backward kernel
//
// One warp per position i. Computes:
//   1) dinp[i] += (1-alpha) * dout[i]                     (direct path)
//   2) dblend = alpha * dout[i]
//   3) dw[w] = dot(dblend, x[j])                          (weight gradient)
//   4) ds[w] = w[w] * (dw[w] - sum_k w[k]*dw[k])         (softmax backward)
//   5) dinp[i] += sum_w ds[w] * scale * x[j]              (query gradient)
//   6) dinp[j] += w[w] * dblend + ds[w] * scale * x[i]    (value + key grad)
//
// Step 6 writes to other positions → uses atomicAdd.
// dalpha, dtau accumulated via block-level reduction + atomicAdd.
//
// Shared memory: W floats (attention weights) + W floats (dw buffer) per warp
// ----------------------------------------------------------------------------
__global__ void sort_backward_kernel(
    float* dinp,            // (B, T, C) gradient w.r.t. input (accumulate)
    float* dalpha_raw,      // scalar gradient for alpha_raw (atomicAdd)
    float* dtau_raw,        // scalar gradient for tau_raw (atomicAdd)
    const float* dout,      // (B, T, C) incoming gradient
    const float* inp,       // (B, T, C) original input from forward
    const float* sort_att,  // (B, T, W) saved attention weights
    float alpha,            // sigmoid(alpha_raw)
    float inv_tau,          // exp(-tau_raw)
    int B, int T, int C, int W
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) { return; }

    int t = idx % T;
    int w_start = (t - W + 1) > 0 ? (t - W + 1) : 0;
    int w_len = t - w_start + 1;

    const float* x_i = inp + idx * C;
    const float* dout_i = dout + idx * C;
    float* dinp_i = dinp + idx * C;

    float scale = rsqrtf((float)C) * inv_tau;

    // Load saved attention weights into shared memory
    extern __shared__ float shared[];
    int wid = warp.meta_group_rank();
    float* w_buf = shared + wid * 2 * W;        // attention weights
    float* dw_buf = shared + wid * 2 * W + W;   // dw buffer

    if (warp.thread_rank() == 0) {
        const float* att_src = sort_att + idx * W;
        for (int w = 0; w < w_len; w++) {
            w_buf[w] = att_src[w];
        }
    }
    warp.sync();

    // --- Phase 1: Direct residual gradient ---
    // dinp[i] += (1 - alpha) * dout[i]
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        atomicAdd(&dinp_i[c], (1.0f - alpha) * dout_i[c]);
    }

    // --- Phase 2: Compute dw[w] = dot(alpha * dout[i], x[j]) ---
    // Also accumulate dalpha: dalpha += dot(dout[i], blend[i] - x[i])
    //   = dot(dout[i], sum_j w[j]*x[j] - x[i])
    float local_dalpha = 0.0f;

    for (int w = 0; w < w_len; w++) {
        int j = w_start + w;
        const float* x_j = inp + (idx - t + j) * C;

        float dot_dout_xj = 0.0f;
        float dot_dout_xi = 0.0f;
        for (int c = warp.thread_rank(); c < C; c += warp.size()) {
            dot_dout_xj += dout_i[c] * x_j[c];
            if (w == 0) {  // only need dot(dout, x_i) once
                dot_dout_xi += dout_i[c] * x_i[c];
            }
        }
        dot_dout_xj = cg::reduce(warp, dot_dout_xj, cg::plus<float>{});
        if (w == 0) {
            dot_dout_xi = cg::reduce(warp, dot_dout_xi, cg::plus<float>{});
        }

        if (warp.thread_rank() == 0) {
            dw_buf[w] = alpha * dot_dout_xj;
            // dalpha += w[j] * dot(dout, x_j) - (if j==self) dot(dout, x_i)
            local_dalpha += w_buf[w] * dot_dout_xj;
            if (w == w_len - 1) {
                // blend - x_i contribution: sum_j w[j]*dot(dout,x_j) - dot(dout,x_i)
                local_dalpha -= dot_dout_xi;
            }
        }
    }
    warp.sync();

    // --- Phase 3: Softmax backward ---
    // ds[w] = w[w] * (dw[w] - sum_k w[k]*dw[k])
    float sum_wdw = 0.0f;
    if (warp.thread_rank() == 0) {
        for (int w = 0; w < w_len; w++) {
            sum_wdw += w_buf[w] * dw_buf[w];
        }
        for (int w = 0; w < w_len; w++) {
            // Reuse dw_buf to store ds
            dw_buf[w] = w_buf[w] * (dw_buf[w] - sum_wdw);
        }
    }
    warp.sync();
    // dw_buf now holds ds[w]

    // --- Phase 4: Value gradient + key gradient to dinp[j] ---
    // For each j in window:
    //   dinp[j] += w[j] * alpha * dout[i]     (value gradient)
    //   dinp[j] += ds[j] * scale * x[i]        (key gradient)
    // Also: query gradient to dinp[i]
    //   dinp[i] += sum_j ds[j] * scale * x[j]

    // Accumulate query gradient into dinp[i]
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float q_grad = 0.0f;
        for (int w = 0; w < w_len; w++) {
            int j = w_start + w;
            float x_jc = inp[(idx - t + j) * C + c];
            q_grad += dw_buf[w] * scale * x_jc;    // ds[w] is in dw_buf
        }
        atomicAdd(&dinp_i[c], q_grad);
    }

    // Scatter value + key gradients to dinp[j]
    for (int w = 0; w < w_len; w++) {
        int j = w_start + w;
        float* dinp_j = dinp + (idx - t + j) * C;
        float wj = w_buf[w];
        float ds_j = dw_buf[w];  // ds stored in dw_buf after phase 3

        for (int c = warp.thread_rank(); c < C; c += warp.size()) {
            float val_grad = wj * alpha * dout_i[c];         // value gradient
            float key_grad = ds_j * scale * x_i[c];          // key gradient
            atomicAdd(&dinp_j[c], val_grad + key_grad);
        }
    }

    // --- Phase 5: Parameter gradients ---
    // dalpha_raw: d/d(alpha_raw) = dalpha * alpha * (1 - alpha)  (sigmoid derivative)
    // dtau_raw:   need dot product of ds with original similarities
    //             but we don't have original sims saved. Recompute:
    float local_dtau = 0.0f;
    for (int w = 0; w < w_len; w++) {
        int j = w_start + w;
        const float* x_j = inp + (idx - t + j) * C;

        // Recompute dot(x_i, x_j) for dtau
        float dot = 0.0f;
        for (int c = warp.thread_rank(); c < C; c += warp.size()) {
            dot += x_i[c] * x_j[c];
        }
        dot = cg::reduce(warp, dot, cg::plus<float>{});

        if (warp.thread_rank() == 0) {
            // ds/dtau = ds[w] * d(dot*scale)/dtau
            //         = ds[w] * dot * rsqrt(C) * d(inv_tau)/dtau
            //         = ds[w] * dot * rsqrt(C) * (-inv_tau)   (since inv_tau = exp(-tau_raw))
            local_dtau += dw_buf[w] * dot * rsqrtf((float)C) * (-inv_tau);
        }
    }

    // Block-level reduction for dalpha, dtau to reduce atomicAdd contention
    if (warp.thread_rank() == 0) {
        // dalpha_raw = dalpha * sigmoid_derivative = dalpha * alpha * (1 - alpha)
        atomicAdd(dalpha_raw, local_dalpha * alpha * (1.0f - alpha));
        atomicAdd(dtau_raw, local_dtau);
    }
}

// Backward launcher
void sort_backward(float* dinp, float* dalpha_raw, float* dtau_raw,
                   const float* dout, const float* inp, const float* sort_att,
                   float alpha, float inv_tau,
                   int B, int T, int C, int W) {
    const int block_size = 512;
    const int num_warps_per_block = block_size / 32;
    const int grid_size = CEIL_DIV(B * T, num_warps_per_block);
    // 2 buffers per warp: W (attention weights) + W (dw/ds scratch)
    size_t shared_mem = num_warps_per_block * 2 * W * sizeof(float);
    sort_backward_kernel<<<grid_size, block_size, shared_mem>>>(
        dinp, dalpha_raw, dtau_raw, dout, inp, sort_att,
        alpha, inv_tau, B, T, C, W);
    cudaCheck(cudaGetLastError());
}


// ============================================================================
// INTEGRATION GUIDE
//
// Below is exactly what to add/change in train_gpt2_fp32.cu to wire this in.
// ============================================================================

/*

// ---- 1. Add sorting parameters to ParameterTensors ----
// After: float* lnfb;
// Add:
//     float* sort_alpha_raw;   // (1,) — raw learnable, alpha = sigmoid(this)
//     float* sort_tau_raw;     // (1,) — raw learnable, tau = exp(this)
//
// Change: #define NUM_PARAMETER_TENSORS 18  (was 16)
//
// In fill_in_parameter_sizes(), add:
//     param_sizes[16] = 1;    // sort_alpha_raw
//     param_sizes[17] = 1;    // sort_tau_raw
//
// Initialize them after model load (in gpt2_build_from_checkpoint or similar):
//     cudaMemset(model->params.sort_alpha_raw, 0, sizeof(float));
//       → sigmoid(0) = 0.5, so sorting starts at 50% mix.
//       → Or set to -2.0f for sigmoid(-2)≈0.12, gentler start.
//     float tau_init = 0.0f;  // exp(0) = 1.0
//     cudaMemcpy(model->params.sort_tau_raw, &tau_init, sizeof(float), cudaMemcpyHostToDevice);


// ---- 2. Add sorting activations to ActivationTensors ----
// After: float* output;
// Add:
//     float* sort_out;    // (L, B, T, C)  — sorted residual stream
//     float* sort_att;    // (L, B, T, SORT_WINDOW) — saved attention weights
//
// Change: #define NUM_ACTIVATION_TENSORS 23  (was 21)
//
// In fill_in_activation_sizes(), add:
//     act_sizes[21] = L * B * T * C;            // sort_out
//     act_sizes[22] = L * B * T * SORT_WINDOW;  // sort_att


// ---- 3. Insert in gpt2_forward() ----
// After the existing layer loop body (after the final residual_forward for residual3),
// add the sorting step:

    // Sort the residual stream (thalamus-style correlation blending)
    float* l_sort_out = acts.sort_out + l * B * T * C;
    float* l_sort_att = acts.sort_att + l * B * T * SORT_WINDOW;

    // Read raw params, compute alpha and tau on host (or use a tiny kernel)
    float alpha_raw_h, tau_raw_h;
    cudaMemcpy(&alpha_raw_h, params.sort_alpha_raw, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tau_raw_h, params.sort_tau_raw, sizeof(float), cudaMemcpyDeviceToHost);
    float alpha_h = 1.0f / (1.0f + expf(-alpha_raw_h));  // sigmoid
    float inv_tau_h = expf(-tau_raw_h);                    // 1/tau

    sort_forward(l_sort_out, l_sort_att, l_residual3,
                 alpha_h, inv_tau_h, B, T, C, SORT_WINDOW);

// Then change the next layer's input from residual3 to sort_out:
//   Old: residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
//   New: residual = l == 0 ? acts.encoded : acts.sort_out + (l-1) * B * T * C;


// ---- 4. Insert in gpt2_backward() ----
// In the backward layer loop, BEFORE the existing residual_backward for residual3,
// add the sorting backward. The gradient flows:
//   dresidual (from next layer) → sort_backward → dresidual3 → rest of layer backward
//
// The backward needs dinp zeroed first (then accumulated by sort_backward):

    float* l_sort_out = acts.sort_out + l * B * T * C;
    float* l_sort_att = acts.sort_att + l * B * T * SORT_WINDOW;

    // dresidual currently holds gradient w.r.t. sort_out (next layer's input).
    // We need gradient w.r.t. residual3 (sort layer's input).
    // Use a scratch buffer for the result, then swap.
    float* dsort_inp = grads_acts.bt4c;  // reuse scratch, size B*T*4C >= B*T*C
    cudaMemset(dsort_inp, 0, B * T * C * sizeof(float));

    float alpha_raw_h, tau_raw_h;
    cudaMemcpy(&alpha_raw_h, params.sort_alpha_raw, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tau_raw_h, params.sort_tau_raw, sizeof(float), cudaMemcpyDeviceToHost);
    float alpha_h = 1.0f / (1.0f + expf(-alpha_raw_h));
    float inv_tau_h = expf(-tau_raw_h);

    sort_backward(dsort_inp,
                  grads.sort_alpha_raw, grads.sort_tau_raw,  // accumulated
                  dresidual,                                  // incoming grad
                  acts.residual3 + l * B * T * C,            // saved forward input
                  l_sort_att,
                  alpha_h, inv_tau_h, B, T, C, SORT_WINDOW);

    // dsort_inp now holds gradient w.r.t. residual3.
    // Copy it into dresidual for the rest of the layer backward.
    cudaMemcpy(dresidual, dsort_inp, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice);


// ---- 5. Add to optimizer ----
// sort_alpha_raw and sort_tau_raw are just two more scalar parameters.
// They will be updated by the existing AdamW loop since they are part of
// the contiguous parameter block. No special handling needed.
// You may want a different learning rate for them (e.g. 10x higher than
// the main LR) since they're scalars that need to move quickly.

*/
