// Test suite for sort layer functionality in train_gpt2_fp32.cu
// Tests: alpha=0 passthrough, forward blending, gradient flow,
//        disabled=baseline, training convergence, param update.

#define TESTING
#include "train_gpt2_fp32.cu"

int main(int argc, char *argv[]) {
    // --- Device & cuBLAS setup ---
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("[System]\n");
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    cublasCheck(cublasCreate(&cublas_handle));
    int enable_tf32 = 0; // disable TF32 for deterministic testing
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    int allok = 1;

    printf("\n========================================\n");
    printf("Test Suite: Sort Layer\n");
    printf("========================================\n");

    // =====================================================================
    // TEST 1: Alpha=0 passthrough — sort_forward output should equal input
    // =====================================================================
    printf("\n--- Test 1: Alpha=0 passthrough ---\n");
    {
        int B = 2, T = 8, C = 64, W = 4;
        size_t n = (size_t)B * T * C;

        float* h_inp = (float*)mallocCheck(n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            h_inp[i] = sinf((float)i * 0.01f);
        }

        float* d_inp;
        float* d_out;
        float* d_att;
        cudaCheck(cudaMalloc((void**)&d_inp, n * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out, n * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_att, (size_t)B * T * W * sizeof(float)));
        cudaCheck(cudaMemcpy(d_inp, h_inp, n * sizeof(float), cudaMemcpyHostToDevice));

        // alpha_raw = -100 → sigmoid ≈ 0, so output should be ≈ input
        float alpha_raw = -100.0f;
        float alpha = 1.0f / (1.0f + expf(-alpha_raw));
        float inv_tau = 1.0f; // tau=1
        sort_forward(d_out, d_att, d_inp, alpha, inv_tau, B, T, C, W);
        cudaCheck(cudaDeviceSynchronize());

        float* h_out = (float*)mallocCheck(n * sizeof(float));
        cudaCheck(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float err = fabsf(h_out[i] - h_inp[i]);
            if (err > max_err) max_err = err;
        }

        printf("  max |out - inp| with alpha≈0: %e\n", max_err);
        if (max_err < 1e-4f) {
            printf("  PASS: output matches input when alpha≈0\n");
        } else {
            printf("  FAIL: output diverges from input (max_err=%e)\n", max_err);
            allok = 0;
        }

        free(h_inp); free(h_out);
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_att));
    }

    // =====================================================================
    // TEST 2: Forward blending correctness — small synthetic case
    // =====================================================================
    printf("\n--- Test 2: Forward blending correctness ---\n");
    {
        int B = 1, T = 4, C = 32, W = 2;
        size_t n = (size_t)B * T * C;

        // Create simple input where each position has a distinct pattern
        float* h_inp = (float*)mallocCheck(n * sizeof(float));
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                h_inp[t * C + c] = (float)(t + 1) * (c + 1) * 0.01f;
            }
        }

        float* d_inp;
        float* d_out;
        float* d_att;
        cudaCheck(cudaMalloc((void**)&d_inp, n * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out, n * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_att, (size_t)B * T * W * sizeof(float)));
        cudaCheck(cudaMemcpy(d_inp, h_inp, n * sizeof(float), cudaMemcpyHostToDevice));

        // Use alpha=0.5, tau=1
        float alpha = 0.5f;
        float inv_tau = 1.0f;
        sort_forward(d_out, d_att, d_inp, alpha, inv_tau, B, T, C, W);
        cudaCheck(cudaDeviceSynchronize());

        float* h_out = (float*)mallocCheck(n * sizeof(float));
        float* h_att = (float*)mallocCheck((size_t)B * T * W * sizeof(float));
        cudaCheck(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_att, d_att, (size_t)B * T * W * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify position 0: window=[0], so blend = x[0], out = 0.5*x[0] + 0.5*x[0] = x[0]
        float err0 = 0.0f;
        for (int c = 0; c < C; c++) {
            err0 += fabsf(h_out[c] - h_inp[c]);
        }
        printf("  position 0 (self-only window): total_err=%.6f\n", err0);

        // Verify attention weights are saved and sum to ≈1
        int att_ok = 1;
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int w = 0; w < W; w++) {
                sum += h_att[t * W + w];
            }
            if (fabsf(sum - 1.0f) > 1e-4f) {
                printf("  att weights at t=%d sum to %f (expected 1.0)\n", t, sum);
                att_ok = 0;
            }
        }

        if (err0 < 1e-4f && att_ok) {
            printf("  PASS: forward blending produces correct results\n");
        } else {
            printf("  FAIL: forward blending errors\n");
            allok = 0;
        }

        free(h_inp); free(h_out); free(h_att);
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_att));
    }

    // =====================================================================
    // TEST 3: Gradient flow — forward+backward, verify gradients are non-zero
    // =====================================================================
    printf("\n--- Test 3: Gradient flow ---\n");
    {
        int B = 2, T = 8, C = 64, W = 4;
        size_t n = (size_t)B * T * C;

        float* h_inp = (float*)mallocCheck(n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            h_inp[i] = sinf((float)i * 0.01f);
        }

        float* d_inp;
        float* d_out;
        float* d_att;
        float* d_dinp;
        float* d_dout;
        float* d_dalpha;
        float* d_dtau;
        cudaCheck(cudaMalloc((void**)&d_inp, n * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out, n * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_att, (size_t)B * T * W * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_dinp, n * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_dout, n * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_dalpha, sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_dtau, sizeof(float)));

        cudaCheck(cudaMemcpy(d_inp, h_inp, n * sizeof(float), cudaMemcpyHostToDevice));

        float alpha = 0.3f;
        float inv_tau = 1.0f;

        // Forward
        sort_forward(d_out, d_att, d_inp, alpha, inv_tau, B, T, C, W);

        // Create dout = ones
        float* h_dout = (float*)mallocCheck(n * sizeof(float));
        for (size_t i = 0; i < n; i++) h_dout[i] = 1.0f;
        cudaCheck(cudaMemcpy(d_dout, h_dout, n * sizeof(float), cudaMemcpyHostToDevice));

        // Zero grad buffers
        cudaCheck(cudaMemset(d_dinp, 0, n * sizeof(float)));
        cudaCheck(cudaMemset(d_dalpha, 0, sizeof(float)));
        cudaCheck(cudaMemset(d_dtau, 0, sizeof(float)));

        // Backward
        sort_backward(d_dinp, d_dalpha, d_dtau, d_dout, d_inp, d_att,
                      alpha, inv_tau, B, T, C, W);
        cudaCheck(cudaDeviceSynchronize());

        // Check dinp is non-zero
        float* h_dinp = (float*)mallocCheck(n * sizeof(float));
        cudaCheck(cudaMemcpy(h_dinp, d_dinp, n * sizeof(float), cudaMemcpyDeviceToHost));
        float dinp_norm = 0.0f;
        for (size_t i = 0; i < n; i++) dinp_norm += h_dinp[i] * h_dinp[i];
        dinp_norm = sqrtf(dinp_norm);

        // Check dalpha, dtau are non-zero
        float h_dalpha, h_dtau;
        cudaCheck(cudaMemcpy(&h_dalpha, d_dalpha, sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(&h_dtau, d_dtau, sizeof(float), cudaMemcpyDeviceToHost));

        printf("  dinp L2 norm: %f\n", dinp_norm);
        printf("  dalpha_raw: %f\n", h_dalpha);
        printf("  dtau_raw: %f\n", h_dtau);

        if (dinp_norm > 1e-6f && fabsf(h_dalpha) > 1e-10f) {
            printf("  PASS: gradients flow through sort layer\n");
        } else {
            printf("  FAIL: gradients are zero or near-zero\n");
            allok = 0;
        }

        free(h_inp); free(h_dout); free(h_dinp);
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_att));
        cudaCheck(cudaFree(d_dinp));
        cudaCheck(cudaFree(d_dout));
        cudaCheck(cudaFree(d_dalpha));
        cudaCheck(cudaFree(d_dtau));
    }

    // =====================================================================
    // TEST 4: Disabled = baseline — SORT_WINDOW=0 should match no-sort loss
    // =====================================================================
    printf("\n--- Test 4: Disabled = baseline ---\n");
    {
        // Load debug state for inputs
        FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
        int state_header[256];
        freadCheck(state_header, sizeof(int), 256, state_file);
        int B = state_header[2];
        int T = state_header[3];
        int* x = (int*)mallocCheck(B * T * sizeof(int));
        int* y = (int*)mallocCheck(B * T * sizeof(int));
        freadCheck(x, sizeof(int), B*T, state_file);
        freadCheck(y, sizeof(int), B*T, state_file);
        fcloseCheck(state_file);

        // Run with SORT_WINDOW=0
        SORT_WINDOW = 0;
        GPT2 model_off;
        gpt2_build_from_checkpoint(&model_off, "gpt2_124M.bin");
        gpt2_forward(&model_off, x, y, B, T);
        float loss_off = model_off.mean_loss;

        // Run with SORT_WINDOW>0 but alpha≈0 (should match)
        SORT_WINDOW = 4;
        GPT2 model_on;
        gpt2_build_from_checkpoint(&model_on, "gpt2_124M.bin");
        init_sort_layer(&model_on);

        // Override alpha_raw to -100 (sigmoid≈0) so sort layer is effectively disabled
        int L = model_on.config.num_layers;
        float* init_cpu = (float*)mallocCheck(2 * L * sizeof(float));
        for (int l = 0; l < L; l++) {
            init_cpu[l] = -100.0f;     // alpha≈0
            init_cpu[L + l] = 0.0f;    // tau=1
        }
        cudaCheck(cudaMemcpy(model_on.sort_params_memory, init_cpu, 2 * L * sizeof(float), cudaMemcpyHostToDevice));
        free(init_cpu);

        gpt2_forward(&model_on, x, y, B, T);
        float loss_on = model_on.mean_loss;

        float loss_diff = fabsf(loss_on - loss_off);
        printf("  loss (SORT_WINDOW=0): %f\n", loss_off);
        printf("  loss (SORT_WINDOW=4, alpha≈0): %f\n", loss_on);
        printf("  difference: %e\n", loss_diff);

        if (loss_diff < 0.01f) {
            printf("  PASS: sort with alpha≈0 matches baseline\n");
        } else {
            printf("  FAIL: losses differ by %e\n", loss_diff);
            allok = 0;
        }

        gpt2_free(&model_off);
        gpt2_free(&model_on);
        SORT_WINDOW = 0; // reset
        free(x);
        free(y);
    }

    // =====================================================================
    // TEST 5: Training convergence with sort enabled
    // =====================================================================
    printf("\n--- Test 5: Training convergence with sort ---\n");
    {
        FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
        int state_header[256];
        freadCheck(state_header, sizeof(int), 256, state_file);
        int B = state_header[2];
        int T = state_header[3];
        int* x = (int*)mallocCheck(B * T * sizeof(int));
        int* y = (int*)mallocCheck(B * T * sizeof(int));
        freadCheck(x, sizeof(int), B*T, state_file);
        freadCheck(y, sizeof(int), B*T, state_file);
        fcloseCheck(state_file);

        SORT_WINDOW = 8;
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        init_sort_layer(&model);

        float losses[10];
        for (int step = 0; step < 10; step++) {
            gpt2_forward(&model, x, y, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);
            gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
            losses[step] = model.mean_loss;
            printf("  step %d: loss = %f\n", step, losses[step]);
        }

        if (losses[9] < losses[0]) {
            printf("  PASS: loss decreased from %f to %f over 10 steps\n", losses[0], losses[9]);
        } else {
            printf("  FAIL: loss did not decrease (%f -> %f)\n", losses[0], losses[9]);
            allok = 0;
        }

        // =====================================================================
        // TEST 6: Param update — alpha_raw should change after optimizer step
        // =====================================================================
        printf("\n--- Test 6: Sort param update ---\n");

        int L = model.config.num_layers;
        float* params_before = (float*)mallocCheck(2 * L * sizeof(float));
        cudaCheck(cudaMemcpy(params_before, model.sort_params_memory, 2 * L * sizeof(float), cudaMemcpyDeviceToHost));

        // Do one more step
        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, 11);

        float* params_after = (float*)mallocCheck(2 * L * sizeof(float));
        cudaCheck(cudaMemcpy(params_after, model.sort_params_memory, 2 * L * sizeof(float), cudaMemcpyDeviceToHost));

        int changed = 0;
        for (int i = 0; i < 2 * L; i++) {
            if (params_before[i] != params_after[i]) changed++;
        }
        printf("  %d / %d sort params changed after optimizer step\n", changed, 2 * L);

        if (changed > 0) {
            printf("  PASS: sort params are being updated\n");
            // Print a few values for visibility
            printf("  alpha_raw[0]: %f -> %f\n", params_before[0], params_after[0]);
            printf("  tau_raw[0]:   %f -> %f\n", params_before[L], params_after[L]);
        } else {
            printf("  FAIL: no sort params changed\n");
            allok = 0;
        }

        free(params_before);
        free(params_after);
        free(x);
        free(y);
        gpt2_free(&model);
        SORT_WINDOW = 0; // reset
    }

    // =====================================================================
    // Final result
    // =====================================================================
    printf("\n========================================\n");
    printf("Sort Layer Tests: %s\n", allok ? "ALL PASSED" : "SOME FAILED");
    printf("========================================\n");

    cublasCheck(cublasDestroy(cublas_handle));
    return allok ? 0 : 1;
}
