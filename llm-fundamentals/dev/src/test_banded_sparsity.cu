// Test suite for banded sparsity functionality in train_gpt2_fp32.cu
// Tests: mask creation, mask density, dense mode, gradient masking,
//        post-update enforcement, and training convergence with sparsity.

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

    // --- Load model ---
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    int C = model.config.channels;   // 768

    printf("\n========================================\n");
    printf("Test Suite: Banded Sparsity\n");
    printf("========================================\n");

    // =====================================================================
    // TEST 1: Mask creation correctness (FC1)
    // =====================================================================
    printf("\n--- Test 1: FC1 mask creation correctness ---\n");
    {
        int bandwidth = 256;
        int OC = 4 * C; // 3072
        int IC = C;      // 768
        size_t mask_size = (size_t)OC * IC;

        float* d_mask;
        cudaCheck(cudaMalloc((void**)&d_mask, mask_size * sizeof(float)));

        int block_size = 256;
        int grid_size = CEIL_DIV(mask_size, block_size);
        create_band_mask_fc1_kernel<<<grid_size, block_size>>>(d_mask, IC, OC, bandwidth);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        // Copy to CPU and verify band pattern
        float* h_mask = (float*)mallocCheck(mask_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_mask, d_mask, mask_size * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int j = 0; j < OC && errors < 10; j++) {
            for (int i = 0; i < IC && errors < 10; i++) {
                float j_center = (float)i * OC / IC;
                float half_bw = bandwidth / 2.0f;
                float dist = fabsf((float)j - j_center);
                float expected = (dist <= half_bw) ? 1.0f : 0.0f;
                float actual = h_mask[j * IC + i];
                if (actual != expected) {
                    printf("  MISMATCH at (%d,%d): expected %.0f, got %.0f (dist=%.1f, half_bw=%.1f)\n",
                           j, i, expected, actual, dist, half_bw);
                    errors++;
                }
            }
        }
        if (errors == 0) {
            printf("  PASS: FC1 mask band pattern correct for bandwidth=%d\n", bandwidth);
        } else {
            printf("  FAIL: FC1 mask has %d errors\n", errors);
            allok = 0;
        }

        free(h_mask);
        cudaCheck(cudaFree(d_mask));
    }

    // =====================================================================
    // TEST 2: Mask creation correctness (FC2)
    // =====================================================================
    printf("\n--- Test 2: FC2 mask creation correctness ---\n");
    {
        int bandwidth = 256;
        int OC = C;      // 768
        int IC = 4 * C;  // 3072
        size_t mask_size = (size_t)OC * IC;

        float* d_mask;
        cudaCheck(cudaMalloc((void**)&d_mask, mask_size * sizeof(float)));

        int block_size = 256;
        int grid_size = CEIL_DIV(mask_size, block_size);
        create_band_mask_fc2_kernel<<<grid_size, block_size>>>(d_mask, OC, IC, bandwidth);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        float* h_mask = (float*)mallocCheck(mask_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_mask, d_mask, mask_size * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int j = 0; j < OC && errors < 10; j++) {
            for (int i = 0; i < IC && errors < 10; i++) {
                float i_center = (float)j * IC / OC;
                float half_bw = bandwidth / 2.0f;
                float dist = fabsf((float)i - i_center);
                float expected = (dist <= half_bw) ? 1.0f : 0.0f;
                float actual = h_mask[j * IC + i];
                if (actual != expected) {
                    printf("  MISMATCH at (%d,%d): expected %.0f, got %.0f\n", j, i, expected, actual);
                    errors++;
                }
            }
        }
        if (errors == 0) {
            printf("  PASS: FC2 mask band pattern correct for bandwidth=%d\n", bandwidth);
        } else {
            printf("  FAIL: FC2 mask has %d errors\n", errors);
            allok = 0;
        }

        free(h_mask);
        cudaCheck(cudaFree(d_mask));
    }

    // =====================================================================
    // TEST 3: Mask density via count_nonzero_kernel
    // =====================================================================
    printf("\n--- Test 3: Mask density check ---\n");
    {
        int bandwidth = 256;
        int OC = 4 * C;
        int IC = C;
        size_t mask_size = (size_t)OC * IC;

        float* d_mask;
        cudaCheck(cudaMalloc((void**)&d_mask, mask_size * sizeof(float)));

        int block_size = 256;
        int grid_size = CEIL_DIV(mask_size, block_size);
        create_band_mask_fc1_kernel<<<grid_size, block_size>>>(d_mask, IC, OC, bandwidth);
        cudaCheck(cudaDeviceSynchronize());

        // Count nonzeros using the kernel
        int* d_count;
        int h_count = 0;
        cudaCheck(cudaMalloc((void**)&d_count, sizeof(int)));
        cudaCheck(cudaMemset(d_count, 0, sizeof(int)));
        grid_size = CEIL_DIV(mask_size, block_size);
        count_nonzero_kernel<<<grid_size, block_size>>>(d_mask, d_count, mask_size);
        cudaCheck(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

        // Also count on CPU for reference
        float* h_mask = (float*)mallocCheck(mask_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_mask, d_mask, mask_size * sizeof(float), cudaMemcpyDeviceToHost));
        int cpu_count = 0;
        for (size_t i = 0; i < mask_size; i++) {
            if (h_mask[i] != 0.0f) cpu_count++;
        }

        float density = (float)h_count / mask_size * 100.0f;
        printf("  GPU count: %d, CPU count: %d, total: %zu, density: %.1f%%\n",
               h_count, cpu_count, mask_size, density);

        if (h_count == cpu_count && h_count > 0 && h_count < (int)mask_size) {
            printf("  PASS: density is %.1f%% (expected ~%.1f%%)\n",
                   density, (float)bandwidth / C * 100.0f);
        } else {
            printf("  FAIL: GPU/CPU count mismatch or unexpected density\n");
            allok = 0;
        }

        free(h_mask);
        cudaCheck(cudaFree(d_count));
        cudaCheck(cudaFree(d_mask));
    }

    // =====================================================================
    // TEST 4: Dense mode (BANDWIDTH=0) - masks should be NULL
    // =====================================================================
    printf("\n--- Test 4: Dense mode (BANDWIDTH=0) ---\n");
    {
        // Build a fresh model to test dense mode
        GPT2 model2;
        gpt2_build_from_checkpoint(&model2, "gpt2_124M.bin");

        // With default BANDWIDTH=0, init_fc_masks should not allocate masks
        int save_bw1 = BANDWIDTH_FC1;
        int save_bw2 = BANDWIDTH_FC2;
        BANDWIDTH_FC1 = 0;
        BANDWIDTH_FC2 = 0;
        init_fc_masks(&model2);

        if (model2.fc1_mask == NULL && model2.fc2_mask == NULL) {
            printf("  PASS: masks are NULL in dense mode\n");
        } else {
            printf("  FAIL: masks should be NULL when bandwidth=0\n");
            allok = 0;
        }

        BANDWIDTH_FC1 = save_bw1;
        BANDWIDTH_FC2 = save_bw2;
        gpt2_free(&model2);
    }

    // =====================================================================
    // TEST 5: Full init_fc_masks with both FC1 and FC2
    // =====================================================================
    printf("\n--- Test 5: init_fc_masks with bandwidth=256 ---\n");
    {
        GPT2 model3;
        gpt2_build_from_checkpoint(&model3, "gpt2_124M.bin");

        int save_bw1 = BANDWIDTH_FC1;
        int save_bw2 = BANDWIDTH_FC2;
        BANDWIDTH_FC1 = 256;
        BANDWIDTH_FC2 = 256;
        init_fc_masks(&model3);

        if (model3.fc1_mask != NULL && model3.fc2_mask != NULL) {
            printf("  PASS: both FC1 and FC2 masks allocated\n");
        } else {
            printf("  FAIL: masks should be non-NULL with bandwidth=256\n");
            allok = 0;
        }

        BANDWIDTH_FC1 = save_bw1;
        BANDWIDTH_FC2 = save_bw2;

        // =====================================================================
        // TEST 6: Gradient masking - FC weight gradients zero outside band
        // =====================================================================
        printf("\n--- Test 6: Gradient masking after backward ---\n");

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

        // Forward + backward
        gpt2_forward(&model3, x, y, B, T);
        gpt2_zero_grad(&model3);
        gpt2_backward(&model3);

        // Check FC1 gradients are zero outside mask
        int OC1 = 4 * C;
        size_t fc1_layer_size = (size_t)OC1 * C;
        float* h_fc1_grad = (float*)mallocCheck(fc1_layer_size * sizeof(float));
        float* h_fc1_mask = (float*)mallocCheck(fc1_layer_size * sizeof(float));

        // Check layer 0 as representative
        cudaCheck(cudaMemcpy(h_fc1_grad, model3.grads.fcw, fc1_layer_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_fc1_mask, model3.fc1_mask, fc1_layer_size * sizeof(float), cudaMemcpyDeviceToHost));

        int grad_errors = 0;
        for (size_t i = 0; i < fc1_layer_size && grad_errors < 10; i++) {
            if (h_fc1_mask[i] == 0.0f && h_fc1_grad[i] != 0.0f) {
                if (grad_errors < 3) {
                    printf("  FC1 grad nonzero outside mask at idx %zu: grad=%.6f\n", i, h_fc1_grad[i]);
                }
                grad_errors++;
            }
        }
        if (grad_errors == 0) {
            printf("  PASS: FC1 gradients are zero outside the band (layer 0)\n");
        } else {
            printf("  FAIL: FC1 has %d gradient values nonzero outside mask\n", grad_errors);
            allok = 0;
        }

        // Check FC2 gradients
        size_t fc2_layer_size = (size_t)C * 4 * C;
        float* h_fc2_grad = (float*)mallocCheck(fc2_layer_size * sizeof(float));
        float* h_fc2_mask = (float*)mallocCheck(fc2_layer_size * sizeof(float));

        cudaCheck(cudaMemcpy(h_fc2_grad, model3.grads.fcprojw, fc2_layer_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_fc2_mask, model3.fc2_mask, fc2_layer_size * sizeof(float), cudaMemcpyDeviceToHost));

        grad_errors = 0;
        for (size_t i = 0; i < fc2_layer_size && grad_errors < 10; i++) {
            if (h_fc2_mask[i] == 0.0f && h_fc2_grad[i] != 0.0f) {
                if (grad_errors < 3) {
                    printf("  FC2 grad nonzero outside mask at idx %zu: grad=%.6f\n", i, h_fc2_grad[i]);
                }
                grad_errors++;
            }
        }
        if (grad_errors == 0) {
            printf("  PASS: FC2 gradients are zero outside the band (layer 0)\n");
        } else {
            printf("  FAIL: FC2 has %d gradient values nonzero outside mask\n", grad_errors);
            allok = 0;
        }

        // =====================================================================
        // TEST 7: Post-update mask enforcement
        // =====================================================================
        printf("\n--- Test 7: Post-update mask enforcement ---\n");

        gpt2_update(&model3, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);

        // Check FC1 weights are zero outside mask after update
        float* h_fc1_weights = (float*)mallocCheck(fc1_layer_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_fc1_weights, model3.params.fcw, fc1_layer_size * sizeof(float), cudaMemcpyDeviceToHost));

        int weight_errors = 0;
        for (size_t i = 0; i < fc1_layer_size && weight_errors < 10; i++) {
            if (h_fc1_mask[i] == 0.0f && h_fc1_weights[i] != 0.0f) {
                if (weight_errors < 3) {
                    printf("  FC1 weight nonzero outside mask at idx %zu: weight=%.6f\n", i, h_fc1_weights[i]);
                }
                weight_errors++;
            }
        }
        if (weight_errors == 0) {
            printf("  PASS: FC1 weights are zero outside the band after update (layer 0)\n");
        } else {
            printf("  FAIL: FC1 has %d weight values nonzero outside mask after update\n", weight_errors);
            allok = 0;
        }

        // Check FC2 weights
        float* h_fc2_weights = (float*)mallocCheck(fc2_layer_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_fc2_weights, model3.params.fcprojw, fc2_layer_size * sizeof(float), cudaMemcpyDeviceToHost));

        weight_errors = 0;
        for (size_t i = 0; i < fc2_layer_size && weight_errors < 10; i++) {
            if (h_fc2_mask[i] == 0.0f && h_fc2_weights[i] != 0.0f) {
                if (weight_errors < 3) {
                    printf("  FC2 weight nonzero outside mask at idx %zu: weight=%.6f\n", i, h_fc2_weights[i]);
                }
                weight_errors++;
            }
        }
        if (weight_errors == 0) {
            printf("  PASS: FC2 weights are zero outside the band after update (layer 0)\n");
        } else {
            printf("  FAIL: FC2 has %d weight values nonzero outside mask after update\n", weight_errors);
            allok = 0;
        }

        // =====================================================================
        // TEST 8: Training convergence with banded sparsity
        // =====================================================================
        printf("\n--- Test 8: Training convergence with banded sparsity ---\n");

        float losses[5];
        for (int step = 0; step < 5; step++) {
            gpt2_forward(&model3, x, y, B, T);
            gpt2_zero_grad(&model3);
            gpt2_backward(&model3);
            gpt2_update(&model3, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 2); // step+2 since we already did step 1
            losses[step] = model3.mean_loss;
            printf("  step %d: loss = %f\n", step, losses[step]);
        }

        // Check that loss generally decreases (last < first)
        if (losses[4] < losses[0]) {
            printf("  PASS: loss decreased from %f to %f over 5 steps\n", losses[0], losses[4]);
        } else {
            printf("  FAIL: loss did not decrease (%f -> %f)\n", losses[0], losses[4]);
            allok = 0;
        }

        // Cleanup
        free(x);
        free(y);
        free(h_fc1_grad);
        free(h_fc1_mask);
        free(h_fc2_grad);
        free(h_fc2_mask);
        free(h_fc1_weights);
        free(h_fc2_weights);
        gpt2_free(&model3);
    }

    // =====================================================================
    // TEST 9: Banded forward kernel matches dense+mask (FC1: C->4C)
    // =====================================================================
    printf("\n--- Test 9: Banded forward matches dense+mask (FC1) ---\n");
    {
        int bandwidth = 256;
        int IC = C;      // 768
        int OC = 4 * C;  // 3072
        int BT = 32;     // small batch for testing
        size_t weight_size = (size_t)OC * IC;

        // Allocate device memory
        float* d_inp;  // (BT, IC)
        float* d_weight; // (OC, IC)
        float* d_bias;   // (OC)
        float* d_mask;   // (OC, IC)
        float* d_out_dense; // (BT, OC)
        float* d_out_banded; // (BT, OC)
        cudaCheck(cudaMalloc((void**)&d_inp, BT * IC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_weight, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_bias, OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_mask, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out_dense, BT * OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out_banded, BT * OC * sizeof(float)));

        // Initialize with model weights (layer 0) and some input
        cudaCheck(cudaMemcpy(d_weight, model.params.fcw, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(d_bias, model.params.fcb, OC * sizeof(float), cudaMemcpyDeviceToDevice));
        // Use first BT rows of wte as dummy input (it's BT * 768)
        cudaCheck(cudaMemcpy(d_inp, model.params.wte, BT * IC * sizeof(float), cudaMemcpyDeviceToDevice));

        // Create mask and apply to weight copy for dense path
        int block_size = 256;
        int grid_size = CEIL_DIV(weight_size, block_size);
        create_band_mask_fc1_kernel<<<grid_size, block_size>>>(d_mask, IC, OC, bandwidth);
        cudaCheck(cudaGetLastError());

        // Dense path: apply mask to weights, then dense matmul
        float* d_weight_masked;
        cudaCheck(cudaMalloc((void**)&d_weight_masked, weight_size * sizeof(float)));
        cudaCheck(cudaMemcpy(d_weight_masked, d_weight, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        apply_mask_kernel<<<grid_size, block_size>>>(d_weight_masked, d_mask, weight_size);
        cudaCheck(cudaGetLastError());

        // Run dense matmul with masked weights (BT = B*T, use B=BT, T=1)
        matmul_forward(d_out_dense, d_inp, d_weight_masked, d_bias, BT, 1, IC, OC);
        cudaCheck(cudaDeviceSynchronize());

        // Run banded matmul
        // FC1: bandwidth is in output space, scale to input space
        float half_bw = bandwidth / 2.0f * (float)IC / (float)OC;
        banded_matmul_forward(d_out_banded, d_inp, d_weight, d_bias, BT, 1, IC, OC, half_bw);
        cudaCheck(cudaDeviceSynchronize());

        // Compare on CPU
        size_t out_size = (size_t)BT * OC;
        float* h_dense = (float*)mallocCheck(out_size * sizeof(float));
        float* h_banded = (float*)mallocCheck(out_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_dense, d_out_dense, out_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_banded, d_out_banded, out_size * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        float max_diff = 0.0f;
        for (size_t idx = 0; idx < out_size; idx++) {
            float diff = fabsf(h_dense[idx] - h_banded[idx]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-3f) {
                if (errors < 3) {
                    printf("  MISMATCH at idx %zu: dense=%.6f banded=%.6f diff=%.6f\n",
                           idx, h_dense[idx], h_banded[idx], diff);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("  PASS: FC1 banded forward matches dense+mask (max_diff=%.2e)\n", max_diff);
        } else {
            printf("  FAIL: FC1 banded forward has %d mismatches (max_diff=%.2e)\n", errors, max_diff);
            allok = 0;
        }

        free(h_dense); free(h_banded);
        cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias)); cudaCheck(cudaFree(d_mask));
        cudaCheck(cudaFree(d_out_dense)); cudaCheck(cudaFree(d_out_banded));
        cudaCheck(cudaFree(d_weight_masked));
    }

    // =====================================================================
    // TEST 10: Banded forward kernel matches dense+mask (FC2: 4C->C)
    // =====================================================================
    printf("\n--- Test 10: Banded forward matches dense+mask (FC2) ---\n");
    {
        int bandwidth = 256;
        int IC = 4 * C;  // 3072
        int OC = C;      // 768
        int BT = 32;
        size_t weight_size = (size_t)OC * IC;

        float* d_inp;
        float* d_weight;
        float* d_bias;
        float* d_mask;
        float* d_out_dense;
        float* d_out_banded;
        cudaCheck(cudaMalloc((void**)&d_inp, BT * IC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_weight, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_bias, OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_mask, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out_dense, BT * OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out_banded, BT * OC * sizeof(float)));

        cudaCheck(cudaMemcpy(d_weight, model.params.fcprojw, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(d_bias, model.params.fcprojb, OC * sizeof(float), cudaMemcpyDeviceToDevice));
        // Use first BT rows of fch from model (we need BT * 4C, use fcw transposed as dummy)
        // Actually, just use the first BT*IC floats from wte (it's Vp * C = 50304 * 768, plenty)
        cudaCheck(cudaMemcpy(d_inp, model.params.wte, BT * IC * sizeof(float), cudaMemcpyDeviceToDevice));

        int block_size = 256;
        int grid_size = CEIL_DIV(weight_size, block_size);
        create_band_mask_fc2_kernel<<<grid_size, block_size>>>(d_mask, OC, IC, bandwidth);
        cudaCheck(cudaGetLastError());

        float* d_weight_masked;
        cudaCheck(cudaMalloc((void**)&d_weight_masked, weight_size * sizeof(float)));
        cudaCheck(cudaMemcpy(d_weight_masked, d_weight, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        apply_mask_kernel<<<grid_size, block_size>>>(d_weight_masked, d_mask, weight_size);
        cudaCheck(cudaGetLastError());

        matmul_forward(d_out_dense, d_inp, d_weight_masked, d_bias, BT, 1, IC, OC);
        cudaCheck(cudaDeviceSynchronize());

        // FC2: bandwidth is already in input space
        float half_bw = bandwidth / 2.0f;
        banded_matmul_forward(d_out_banded, d_inp, d_weight, d_bias, BT, 1, IC, OC, half_bw);
        cudaCheck(cudaDeviceSynchronize());

        size_t out_size = (size_t)BT * OC;
        float* h_dense = (float*)mallocCheck(out_size * sizeof(float));
        float* h_banded = (float*)mallocCheck(out_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_dense, d_out_dense, out_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_banded, d_out_banded, out_size * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        float max_diff = 0.0f;
        for (size_t idx = 0; idx < out_size; idx++) {
            float diff = fabsf(h_dense[idx] - h_banded[idx]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-3f) {
                if (errors < 3) {
                    printf("  MISMATCH at idx %zu: dense=%.6f banded=%.6f diff=%.6f\n",
                           idx, h_dense[idx], h_banded[idx], diff);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("  PASS: FC2 banded forward matches dense+mask (max_diff=%.2e)\n", max_diff);
        } else {
            printf("  FAIL: FC2 banded forward has %d mismatches (max_diff=%.2e)\n", errors, max_diff);
            allok = 0;
        }

        free(h_dense); free(h_banded);
        cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias)); cudaCheck(cudaFree(d_mask));
        cudaCheck(cudaFree(d_out_dense)); cudaCheck(cudaFree(d_out_banded));
        cudaCheck(cudaFree(d_weight_masked));
    }

    // =====================================================================
    // TEST 11: Banded backward-to-input matches dense (FC1)
    // =====================================================================
    printf("\n--- Test 11: Banded backward-to-input matches dense (FC1) ---\n");
    {
        int bandwidth = 256;
        int IC = C;      // 768
        int OC = 4 * C;  // 3072
        int BT = 32;
        size_t weight_size = (size_t)OC * IC;

        float* d_dout;
        float* d_weight;
        float* d_mask;
        float* d_dinp_dense;
        float* d_dinp_banded;
        cudaCheck(cudaMalloc((void**)&d_dout, BT * OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_weight, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_mask, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_dinp_dense, BT * IC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_dinp_banded, BT * IC * sizeof(float)));

        // Use model weights and some dout
        cudaCheck(cudaMemcpy(d_weight, model.params.fcw, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(d_dout, model.params.wte, BT * OC * sizeof(float), cudaMemcpyDeviceToDevice));

        // Create mask and apply to weight copy
        int block_size = 256;
        int grid_size = CEIL_DIV(weight_size, block_size);
        create_band_mask_fc1_kernel<<<grid_size, block_size>>>(d_mask, IC, OC, bandwidth);
        cudaCheck(cudaGetLastError());

        float* d_weight_masked;
        cudaCheck(cudaMalloc((void**)&d_weight_masked, weight_size * sizeof(float)));
        cudaCheck(cudaMemcpy(d_weight_masked, d_weight, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        apply_mask_kernel<<<grid_size, block_size>>>(d_weight_masked, d_mask, weight_size);
        cudaCheck(cudaGetLastError());

        // Dense backward-to-input: dinp = weight^T @ dout (using cuBLAS)
        float one = 1.0f, zero = 0.0f;
        cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            IC, BT, OC, &one, d_weight_masked, IC, d_dout, OC, &zero, d_dinp_dense, IC));
        cudaCheck(cudaDeviceSynchronize());

        // Banded backward-to-input (FC1: scale bandwidth to input space)
        float half_bw = bandwidth / 2.0f * (float)IC / (float)OC;
        grid_size = CEIL_DIV(BT * IC, block_size);
        banded_matmul_backward_input_kernel<<<grid_size, block_size>>>(d_dinp_banded, d_dout, d_weight, BT, IC, OC, half_bw);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        size_t dinp_size = (size_t)BT * IC;
        float* h_dense = (float*)mallocCheck(dinp_size * sizeof(float));
        float* h_banded = (float*)mallocCheck(dinp_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_dense, d_dinp_dense, dinp_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_banded, d_dinp_banded, dinp_size * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        float max_diff = 0.0f;
        for (size_t idx = 0; idx < dinp_size; idx++) {
            float diff = fabsf(h_dense[idx] - h_banded[idx]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-3f) {
                if (errors < 3) {
                    printf("  MISMATCH at idx %zu: dense=%.6f banded=%.6f diff=%.6f\n",
                           idx, h_dense[idx], h_banded[idx], diff);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("  PASS: FC1 banded backward-to-input matches dense (max_diff=%.2e)\n", max_diff);
        } else {
            printf("  FAIL: FC1 banded backward-to-input has %d mismatches (max_diff=%.2e)\n", errors, max_diff);
            allok = 0;
        }

        free(h_dense); free(h_banded);
        cudaCheck(cudaFree(d_dout)); cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_mask)); cudaCheck(cudaFree(d_dinp_dense));
        cudaCheck(cudaFree(d_dinp_banded)); cudaCheck(cudaFree(d_weight_masked));
    }

    // =====================================================================
    // TEST 12: Banded backward-to-input matches dense (FC2)
    // =====================================================================
    printf("\n--- Test 12: Banded backward-to-input matches dense (FC2) ---\n");
    {
        int bandwidth = 256;
        int IC = 4 * C;  // 3072
        int OC = C;      // 768
        int BT = 32;
        size_t weight_size = (size_t)OC * IC;

        float* d_dout;
        float* d_weight;
        float* d_mask;
        float* d_dinp_dense;
        float* d_dinp_banded;
        cudaCheck(cudaMalloc((void**)&d_dout, BT * OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_weight, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_mask, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_dinp_dense, BT * IC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_dinp_banded, BT * IC * sizeof(float)));

        cudaCheck(cudaMemcpy(d_weight, model.params.fcprojw, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(d_dout, model.params.wte, BT * OC * sizeof(float), cudaMemcpyDeviceToDevice));

        int block_size = 256;
        int grid_size = CEIL_DIV(weight_size, block_size);
        create_band_mask_fc2_kernel<<<grid_size, block_size>>>(d_mask, OC, IC, bandwidth);
        cudaCheck(cudaGetLastError());

        float* d_weight_masked;
        cudaCheck(cudaMalloc((void**)&d_weight_masked, weight_size * sizeof(float)));
        cudaCheck(cudaMemcpy(d_weight_masked, d_weight, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        apply_mask_kernel<<<grid_size, block_size>>>(d_weight_masked, d_mask, weight_size);
        cudaCheck(cudaGetLastError());

        // Dense backward-to-input
        float one = 1.0f, zero = 0.0f;
        cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            IC, BT, OC, &one, d_weight_masked, IC, d_dout, OC, &zero, d_dinp_dense, IC));
        cudaCheck(cudaDeviceSynchronize());

        // Banded backward-to-input (FC2: bandwidth is already in input space)
        float half_bw = bandwidth / 2.0f;
        grid_size = CEIL_DIV(BT * IC, block_size);
        banded_matmul_backward_input_kernel<<<grid_size, block_size>>>(d_dinp_banded, d_dout, d_weight, BT, IC, OC, half_bw);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());

        size_t dinp_size = (size_t)BT * IC;
        float* h_dense = (float*)mallocCheck(dinp_size * sizeof(float));
        float* h_banded = (float*)mallocCheck(dinp_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_dense, d_dinp_dense, dinp_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_banded, d_dinp_banded, dinp_size * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        float max_diff = 0.0f;
        for (size_t idx = 0; idx < dinp_size; idx++) {
            float diff = fabsf(h_dense[idx] - h_banded[idx]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-3f) {
                if (errors < 3) {
                    printf("  MISMATCH at idx %zu: dense=%.6f banded=%.6f diff=%.6f\n",
                           idx, h_dense[idx], h_banded[idx], diff);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("  PASS: FC2 banded backward-to-input matches dense (max_diff=%.2e)\n", max_diff);
        } else {
            printf("  FAIL: FC2 banded backward-to-input has %d mismatches (max_diff=%.2e)\n", errors, max_diff);
            allok = 0;
        }

        free(h_dense); free(h_banded);
        cudaCheck(cudaFree(d_dout)); cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_mask)); cudaCheck(cudaFree(d_dinp_dense));
        cudaCheck(cudaFree(d_dinp_banded)); cudaCheck(cudaFree(d_weight_masked));
    }

    // =====================================================================
    // TEST 13: Edge case - large half_bw matches dense (FC2)
    // =====================================================================
    printf("\n--- Test 13: Edge case - large half_bw matches dense (FC2) ---\n");
    {
        int IC = 4 * C;  // 3072
        int OC = C;      // 768
        int BT = 32;
        size_t weight_size = (size_t)OC * IC;

        float* d_inp;
        float* d_weight;
        float* d_bias;
        float* d_out_dense;
        float* d_out_banded;
        cudaCheck(cudaMalloc((void**)&d_inp, BT * IC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_weight, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_bias, OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out_dense, BT * OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out_banded, BT * OC * sizeof(float)));

        cudaCheck(cudaMemcpy(d_weight, model.params.fcprojw, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(d_bias, model.params.fcprojb, OC * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(d_inp, model.params.wte, BT * IC * sizeof(float), cudaMemcpyDeviceToDevice));

        matmul_forward(d_out_dense, d_inp, d_weight, d_bias, BT, 1, IC, OC);
        cudaCheck(cudaDeviceSynchronize());

        // Use half_bw = IC to cover all columns (equivalent to dense)
        float half_bw = (float)IC;
        banded_matmul_forward(d_out_banded, d_inp, d_weight, d_bias, BT, 1, IC, OC, half_bw);
        cudaCheck(cudaDeviceSynchronize());

        size_t out_size = (size_t)BT * OC;
        float* h_dense = (float*)mallocCheck(out_size * sizeof(float));
        float* h_banded = (float*)mallocCheck(out_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_dense, d_out_dense, out_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_banded, d_out_banded, out_size * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        float max_diff = 0.0f;
        for (size_t idx = 0; idx < out_size; idx++) {
            float diff = fabsf(h_dense[idx] - h_banded[idx]);
            if (diff > max_diff) max_diff = diff;
            // Tolerance is higher because summation order differs
            float tol = fmaxf(1e-3f, 1e-4f * fabsf(h_dense[idx]));
            if (diff > tol) {
                if (errors < 3) {
                    printf("  MISMATCH at idx %zu: dense=%.6f banded=%.6f diff=%.6f\n",
                           idx, h_dense[idx], h_banded[idx], diff);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("  PASS: large half_bw matches dense (max_diff=%.2e)\n", max_diff);
        } else {
            printf("  FAIL: large half_bw has %d mismatches (max_diff=%.2e)\n", errors, max_diff);
            allok = 0;
        }

        free(h_dense); free(h_banded);
        cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias)); cudaCheck(cudaFree(d_out_dense));
        cudaCheck(cudaFree(d_out_banded));
    }

    // =====================================================================
    // TEST 14: Edge case - bandwidth=1
    // =====================================================================
    printf("\n--- Test 14: Edge case - bandwidth=1 ---\n");
    {
        int IC = C;      // 768
        int OC = 4 * C;  // 3072
        int bandwidth = 1;
        int BT = 32;
        size_t weight_size = (size_t)OC * IC;

        float* d_inp;
        float* d_weight;
        float* d_mask;
        float* d_out_dense;
        float* d_out_banded;
        cudaCheck(cudaMalloc((void**)&d_inp, BT * IC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_weight, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_mask, weight_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out_dense, BT * OC * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&d_out_banded, BT * OC * sizeof(float)));

        cudaCheck(cudaMemcpy(d_weight, model.params.fcw, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(d_inp, model.params.wte, BT * IC * sizeof(float), cudaMemcpyDeviceToDevice));

        int block_size = 256;
        int grid_size = CEIL_DIV(weight_size, block_size);
        create_band_mask_fc1_kernel<<<grid_size, block_size>>>(d_mask, IC, OC, bandwidth);
        cudaCheck(cudaGetLastError());

        float* d_weight_masked;
        cudaCheck(cudaMalloc((void**)&d_weight_masked, weight_size * sizeof(float)));
        cudaCheck(cudaMemcpy(d_weight_masked, d_weight, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        apply_mask_kernel<<<grid_size, block_size>>>(d_weight_masked, d_mask, weight_size);
        cudaCheck(cudaGetLastError());

        // No bias for simplicity
        matmul_forward(d_out_dense, d_inp, d_weight_masked, NULL, BT, 1, IC, OC);
        cudaCheck(cudaDeviceSynchronize());

        // FC1: scale bandwidth to input space
        float half_bw = bandwidth / 2.0f * (float)IC / (float)OC;
        banded_matmul_forward(d_out_banded, d_inp, d_weight, NULL, BT, 1, IC, OC, half_bw);
        cudaCheck(cudaDeviceSynchronize());

        size_t out_size = (size_t)BT * OC;
        float* h_dense = (float*)mallocCheck(out_size * sizeof(float));
        float* h_banded = (float*)mallocCheck(out_size * sizeof(float));
        cudaCheck(cudaMemcpy(h_dense, d_out_dense, out_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_banded, d_out_banded, out_size * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        float max_diff = 0.0f;
        for (size_t idx = 0; idx < out_size; idx++) {
            float diff = fabsf(h_dense[idx] - h_banded[idx]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-5f) {
                if (errors < 3) {
                    printf("  MISMATCH at idx %zu: dense=%.6f banded=%.6f diff=%.6f\n",
                           idx, h_dense[idx], h_banded[idx], diff);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("  PASS: bandwidth=1 works correctly (max_diff=%.2e)\n", max_diff);
        } else {
            printf("  FAIL: bandwidth=1 has %d mismatches (max_diff=%.2e)\n", errors, max_diff);
            allok = 0;
        }

        free(h_dense); free(h_banded);
        cudaCheck(cudaFree(d_inp)); cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_mask)); cudaCheck(cudaFree(d_out_dense));
        cudaCheck(cudaFree(d_out_banded)); cudaCheck(cudaFree(d_weight_masked));
    }

    // =====================================================================
    // TEST 15: End-to-end training equivalence (banded kernels vs dense+mask)
    // =====================================================================
    printf("\n--- Test 15: End-to-end training equivalence ---\n");
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

        // Train 5 steps with banded kernels (current code path)
        BANDWIDTH_FC1 = 256;
        BANDWIDTH_FC2 = 256;

        GPT2 model_banded;
        gpt2_build_from_checkpoint(&model_banded, "gpt2_124M.bin");
        init_fc_masks(&model_banded);

        // Apply initial masks to zero out-of-band weights
        int L = model_banded.config.num_layers;
        int block_size = 256;
        for (int l = 0; l < L; l++) {
            float* layer_fcw = model_banded.params.fcw + l * 4*C * C;
            float* layer_mask = model_banded.fc1_mask + l * 4*C * C;
            int grid_size = CEIL_DIV(4*C * C, block_size);
            apply_mask_kernel<<<grid_size, block_size>>>(layer_fcw, layer_mask, 4*C * C);
        }
        for (int l = 0; l < L; l++) {
            float* layer_fcprojw = model_banded.params.fcprojw + l * C * 4*C;
            float* layer_mask = model_banded.fc2_mask + l * C * 4*C;
            int grid_size = CEIL_DIV(C * 4*C, block_size);
            apply_mask_kernel<<<grid_size, block_size>>>(layer_fcprojw, layer_mask, C * 4*C);
        }
        cudaCheck(cudaDeviceSynchronize());

        float losses_banded[5];
        for (int step = 0; step < 5; step++) {
            gpt2_forward(&model_banded, x, y, B, T);
            gpt2_zero_grad(&model_banded);
            gpt2_backward(&model_banded);
            gpt2_update(&model_banded, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
            losses_banded[step] = model_banded.mean_loss;
            printf("  banded step %d: loss = %f\n", step, losses_banded[step]);
        }

        // Verify losses are reasonable (decreasing)
        int ok = 1;
        if (losses_banded[4] >= losses_banded[0]) {
            printf("  WARNING: banded losses did not decrease\n");
            ok = 0;
        }
        // Check that training completes without NaN
        for (int step = 0; step < 5; step++) {
            if (isnan(losses_banded[step]) || isinf(losses_banded[step])) {
                printf("  FAIL: NaN/Inf loss at step %d\n", step);
                ok = 0;
                break;
            }
        }
        if (ok) {
            printf("  PASS: end-to-end training with banded kernels works correctly\n");
        } else {
            printf("  FAIL: end-to-end training issues detected\n");
            allok = 0;
        }

        free(x); free(y);
        gpt2_free(&model_banded);
        BANDWIDTH_FC1 = 0;
        BANDWIDTH_FC2 = 0;
    }

    // =====================================================================
    // Final result
    // =====================================================================
    printf("\n========================================\n");
    printf("Banded Sparsity Tests: %s\n", allok ? "ALL PASSED" : "SOME FAILED");
    printf("========================================\n");

    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    return allok ? 0 : 1;
}
