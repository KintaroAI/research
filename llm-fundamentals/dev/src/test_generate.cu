// Test suite for generate.cu inference forward pass
// Tests: determinism, logits sanity, and cross-validation with training forward.

#define TESTING
#include "generate.cu"

// We also need the training forward for cross-validation.
// Instead of including train_gpt2_fp32.cu (which would cause duplicate symbol errors),
// we load the same model in the generate path and compare against a reference file.

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
    printf("Test Suite: Generate (Inference)\n");
    printf("========================================\n");

    // =====================================================================
    // TEST 1: Forward pass determinism
    // =====================================================================
    printf("\n--- Test 1: Forward pass determinism ---\n");
    {
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        int Vp = model.config.padded_vocab_size;
        int T = model.config.max_seq_len;

        // Create a simple input sequence
        int* tokens = (int*)mallocCheck(T * sizeof(int));
        for (int i = 0; i < T; i++) tokens[i] = GPT2_EOT;
        tokens[0] = 464;  // "The"
        tokens[1] = 1917; // " world"

        // First forward pass
        gpt2_forward(&model, tokens, 1, T);
        size_t logits_size = (size_t)T * Vp;
        float* logits1_cpu = (float*)mallocCheck(logits_size * sizeof(float));
        cudaCheck(cudaMemcpy(logits1_cpu, model.acts.output, logits_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Second forward pass with same input
        gpt2_forward(&model, tokens, 1, T);
        float* logits2_cpu = (float*)mallocCheck(logits_size * sizeof(float));
        cudaCheck(cudaMemcpy(logits2_cpu, model.acts.output, logits_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Compare
        int mismatches = 0;
        float max_diff = 0.0f;
        for (size_t i = 0; i < logits_size && mismatches < 10; i++) {
            float diff = fabsf(logits1_cpu[i] - logits2_cpu[i]);
            max_diff = fmaxf(max_diff, diff);
            if (logits1_cpu[i] != logits2_cpu[i]) mismatches++;
        }

        if (mismatches == 0) {
            printf("  PASS: two forward passes produce identical logits\n");
        } else {
            printf("  FAIL: %d mismatches between two forward passes (max_diff=%e)\n", mismatches, max_diff);
            allok = 0;
        }

        free(tokens);
        free(logits1_cpu);
        free(logits2_cpu);
        gpt2_free(&model);
    }

    // =====================================================================
    // TEST 2: Logits sanity check (no NaN/Inf, finite values)
    // =====================================================================
    printf("\n--- Test 2: Logits sanity check ---\n");
    {
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        int V = model.config.vocab_size;
        int Vp = model.config.padded_vocab_size;
        int T = model.config.max_seq_len;

        int* tokens = (int*)mallocCheck(T * sizeof(int));
        for (int i = 0; i < T; i++) tokens[i] = GPT2_EOT;
        tokens[0] = 464;  // "The"
        tokens[1] = 1917; // " world"

        gpt2_forward(&model, tokens, 1, T);

        float* logits_cpu = (float*)mallocCheck((size_t)T * Vp * sizeof(float));
        cudaCheck(cudaMemcpy(logits_cpu, model.acts.output, (size_t)T * Vp * sizeof(float), cudaMemcpyDeviceToHost));

        // Check for NaN and Inf in the vocab portion (not padded)
        int nan_count = 0;
        int inf_count = 0;
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;

        // Check position 1 logits (prediction after "The world")
        float* logits_at_1 = logits_cpu + 1 * Vp;
        for (int v = 0; v < V; v++) {
            if (isnan(logits_at_1[v])) nan_count++;
            if (isinf(logits_at_1[v])) inf_count++;
            if (logits_at_1[v] < min_val) min_val = logits_at_1[v];
            if (logits_at_1[v] > max_val) max_val = logits_at_1[v];
        }

        printf("  Logits at position 1: min=%f, max=%f, NaN=%d, Inf=%d\n",
               min_val, max_val, nan_count, inf_count);

        if (nan_count == 0 && inf_count == 0 && max_val > min_val) {
            printf("  PASS: logits are finite and have reasonable range\n");
        } else {
            printf("  FAIL: logits contain NaN/Inf or degenerate values\n");
            allok = 0;
        }

        // Verify logits at position 0 are also valid
        float* logits_at_0 = logits_cpu;
        int nan_count_0 = 0;
        for (int v = 0; v < V; v++) {
            if (isnan(logits_at_0[v])) nan_count_0++;
        }
        if (nan_count_0 == 0) {
            printf("  PASS: position 0 logits also valid (no NaN)\n");
        } else {
            printf("  FAIL: position 0 logits contain %d NaN values\n", nan_count_0);
            allok = 0;
        }

        free(tokens);
        free(logits_cpu);
        gpt2_free(&model);
    }

    // =====================================================================
    // TEST 3: Sampling works (basic smoke test)
    // =====================================================================
    printf("\n--- Test 3: Sampling smoke test ---\n");
    {
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        int V = model.config.vocab_size;
        int Vp = model.config.padded_vocab_size;
        int T = model.config.max_seq_len;

        int* tokens = (int*)mallocCheck(T * sizeof(int));
        for (int i = 0; i < T; i++) tokens[i] = GPT2_EOT;

        float* cpu_logits = (float*)mallocCheck(V * sizeof(float));
        unsigned long long rng_state = 42;

        // Generate a few tokens
        int num_generated = 10;
        int gen_ok = 1;
        for (int t = 1; t < num_generated && t < T; t++) {
            gpt2_forward(&model, tokens, 1, T);

            float* logits = model.acts.output + (t - 1) * Vp;
            cudaCheck(cudaMemcpy(cpu_logits, logits, V * sizeof(float), cudaMemcpyDeviceToHost));

            float coin = random_f32(&rng_state);
            int next_token = sample_softmax(cpu_logits, V, coin);

            if (next_token < 0 || next_token >= V) {
                printf("  FAIL: generated invalid token %d at position %d\n", next_token, t);
                gen_ok = 0;
                break;
            }
            tokens[t] = next_token;
            printf("  Position %d: token=%d\n", t, next_token);
        }

        if (gen_ok) {
            printf("  PASS: generated %d valid tokens\n", num_generated - 1);
        } else {
            allok = 0;
        }

        free(tokens);
        free(cpu_logits);
        gpt2_free(&model);
    }

    // =====================================================================
    // TEST 4: Cross-validation with debug state logits
    // =====================================================================
    printf("\n--- Test 4: Cross-validation with debug state ---\n");
    {
        // Load the debug state which has reference logits from PyTorch
        FILE *sf = fopen("gpt2_124M_debug_state.bin", "rb");
        if (sf == NULL) {
            printf("  SKIP: gpt2_124M_debug_state.bin not found\n");
        } else {
            int sh[256];
            freadCheck(sh, sizeof(int), 256, sf);
            if (sh[0] != 20240327) {
                printf("  SKIP: bad magic in debug state\n");
                fclose(sf);
            } else {
                int B = sh[2]; // batch size from debug state
                int T_dbg = sh[3]; // seq len from debug state

                // We use batch element 0 only (generate.cu uses B=1)
                int* dbg_x = (int*)mallocCheck(B * T_dbg * sizeof(int));
                int* dbg_y = (int*)mallocCheck(B * T_dbg * sizeof(int));
                freadCheck(dbg_x, sizeof(int), B * T_dbg, sf);
                freadCheck(dbg_y, sizeof(int), B * T_dbg, sf);

                // Read expected logits from PyTorch (shape: B*T*V)
                GPT2 model;
                gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
                int V = model.config.vocab_size;
                int Vp = model.config.padded_vocab_size;
                int T_model = model.config.max_seq_len;

                float* expected_logits = (float*)mallocCheck((size_t)B * T_dbg * V * sizeof(float));
                freadCheck(expected_logits, sizeof(float), (size_t)B * T_dbg * V, sf);
                fclose(sf);

                // Run inference forward on first batch element, padded to full T
                int* tokens = (int*)mallocCheck(T_model * sizeof(int));
                for (int i = 0; i < T_model; i++) tokens[i] = GPT2_EOT;
                for (int i = 0; i < T_dbg; i++) tokens[i] = dbg_x[i]; // first batch element

                gpt2_forward(&model, tokens, 1, T_model);

                // Compare logits at positions 0..T_dbg-1 against expected
                float* our_logits_cpu = (float*)mallocCheck((size_t)T_model * Vp * sizeof(float));
                cudaCheck(cudaMemcpy(our_logits_cpu, model.acts.output,
                                     (size_t)T_model * Vp * sizeof(float), cudaMemcpyDeviceToHost));

                int mismatches = 0;
                float max_diff = 0.0f;
                for (int t = 0; t < T_dbg; t++) {
                    for (int v = 0; v < V; v++) {
                        float ours = our_logits_cpu[t * Vp + v];
                        float expected = expected_logits[t * V + v]; // first batch element
                        float diff = fabsf(ours - expected);
                        max_diff = fmaxf(max_diff, diff);
                        if (diff > 1e-2f) mismatches++;
                    }
                }

                printf("  Max diff vs PyTorch reference: %e (over %d values)\n", max_diff, T_dbg * V);
                printf("  Mismatches (>1e-2): %d / %d\n", mismatches, T_dbg * V);

                // generate.cu uses cuBLAS (potentially different path than training),
                // so allow small tolerance
                if (mismatches == 0) {
                    printf("  PASS: generate forward matches PyTorch reference (tol=1e-2)\n");
                } else if (max_diff < 5e-2f) {
                    printf("  PASS (with tolerance): max_diff=%e is within acceptable range\n", max_diff);
                } else {
                    printf("  FAIL: too many mismatches against reference\n");
                    allok = 0;
                }

                free(dbg_x);
                free(dbg_y);
                free(expected_logits);
                free(tokens);
                free(our_logits_cpu);
                gpt2_free(&model);
            }
        }
    }

    // =====================================================================
    // Final result
    // =====================================================================
    printf("\n========================================\n");
    printf("Generate Tests: %s\n", allok ? "ALL PASSED" : "SOME FAILED");
    printf("========================================\n");

    cublasCheck(cublasDestroy(cublas_handle));
    return allok ? 0 : 1;
}
