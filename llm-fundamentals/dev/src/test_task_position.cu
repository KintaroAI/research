// Test suite for task_position loss masking in train_gpt2_fp32.cu
// Tests: default (all positions), specific position loss, and dlosses_mask correctness.

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
    int enable_tf32 = 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    int allok = 1;

    // --- Load debug state ---
    FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { printf("Bad magic state file\n"); exit(EXIT_FAILURE); }
    int B = state_header[2];
    int T = state_header[3];
    printf("[State] B=%d, T=%d\n", B, T);

    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    freadCheck(x, sizeof(int), B*T, state_file);
    freadCheck(y, sizeof(int), B*T, state_file);
    fcloseCheck(state_file);

    printf("\n========================================\n");
    printf("Test Suite: Task Position Loss Masking\n");
    printf("========================================\n");

    // =====================================================================
    // TEST 1: Default task_position=-1 (all positions contribute to loss)
    // =====================================================================
    printf("\n--- Test 1: Default task_position=-1 ---\n");
    float default_loss;
    {
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        // task_position is -1 by default (set in gpt2_build_from_checkpoint)

        gpt2_forward(&model, x, y, B, T);
        default_loss = model.mean_loss;
        printf("  Default loss (all positions): %f\n", default_loss);

        if (default_loss > 0.0f && !isnan(default_loss) && !isinf(default_loss)) {
            printf("  PASS: default loss is valid\n");
        } else {
            printf("  FAIL: default loss is invalid\n");
            allok = 0;
        }

        // Verify: cpu_losses should have a value at every position
        int nonzero_count = 0;
        for (int i = 0; i < B * T; i++) {
            if (model.cpu_losses[i] > 0.0f) nonzero_count++;
        }
        printf("  Non-zero loss positions: %d / %d\n", nonzero_count, B * T);
        if (nonzero_count == B * T) {
            printf("  PASS: all positions contribute to loss\n");
        } else {
            printf("  FAIL: expected all %d positions to have loss\n", B * T);
            allok = 0;
        }

        // Verify dlosses_mask is NULL (not allocated in default mode)
        if (model.dlosses_mask == NULL) {
            printf("  PASS: dlosses_mask is NULL in default mode\n");
        } else {
            printf("  FAIL: dlosses_mask should be NULL when task_position=-1\n");
            allok = 0;
        }

        gpt2_free(&model);
    }

    // =====================================================================
    // TEST 2: task_position=N (only position N contributes to loss)
    // =====================================================================
    printf("\n--- Test 2: task_position=N ---\n");
    {
        int tp = T / 2;  // pick a position in the middle
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        model.task_position = tp;

        gpt2_forward(&model, x, y, B, T);
        float masked_loss = model.mean_loss;
        printf("  Masked loss (position %d only): %f\n", tp, masked_loss);

        if (masked_loss > 0.0f && !isnan(masked_loss) && !isinf(masked_loss)) {
            printf("  PASS: masked loss is valid\n");
        } else {
            printf("  FAIL: masked loss is invalid\n");
            allok = 0;
        }

        // Verify mean_loss is average of cpu_losses at task_position across batch
        float manual_loss = 0.0f;
        for (int b = 0; b < B; b++) {
            manual_loss += model.cpu_losses[b * T + tp];
        }
        manual_loss /= B;
        float loss_diff = fabsf(masked_loss - manual_loss);
        printf("  Manual loss from cpu_losses: %f (diff: %e)\n", manual_loss, loss_diff);
        if (loss_diff < 1e-5f) {
            printf("  PASS: mean_loss matches manual calculation from cpu_losses\n");
        } else {
            printf("  FAIL: mean_loss does not match (diff=%e)\n", loss_diff);
            allok = 0;
        }

        // Loss with masking should differ from default (very unlikely to be equal)
        if (fabsf(masked_loss - default_loss) > 1e-3f) {
            printf("  PASS: masked loss (%f) differs from default loss (%f)\n", masked_loss, default_loss);
        } else {
            printf("  WARN: masked loss suspiciously close to default loss\n");
        }

        gpt2_free(&model);
    }

    // =====================================================================
    // TEST 3: dlosses_mask correctness
    // =====================================================================
    printf("\n--- Test 3: dlosses_mask correctness ---\n");
    {
        int tp = 10;
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        model.task_position = tp;

        // Forward triggers lazy allocation of dlosses_mask
        gpt2_forward(&model, x, y, B, T);

        if (model.dlosses_mask == NULL) {
            printf("  FAIL: dlosses_mask should be allocated after forward with task_position=%d\n", tp);
            allok = 0;
        } else {
            // Copy mask to CPU
            float* h_mask = (float*)mallocCheck(B * T * sizeof(float));
            cudaCheck(cudaMemcpy(h_mask, model.dlosses_mask, B * T * sizeof(float), cudaMemcpyDeviceToHost));

            int errors = 0;
            float expected_val = 1.0f / B;
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    float actual = h_mask[b * T + t];
                    if (t == tp) {
                        // Should be 1/B
                        if (fabsf(actual - expected_val) > 1e-6f) {
                            printf("  MISMATCH at (b=%d, t=%d): expected %f, got %f\n", b, t, expected_val, actual);
                            errors++;
                        }
                    } else {
                        // Should be 0
                        if (actual != 0.0f) {
                            printf("  MISMATCH at (b=%d, t=%d): expected 0.0, got %f\n", b, t, actual);
                            errors++;
                        }
                    }
                    if (errors >= 10) break;
                }
                if (errors >= 10) break;
            }

            if (errors == 0) {
                printf("  PASS: dlosses_mask has 1/%d=%f at position %d and 0 elsewhere\n",
                       B, expected_val, tp);
            } else {
                printf("  FAIL: dlosses_mask has %d errors\n", errors);
                allok = 0;
            }

            free(h_mask);
        }

        gpt2_free(&model);
    }

    // =====================================================================
    // TEST 4: Different task positions give different losses
    // =====================================================================
    printf("\n--- Test 4: Different positions give different losses ---\n");
    {
        float losses[3];
        int positions[3] = {0, T/2, T-1};

        for (int p = 0; p < 3; p++) {
            GPT2 model;
            gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
            model.task_position = positions[p];
            gpt2_forward(&model, x, y, B, T);
            losses[p] = model.mean_loss;
            printf("  Position %d: loss = %f\n", positions[p], losses[p]);
            gpt2_free(&model);
        }

        // At least two of three positions should give different losses
        int differ_count = 0;
        if (fabsf(losses[0] - losses[1]) > 1e-3f) differ_count++;
        if (fabsf(losses[1] - losses[2]) > 1e-3f) differ_count++;
        if (fabsf(losses[0] - losses[2]) > 1e-3f) differ_count++;

        if (differ_count >= 2) {
            printf("  PASS: different positions produce different losses\n");
        } else {
            printf("  FAIL: losses are suspiciously similar across positions\n");
            allok = 0;
        }
    }

    // =====================================================================
    // TEST 5: Training with task_position converges
    // =====================================================================
    printf("\n--- Test 5: Training convergence with task_position ---\n");
    {
        int tp = T / 2;
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        model.task_position = tp;

        float losses[5];
        for (int step = 0; step < 5; step++) {
            gpt2_forward(&model, x, y, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);
            gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
            losses[step] = model.mean_loss;
            printf("  step %d: loss = %f\n", step, losses[step]);
        }

        if (losses[4] < losses[0]) {
            printf("  PASS: loss decreased from %f to %f\n", losses[0], losses[4]);
        } else {
            printf("  FAIL: loss did not decrease\n");
            allok = 0;
        }

        gpt2_free(&model);
    }

    // =====================================================================
    // Final result
    // =====================================================================
    printf("\n========================================\n");
    printf("Task Position Tests: %s\n", allok ? "ALL PASSED" : "SOME FAILED");
    printf("========================================\n");

    free(x);
    free(y);
    cublasCheck(cublasDestroy(cublas_handle));
    return allok ? 0 : 1;
}
