// Test suite for checkpoint save/load round-trip in train_gpt2_fp32.cu
// Tests: parameter round-trip, forward consistency, and post-training round-trip.

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

    // --- Load debug state for inputs ---
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

    const char* tmp_checkpoint = "/tmp/test_checkpoint_roundtrip.bin";

    printf("\n========================================\n");
    printf("Test Suite: Checkpoint Save/Load\n");
    printf("========================================\n");

    // =====================================================================
    // TEST 1: Round-trip - save and reload, parameters should be bit-exact
    // =====================================================================
    printf("\n--- Test 1: Parameter round-trip (bit-exact) ---\n");
    {
        GPT2 model1;
        gpt2_build_from_checkpoint(&model1, "gpt2_124M.bin");

        // Save to temp file
        gpt2_save_checkpoint(&model1, tmp_checkpoint);

        // Load into second model
        GPT2 model2;
        gpt2_build_from_checkpoint(&model2, tmp_checkpoint);

        // Compare config
        int config_ok = 1;
        if (model1.config.max_seq_len != model2.config.max_seq_len) config_ok = 0;
        if (model1.config.vocab_size != model2.config.vocab_size) config_ok = 0;
        if (model1.config.padded_vocab_size != model2.config.padded_vocab_size) config_ok = 0;
        if (model1.config.num_layers != model2.config.num_layers) config_ok = 0;
        if (model1.config.num_heads != model2.config.num_heads) config_ok = 0;
        if (model1.config.channels != model2.config.channels) config_ok = 0;
        if (model1.num_parameters != model2.num_parameters) config_ok = 0;

        if (config_ok) {
            printf("  PASS: config matches after round-trip\n");
        } else {
            printf("  FAIL: config mismatch after round-trip\n");
            allok = 0;
        }

        // Compare all parameters (copy both to CPU)
        size_t num_params = model1.num_parameters;
        float* params1_cpu = (float*)mallocCheck(num_params * sizeof(float));
        float* params2_cpu = (float*)mallocCheck(num_params * sizeof(float));
        cudaCheck(cudaMemcpy(params1_cpu, model1.params_memory, num_params * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(params2_cpu, model2.params_memory, num_params * sizeof(float), cudaMemcpyDeviceToHost));

        int mismatches = 0;
        for (size_t i = 0; i < num_params && mismatches < 10; i++) {
            if (params1_cpu[i] != params2_cpu[i]) {
                if (mismatches < 3) {
                    printf("  MISMATCH at param[%zu]: %f vs %f\n", i, params1_cpu[i], params2_cpu[i]);
                }
                mismatches++;
            }
        }

        if (mismatches == 0) {
            printf("  PASS: all %zu parameters are bit-exact after round-trip\n", num_params);
        } else {
            printf("  FAIL: %d parameter mismatches\n", mismatches);
            allok = 0;
        }

        free(params1_cpu);
        free(params2_cpu);
        gpt2_free(&model1);
        gpt2_free(&model2);
    }

    // =====================================================================
    // TEST 2: Forward consistency - same logits from original and reloaded
    // =====================================================================
    printf("\n--- Test 2: Forward consistency after round-trip ---\n");
    {
        GPT2 model1;
        gpt2_build_from_checkpoint(&model1, "gpt2_124M.bin");
        gpt2_save_checkpoint(&model1, tmp_checkpoint);

        GPT2 model2;
        gpt2_build_from_checkpoint(&model2, tmp_checkpoint);

        int Vp = model1.config.padded_vocab_size;

        // Forward pass on both
        gpt2_forward(&model1, x, NULL, B, T);
        gpt2_forward(&model2, x, NULL, B, T);

        // Compare logits
        size_t logits_size = (size_t)B * T * Vp;
        float* logits1_cpu = (float*)mallocCheck(logits_size * sizeof(float));
        float* logits2_cpu = (float*)mallocCheck(logits_size * sizeof(float));
        cudaCheck(cudaMemcpy(logits1_cpu, model1.acts.output, logits_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(logits2_cpu, model2.acts.output, logits_size * sizeof(float), cudaMemcpyDeviceToHost));

        int mismatches = 0;
        float max_diff = 0.0f;
        for (size_t i = 0; i < logits_size && mismatches < 10; i++) {
            float diff = fabsf(logits1_cpu[i] - logits2_cpu[i]);
            max_diff = fmaxf(max_diff, diff);
            if (diff > 1e-5f) {
                if (mismatches < 3) {
                    printf("  LOGIT MISMATCH at [%zu]: %f vs %f (diff=%e)\n",
                           i, logits1_cpu[i], logits2_cpu[i], diff);
                }
                mismatches++;
            }
        }

        if (mismatches == 0) {
            printf("  PASS: logits match exactly (max_diff=%e)\n", max_diff);
        } else {
            printf("  FAIL: %d logit mismatches (max_diff=%e)\n", mismatches, max_diff);
            allok = 0;
        }

        free(logits1_cpu);
        free(logits2_cpu);
        gpt2_free(&model1);
        gpt2_free(&model2);
    }

    // =====================================================================
    // TEST 3: Post-training round-trip
    // =====================================================================
    printf("\n--- Test 3: Post-training round-trip ---\n");
    {
        GPT2 model1;
        gpt2_build_from_checkpoint(&model1, "gpt2_124M.bin");

        // Do 3 training steps
        for (int step = 0; step < 3; step++) {
            gpt2_forward(&model1, x, y, B, T);
            gpt2_zero_grad(&model1);
            gpt2_backward(&model1);
            gpt2_update(&model1, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
        }
        float loss_before_save = model1.mean_loss;
        printf("  Loss after 3 training steps: %f\n", loss_before_save);

        // Save trained model
        gpt2_save_checkpoint(&model1, tmp_checkpoint);

        // Load into new model
        GPT2 model2;
        gpt2_build_from_checkpoint(&model2, tmp_checkpoint);

        // Compare parameters
        size_t num_params = model1.num_parameters;
        float* params1_cpu = (float*)mallocCheck(num_params * sizeof(float));
        float* params2_cpu = (float*)mallocCheck(num_params * sizeof(float));
        cudaCheck(cudaMemcpy(params1_cpu, model1.params_memory, num_params * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(params2_cpu, model2.params_memory, num_params * sizeof(float), cudaMemcpyDeviceToHost));

        int param_mismatches = 0;
        for (size_t i = 0; i < num_params; i++) {
            if (params1_cpu[i] != params2_cpu[i]) param_mismatches++;
        }

        if (param_mismatches == 0) {
            printf("  PASS: trained parameters are bit-exact after round-trip\n");
        } else {
            printf("  FAIL: %d parameter mismatches after training+round-trip\n", param_mismatches);
            allok = 0;
        }

        // Forward on reloaded model, check logits match
        int Vp = model1.config.padded_vocab_size;
        gpt2_forward(&model1, x, NULL, B, T);
        gpt2_forward(&model2, x, NULL, B, T);

        size_t logits_size = (size_t)B * T * Vp;
        float* logits1_cpu = (float*)mallocCheck(logits_size * sizeof(float));
        float* logits2_cpu = (float*)mallocCheck(logits_size * sizeof(float));
        cudaCheck(cudaMemcpy(logits1_cpu, model1.acts.output, logits_size * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(logits2_cpu, model2.acts.output, logits_size * sizeof(float), cudaMemcpyDeviceToHost));

        int logit_mismatches = 0;
        float max_diff = 0.0f;
        for (size_t i = 0; i < logits_size; i++) {
            float diff = fabsf(logits1_cpu[i] - logits2_cpu[i]);
            max_diff = fmaxf(max_diff, diff);
            if (diff > 1e-5f) logit_mismatches++;
        }

        if (logit_mismatches == 0) {
            printf("  PASS: post-training logits match exactly (max_diff=%e)\n", max_diff);
        } else {
            printf("  FAIL: %d logit mismatches post-training (max_diff=%e)\n", logit_mismatches, max_diff);
            allok = 0;
        }

        free(params1_cpu);
        free(params2_cpu);
        free(logits1_cpu);
        free(logits2_cpu);
        gpt2_free(&model1);
        gpt2_free(&model2);
    }

    // Cleanup temp file
    unlink(tmp_checkpoint);

    // =====================================================================
    // Final result
    // =====================================================================
    printf("\n========================================\n");
    printf("Checkpoint Tests: %s\n", allok ? "ALL PASSED" : "SOME FAILED");
    printf("========================================\n");

    free(x);
    free(y);
    cublasCheck(cublasDestroy(cublas_handle));
    return allok ? 0 : 1;
}
