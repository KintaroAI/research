// Test suite for training resume support in train_gpt2_fp32.cu
// Tests: header round-trip, optimizer state round-trip, training equivalence,
//        backward compatibility, and missing .optim sidecar handling.

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
    int enable_tf32 = 0;  // NOTE: disable TF32 for testing!!!
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

    const char* tmp_checkpoint = "/tmp/test_resume_ckpt.bin";
    const char* tmp_optim = "/tmp/test_resume_ckpt.bin.optim";

    printf("\n========================================\n");
    printf("Test Suite: Training Resume\n");
    printf("========================================\n");

    // =====================================================================
    // TEST 1: Header round-trip (step/shard/sample)
    // =====================================================================
    printf("\n--- Test 1: Header round-trip (step/shard/sample) ---\n");
    {
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        // Save with specific resume state
        gpt2_save_checkpoint(&model, tmp_checkpoint, 42, 3, 1000);

        // Re-read the header directly from file
        FILE *f = fopenCheck(tmp_checkpoint, "rb");
        int header[256];
        freadCheck(header, sizeof(int), 256, f);
        fcloseCheck(f);

        // Verify config fields unchanged
        int config_ok = 1;
        if (header[0] != 20240326) { printf("  FAIL: magic=%d\n", header[0]); config_ok = 0; }
        if (header[1] != 3) { printf("  FAIL: version=%d\n", header[1]); config_ok = 0; }
        if (header[2] != model.config.max_seq_len) { printf("  FAIL: max_seq_len mismatch\n"); config_ok = 0; }
        if (header[3] != model.config.vocab_size) { printf("  FAIL: vocab_size mismatch\n"); config_ok = 0; }
        if (header[4] != model.config.num_layers) { printf("  FAIL: num_layers mismatch\n"); config_ok = 0; }
        if (header[5] != model.config.num_heads) { printf("  FAIL: num_heads mismatch\n"); config_ok = 0; }
        if (header[6] != model.config.channels) { printf("  FAIL: channels mismatch\n"); config_ok = 0; }
        if (header[7] != model.config.padded_vocab_size) { printf("  FAIL: padded_vocab_size mismatch\n"); config_ok = 0; }

        if (config_ok) {
            printf("  PASS: config fields [0-7] unchanged\n");
        } else {
            allok = 0;
        }

        // Verify resume fields
        int resume_ok = 1;
        if (header[8] != 42) { printf("  FAIL: step=%d, expected 42\n", header[8]); resume_ok = 0; }
        if (header[9] != 3) { printf("  FAIL: shard=%d, expected 3\n", header[9]); resume_ok = 0; }
        if (header[10] != 1000) { printf("  FAIL: sample=%d, expected 1000\n", header[10]); resume_ok = 0; }

        if (resume_ok) {
            printf("  PASS: resume fields header[8]=42, header[9]=3, header[10]=1000\n");
        } else {
            allok = 0;
        }

        gpt2_free(&model);
        unlink(tmp_checkpoint);
        unlink(tmp_optim);  // save_checkpoint won't write .optim (no optimizer state yet), but clean up just in case
    }

    // =====================================================================
    // TEST 2: Optimizer state round-trip (bit-exact)
    // =====================================================================
    printf("\n--- Test 2: Optimizer state round-trip (bit-exact) ---\n");
    {
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        // Run 3 training steps to populate m/v
        for (int step = 0; step < 3; step++) {
            gpt2_forward(&model, x, y, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);
            gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
        }

        // Save checkpoint (will produce .optim sidecar since m/v are populated)
        gpt2_save_checkpoint(&model, tmp_checkpoint, 3, 0, 0);

        // Copy m/v to CPU for later comparison
        size_t np = model.num_parameters;
        float* m_orig = (float*)mallocCheck(np * sizeof(float));
        float* v_orig = (float*)mallocCheck(np * sizeof(float));
        cudaCheck(cudaMemcpy(m_orig, model.m_memory, np * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(v_orig, model.v_memory, np * sizeof(float), cudaMemcpyDeviceToHost));

        // Load optimizer state into a fresh model copy
        GPT2 model2;
        gpt2_build_from_checkpoint(&model2, tmp_checkpoint);
        gpt2_load_optimizer_state(&model2, tmp_checkpoint);

        // Compare m and v buffers
        float* m_loaded = (float*)mallocCheck(np * sizeof(float));
        float* v_loaded = (float*)mallocCheck(np * sizeof(float));
        cudaCheck(cudaMemcpy(m_loaded, model2.m_memory, np * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(v_loaded, model2.v_memory, np * sizeof(float), cudaMemcpyDeviceToHost));

        int m_mismatches = 0;
        int v_mismatches = 0;
        for (size_t i = 0; i < np; i++) {
            if (m_orig[i] != m_loaded[i]) m_mismatches++;
            if (v_orig[i] != v_loaded[i]) v_mismatches++;
        }

        if (m_mismatches == 0) {
            printf("  PASS: m_memory is bit-exact after round-trip (%zu values)\n", np);
        } else {
            printf("  FAIL: m_memory has %d mismatches out of %zu\n", m_mismatches, np);
            allok = 0;
        }
        if (v_mismatches == 0) {
            printf("  PASS: v_memory is bit-exact after round-trip (%zu values)\n", np);
        } else {
            printf("  FAIL: v_memory has %d mismatches out of %zu\n", v_mismatches, np);
            allok = 0;
        }

        free(m_orig);
        free(v_orig);
        free(m_loaded);
        free(v_loaded);
        gpt2_free(&model);
        gpt2_free(&model2);
        unlink(tmp_checkpoint);
        unlink(tmp_optim);
    }

    // =====================================================================
    // TEST 3: Training equivalence (continuous vs resumed)
    // =====================================================================
    printf("\n--- Test 3: Training equivalence (continuous vs resumed) ---\n");
    {
        int total_steps = 6;
        int save_at = 3;
        float losses_continuous[6];
        float losses_resumed[6];

        // Path A: 6 continuous training steps
        {
            GPT2 model;
            gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
            for (int step = 0; step < total_steps; step++) {
                gpt2_forward(&model, x, y, B, T);
                losses_continuous[step] = model.mean_loss;
                gpt2_zero_grad(&model);
                gpt2_backward(&model);
                gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
            }
            gpt2_free(&model);
        }

        // Path B: 3 steps → save → fresh model → load optimizer → 3 more steps
        {
            // Phase 1: train 3 steps, save
            GPT2 model;
            gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
            for (int step = 0; step < save_at; step++) {
                gpt2_forward(&model, x, y, B, T);
                losses_resumed[step] = model.mean_loss;
                gpt2_zero_grad(&model);
                gpt2_backward(&model);
                gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
            }
            gpt2_save_checkpoint(&model, tmp_checkpoint, save_at, 0, 0);
            gpt2_free(&model);

            // Phase 2: load from checkpoint, resume training
            GPT2 model2;
            gpt2_build_from_checkpoint(&model2, tmp_checkpoint);
            gpt2_load_optimizer_state(&model2, tmp_checkpoint);
            for (int step = save_at; step < total_steps; step++) {
                gpt2_forward(&model2, x, y, B, T);
                losses_resumed[step] = model2.mean_loss;
                gpt2_zero_grad(&model2);
                gpt2_backward(&model2);
                gpt2_update(&model2, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
            }
            gpt2_free(&model2);
        }

        // Compare losses
        int loss_ok = 1;
        for (int step = 0; step < total_steps; step++) {
            float diff = fabsf(losses_continuous[step] - losses_resumed[step]);
            printf("  step %d: continuous=%.6f resumed=%.6f diff=%e\n",
                   step, losses_continuous[step], losses_resumed[step], diff);
            if (diff > 1e-2f) {
                printf("  FAIL: loss diff exceeds 1e-2 at step %d\n", step);
                loss_ok = 0;
            }
        }

        if (loss_ok) {
            printf("  PASS: all %d losses match (tolerance 1e-2)\n", total_steps);
        } else {
            allok = 0;
        }

        unlink(tmp_checkpoint);
        unlink(tmp_optim);
    }

    // =====================================================================
    // TEST 4: Backward compatibility (old checkpoint, no resume state)
    // =====================================================================
    printf("\n--- Test 4: Backward compatibility (old checkpoint) ---\n");
    {
        // gpt2_124M.bin is an original checkpoint with header[8-10] = 0
        FILE *f = fopenCheck("gpt2_124M.bin", "rb");
        int header[256];
        freadCheck(header, sizeof(int), 256, f);
        fcloseCheck(f);

        int compat_ok = 1;
        if (header[8] != 0) { printf("  FAIL: resume_step=%d, expected 0\n", header[8]); compat_ok = 0; }
        if (header[9] != 0) { printf("  FAIL: resume_shard=%d, expected 0\n", header[9]); compat_ok = 0; }
        if (header[10] != 0) { printf("  FAIL: resume_sample=%d, expected 0\n", header[10]); compat_ok = 0; }

        if (compat_ok) {
            printf("  PASS: old checkpoint has header[8-10]=0\n");
        } else {
            allok = 0;
        }

        // Verify load_optimizer_state handles missing .optim gracefully
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        printf("  Calling gpt2_load_optimizer_state on checkpoint without .optim...\n");
        gpt2_load_optimizer_state(&model, "gpt2_124M.bin");

        if (model.m_memory == NULL && model.v_memory == NULL) {
            printf("  PASS: optimizer state remains NULL (graceful fallback)\n");
        } else {
            printf("  FAIL: m_memory or v_memory unexpectedly allocated\n");
            allok = 0;
        }

        gpt2_free(&model);
    }

    // =====================================================================
    // TEST 5: Missing .optim sidecar
    // =====================================================================
    printf("\n--- Test 5: Missing .optim sidecar ---\n");
    {
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        // Run 1 training step so optimizer state is populated
        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, 1);

        // Save checkpoint (creates .optim sidecar)
        gpt2_save_checkpoint(&model, tmp_checkpoint, 1, 0, 0);
        gpt2_free(&model);

        // Delete the .optim file
        unlink(tmp_optim);

        // Load model from checkpoint and try to load optimizer state
        GPT2 model2;
        gpt2_build_from_checkpoint(&model2, tmp_checkpoint);
        printf("  Calling gpt2_load_optimizer_state after deleting .optim...\n");
        gpt2_load_optimizer_state(&model2, tmp_checkpoint);

        if (model2.m_memory == NULL && model2.v_memory == NULL) {
            printf("  PASS: optimizer state remains NULL after missing .optim\n");
        } else {
            printf("  FAIL: m_memory or v_memory unexpectedly allocated\n");
            allok = 0;
        }

        gpt2_free(&model2);
        unlink(tmp_checkpoint);
    }

    // =====================================================================
    // Final result
    // =====================================================================
    printf("\n========================================\n");
    printf("Resume Tests: %s\n", allok ? "ALL PASSED" : "SOME FAILED");
    printf("========================================\n");

    free(x);
    free(y);
    cublasCheck(cublasDestroy(cublas_handle));
    return allok ? 0 : 1;
}
