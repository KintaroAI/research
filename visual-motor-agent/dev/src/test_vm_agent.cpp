// test_vm_agent.cpp — C++ unit tests for the visual–motor agent.
// Pattern from llm-fundamentals: #define TESTING + #include the implementation to reach internals.
// Gradient correctness against PyTorch lives in tests/test_model_reference.py (float64 reference).
#define TESTING
#include "vm_agent.cpp"

static int g_fail = 0;
#define EXPECT(cond, ...)                                                   \
    do {                                                                    \
        if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail++; } \
    } while (0)

static VMConfig small_config() {
    VMConfig c{};
    c.img = 32; c.c1 = 4; c.c2 = 8; c.c3 = 8; c.feat = 16; c.d_model = 16; c.n_layer = 2; c.n_head = 4;
    c.d_ff = 32; c.ctx = 8; c.pred_img = 16; c.pc1 = 4; c.pc2 = 8; c.pred_hidden = 16; c.use_prev_reward = 1;
    c.logstd_init = -0.5f;
    c.coord = 1;   // exercise the CoordConv / broadcast decoder path (the Python reference covers both)
    c.pb = 4;
    c.pred_residual = 1;
    c.pred_skip = 1;
    c.inv_hidden = 8;
    c.dyn_hidden = 8;   // latent dynamics heads (gradients are checked against PyTorch in tests/)
    return c;
}

static PPOConfig default_ppo() {
    PPOConfig p{};
    p.gamma = 0.99f; p.lam = 0.95f; p.clip_eps = 0.2f; p.vf_coef = 0.5f; p.ent_coef = 0.01f; p.pred_coef = 1.0f;
    p.drive_coef = 0.5f; p.pred_backbone_scale = 1.0f; p.max_grad_norm = 0.5f; p.lr = 3e-4f; p.beta1 = 0.9f;
    p.beta2 = 0.999f; p.adam_eps = 1e-5f; p.weight_decay = 0.f; p.target_kl = 0.f; p.reward_scale = 0.1f;
    p.vclip_eps = 0.2f; p.pred_changed_weight = 0.f; p.inv_coef = 0.5f; p.epochs = 2; p.minibatches = 2; p.use_ppo = 1; p.norm_adv = 1; p.clip_vloss = 0;
    p.dyn_coef = 1.0f; p.dyn_h_coef = 1.0f; p.dyn_x_coef = 1.0f; p.dyn_horizon = 3;
    return p;
}

static void test_gemm() {
    printf("test_gemm\n");
    Rng r;
    rng_seed(r, 1);
    int shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {4, 8, 16}, {5, 9, 3}, {17, 33, 65}, {32, 384, 128}, {1, 130, 64}};
    for (auto& s : shapes) {
        int M = s[0], N = s[1], K = s[2];
        std::vector<float> A(M * K), B(K * N), C(M * N), C0(M * N);
        for (auto& x : A) x = rng_uniform(r) - 0.5f;
        for (auto& x : B) x = rng_uniform(r) - 0.5f;
        for (auto& x : C0) x = rng_uniform(r) - 0.5f;
        for (int acc = 0; acc < 2; acc++) {
            C = C0;
            gemm_nn(M, N, K, A.data(), K, B.data(), N, C.data(), N, acc);
            double maxerr = 0;
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    double ref = acc ? C0[i * N + j] : 0.0;
                    for (int k = 0; k < K; k++) ref += (double)A[i * K + k] * B[k * N + j];
                    maxerr = std::max(maxerr, fabs(ref - C[i * N + j]));
                }
            EXPECT(maxerr < 1e-4, "gemm %dx%dx%d acc=%d maxerr=%g", M, N, K, acc, maxerr);
        }
    }
}

// Fill a rollout by acting on random frames. Returns the model's final peek values.
static std::vector<float> fill_rollout(Model* M, ActCtx* C, Rollout* R, uint64_t seed) {
    const Dims& D = M->D;
    const int N = R->N;
    Rng r;
    rng_seed(r, seed);
    std::vector<uint8_t> img((size_t)N * D.img_px), nimg((size_t)N * D.img_px), es(N, 1), done(N);
    std::vector<float> drv(2 * N), pa(3 * N, 0.f), pr(N, 0.f), act(3 * N), lp(N), val(N), rew(N), nd(2 * N);
    for (auto& x : img) x = (uint8_t)(rng_next(r) & 255);
    for (auto& x : drv) x = rng_uniform(r);
    for (int t = 0; t < R->T; t++) {
        vm_act(M, C, img.data(), drv.data(), pa.data(), pr.data(), es.data(), 0, act.data(), lp.data(), val.data(),
               nullptr, nullptr, R, t);
        for (auto& x : nimg) x = (uint8_t)(rng_next(r) & 255);
        for (int n = 0; n < N; n++) {
            rew[n] = rng_uniform(r) < 0.2f ? 1.f : 0.f;
            done[n] = rng_uniform(r) < 0.1f;
            nd[2 * n] = rng_uniform(r); nd[2 * n + 1] = rng_uniform(r);
        }
        vm_rollout_outcome(R, t, rew.data(), done.data(), nimg.data(), nd.data());
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < 2; k++) pa[3 * n + k] = clampf_(act[3 * n + k], -1.f, 1.f);
            pa[3 * n + 2] = act[3 * n + 2];
            pr[n] = rew[n];
            es[n] = done[n];
        }
        img = nimg;
        drv = nd;
    }
    vm_act(M, C, img.data(), drv.data(), pa.data(), pr.data(), es.data(), VM_ACT_PEEK, act.data(), lp.data(),
           val.data(), nullptr, nullptr, nullptr, 0);
    return val;
}

static void test_act_equals_train_and_ratio_one(int one_frame) {
    printf("test_act_equals_train_and_ratio_one (one_frame=%d)\n", one_frame);
    VMConfig cfg = small_config();
    cfg.one_frame = one_frame;
    Model* M = (Model*)vm_create(&cfg, 42);
    const int N = 5, T = 3 * cfg.ctx;
    ActCtx* C = (ActCtx*)vm_ctx_create(M, N, 7);
    Rollout* R = (Rollout*)vm_rollout_create(M, T, N);
    std::vector<float> lv = fill_rollout(M, C, R, 99);
    // training-path forward over every rollout sequence must reproduce stored logp/value bitwise
    ensure_threads(M);
    Workspace* W = M->ws[0];
    const int nseq = N * (T / cfg.ctx);
    int mism = 0;
    for (int s = 0; s < nseq; s++) {
        SeqView sv = rollout_view(R, s);
        seq_forward(M, W, sv, false);
        for (int t = 0; t < sv.T; t++) {
            PolicyEval e;
            policy_eval(W->pi.data() + 3 * t, M->params.data() + M->off[LOGSTD], sv.action + (size_t)t * sv.act_st, e);
            if (e.logp != sv.old_logp[(size_t)t * sv.sc_st] || W->val[t] != sv.old_value[(size_t)t * sv.sc_st]) mism++;
        }
    }
    EXPECT(mism == 0, "%d tokens differ between acting and training forward", mism);
    // first gradient pass on a fresh rollout: ratio == 1 -> approx KL and clip fraction exactly 0
    PPOConfig pc = default_ppo();
    std::vector<int> ids(nseq);
    for (int i = 0; i < nseq; i++) ids[i] = i;
    PPOStats st;
    vm_debug_grads(M, R, lv.data(), &pc, ids.data(), nseq, &st);
    EXPECT(fabsf(st.approx_kl) < 1e-7f && st.clipfrac == 0.f, "fresh rollout kl=%g clipfrac=%g", st.approx_kl, st.clipfrac);
    vm_rollout_free(R);
    vm_ctx_free(C);
    vm_free(M);
}

static void test_update_deterministic_and_learns() {
    printf("test_update_deterministic_and_learns\n");
    VMConfig cfg = small_config();
    std::vector<float> after[2];
    float pred0 = 0, pred1 = 0;
    for (int rep = 0; rep < 2; rep++) {
        Model* M = (Model*)vm_create(&cfg, 5);
        ActCtx* C = (ActCtx*)vm_ctx_create(M, 4, 3);
        Rollout* R = (Rollout*)vm_rollout_create(M, 2 * cfg.ctx, 4);
        std::vector<float> lv = fill_rollout(M, C, R, 17);
        PPOConfig pc = default_ppo();
        pc.epochs = 30;
        PPOStats st;
        pc.epochs = 1;
        vm_update(M, R, lv.data(), &pc, &st);
        if (rep == 0) pred0 = st.pred_mse;
        pc.epochs = 30;
        vm_update(M, R, lv.data(), &pc, &st);
        if (rep == 0) pred1 = st.pred_mse;
        after[rep] = M->params;
        vm_rollout_free(R);
        vm_ctx_free(C);
        vm_free(M);
    }
    EXPECT(after[0] == after[1], "two identical runs ended with different parameters");
    EXPECT(pred1 < pred0, "prediction loss did not decrease on a fixed rollout (%g -> %g)", pred0, pred1);
}

static void test_checkpoint() {
    printf("test_checkpoint\n");
    VMConfig cfg = small_config();
    Model* M = (Model*)vm_create(&cfg, 11);
    M->step = 7;
    for (long i = 0; i < M->n_params; i++) { M->m[i] = 0.001f * i; M->v[i] = 0.002f * i; }
    const char* path = "/tmp/vm_agent_test_ckpt.bin";
    EXPECT(vm_save(M, path, 1) == 0, "save failed");
    Model* L = (Model*)vm_load(path);
    EXPECT(L != nullptr, "load failed");
    if (L) {
        EXPECT(L->params == M->params && L->m == M->m && L->v == M->v && L->step == 7, "roundtrip mismatch");
        EXPECT(L->wt == M->wt, "transposed weights not refreshed after load");
        vm_free(L);
    }
    remove(path);
    vm_free(M);
}

static void test_peek_does_not_advance() {
    printf("test_peek_does_not_advance\n");
    VMConfig cfg = small_config();
    Model* M = (Model*)vm_create(&cfg, 2);
    ActCtx* C = (ActCtx*)vm_ctx_create(M, 2, 1);
    const Dims& D = M->D;
    std::vector<uint8_t> img(2 * D.img_px, 100), es(2, 0);
    std::vector<float> drv(4, 0.3f), pa(6, 0.f), pr(2, 0.f), act(6), lp(2), val(2), val2(2);
    vm_act(M, C, img.data(), drv.data(), pa.data(), pr.data(), nullptr, 0, act.data(), lp.data(), val.data(), nullptr, nullptr, nullptr, 0);
    long step = C->step;
    int len0 = C->len[0];
    vm_act(M, C, img.data(), drv.data(), pa.data(), pr.data(), nullptr, VM_ACT_PEEK, act.data(), lp.data(), val.data(), nullptr, nullptr, nullptr, 0);
    EXPECT(C->step == step && C->len[0] == len0, "peek advanced the context");
    vm_act(M, C, img.data(), drv.data(), pa.data(), pr.data(), nullptr, 0, act.data(), lp.data(), val2.data(), nullptr, nullptr, nullptr, 0);
    EXPECT(val[0] == val2[0], "committed call after peek differs from the peek");
    vm_ctx_free(C);
    vm_free(M);
}

int main() {
    omp_set_num_threads(4);
    test_gemm();
    test_act_equals_train_and_ratio_one(0);
    test_act_equals_train_and_ratio_one(1);
    test_update_deterministic_and_learns();
    test_checkpoint();
    test_peek_does_not_advance();
    if (g_fail) { printf("%d FAILURE(S)\n", g_fail); return 1; }
    printf("ALL TESTS PASSED\n");
    return 0;
}
