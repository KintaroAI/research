// bench_vm_agent.cpp — throughput of acting and PPO updates for the default model (random frames).
// Usage: ./bench_vm_agent [n_envs=128] [rollout_T=128] [pred=1] [coord=0]
#define TESTING
#include "vm_agent.cpp"
#include <chrono>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 128;
    const int T = argc > 2 ? atoi(argv[2]) : 128;
    const int pred = argc > 3 ? atoi(argv[3]) : 1;
    const int coord = argc > 4 ? atoi(argv[4]) : 0;
    VMConfig c;
    c.img = 64; c.c1 = 16; c.c2 = 32; c.c3 = 32; c.feat = 128; c.d_model = 128; c.n_layer = 2; c.n_head = 4;
    c.d_ff = 512; c.ctx = 32; c.pred_img = 32; c.pc1 = 16; c.pc2 = 32; c.pred_hidden = 256; c.use_prev_reward = 0;
    c.logstd_init = -0.5f;
    c.coord = coord;
    c.pb = 32;
    c.pred_residual = 0;
    c.pred_skip = 0;
    c.inv_hidden = 0;
    Model* M = (Model*)vm_create(&c, 1);
    printf("params: %ld  threads: %d  envs: %d  rollout: %d  pred: %d  coord: %d\n", M->n_params, omp_get_max_threads(), N, T, pred, coord);
    ActCtx* C = (ActCtx*)vm_ctx_create(M, N, 2);
    Rollout* R = (Rollout*)vm_rollout_create(M, T, N);
    const Dims& D = M->D;
    Rng r;
    rng_seed(r, 3);
    std::vector<uint8_t> img((size_t)N * D.img_px), es(N, 0), done(N, 0);
    std::vector<float> drv(2 * N, 0.2f), pa(3 * N, 0.f), pr(N, 0.f), act(3 * N), lp(N), val(N), rew(N, 0.f), nd(2 * N, 0.2f);
    for (auto& x : img) x = (uint8_t)(rng_next(r) & 255);
    double t0 = now_s();
    for (int t = 0; t < T; t++) {
        vm_act(M, C, img.data(), drv.data(), pa.data(), pr.data(), es.data(), 0, act.data(), lp.data(), val.data(),
               nullptr, nullptr, R, t);
        vm_rollout_outcome(R, t, rew.data(), done.data(), img.data(), nd.data());
    }
    double t_act = now_s() - t0;
    printf("act:    %.1f ms/step for %d envs  (%.0f env-steps/s)\n", 1e3 * t_act / T, N, N * T / t_act);
    PPOConfig p;
    p.gamma = 0.99f; p.lam = 0.95f; p.clip_eps = 0.2f; p.vf_coef = 0.5f; p.ent_coef = 0.01f;
    p.pred_coef = pred ? 1.0f : 0.f; p.drive_coef = pred ? 0.5f : 0.f; p.pred_backbone_scale = 1.0f;
    p.max_grad_norm = 0.5f; p.lr = 2.5e-4f; p.beta1 = 0.9f; p.beta2 = 0.999f; p.adam_eps = 1e-5f;
    p.weight_decay = 0.f; p.target_kl = 0.f; p.reward_scale = 0.1f; p.vclip_eps = 0.2f; p.pred_changed_weight = 0.f; p.inv_coef = 0.f;
    p.epochs = 1; p.minibatches = 4; p.use_ppo = 1; p.norm_adv = 1; p.clip_vloss = 0;
    PPOStats st;
    vm_update(M, R, val.data(), &p, &st);  // warm-up (allocations)
    t0 = now_s();
    vm_update(M, R, val.data(), &p, &st);
    double t_upd = now_s() - t0;
    printf("update: %.2f s per epoch over %d transitions (%.0f transitions/s/epoch, %.0f ms/minibatch)\n", t_upd,
           N * T, N * T / t_upd, 1e3 * t_upd / p.minibatches);
    printf("=> with 4 epochs: ~%.0f env-steps/s end-to-end (excluding the Python world)\n",
           N * T / (t_act + 4 * t_upd));
    return 0;
}
