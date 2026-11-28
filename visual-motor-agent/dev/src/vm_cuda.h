// vm_cuda.h — the basic agent's update on the GPU (task 00011).
//
// The same model and parameter layout as vm_agent.cpp (CNN model: CoordConv encoder, causal Transformer, policy
// head, prediction MLP with the U-Net-skip frame decoder and the drives head), for the prediction-only update of the
// basic agent: next frame, next drives and next move, gradient-norm clip and AdamW. All of an update runs on the
// device without host synchronization. Inverse dynamics (task 00013) and the planner agent's latent dynamics are
// supported. Refuses PPO, patch tokens, and models without coord, pred_residual and pred_skip.
#pragma once
#include <stdint.h>
#include "vm_agent.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float loss;          // total loss per token (as vm_debug_grads returns)
    float pred_mse, drive_mse, move_nll, click_p;
    float pred_changed_mse;   // frame MSE over the pixels that change between consecutive frames
    float grad_norm, gn_backbone, gn_policy, gn_pred;   // pre-clip gradient norms (after an update)
    float inv_mse, inv_r2;    // inverse dynamics: MSE of (dx, dy) and R^2 against predicting 0 (as vm_agent)
    float dyn_mse, dyn_hunger_mse, dyn_recon_mse;   // latent dynamics (task 00014): latent, hunger, decoder MSE
    float act_drive_mse, act_frame_mse;   // act token (task 00020): its next-drives and coarse-next-frame MSE
    // universal model (task 00023): cross-entropy and expected-value squared error by predicted kind
    float uni_ce_body, uni_ce_patch, uni_ce_move, uni_mse_body, uni_mse_patch, uni_mse_move;
} VMCStats;

// Creates the model from a config and the flat parameters (n_params floats, the layout of vm_param_layout: offsets
// and sizes of every parameter tensor in ParamId order, n_tensors of them). NULL if the config is not supported or
// the layout differs from the one expected.
void* vmc_create(const VMConfig* cfg, const float* params, long n_params, const long* offsets, const long* sizes,
                 int n_tensors);
void vmc_free(void* h);
// The current device's free memory in bytes (cudaMemGetInfo), or -1 (for tests: models must free all they allocate)
long vmc_device_free(void);
// One batch of S sequences of ctx steps, [S, ctx, ...] C-contiguous host arrays (images u8 HWC, next_small u8 HWC
// at pred_img, or the full-size next frames if next_full: then downsampled on the device). prev_reward is the
// command slot. Copied to the device.
int vmc_set_batch(void* h, int S, const uint8_t* images, const float* drives, const float* prev_action,
                  const float* prev_reward, const uint8_t* ep_start, const float* action, const uint8_t* next_small,
                  const float* next_drives, int next_full);
// The universal model (task 00039, self-forcing): the token inputs' frames and bodies of the current batch (S sequences
// as in vmc_set_batch), used instead of the batch's own; the targets stay the batch's. Cleared by the next batch.
int vmc_set_uni_inputs(void* h, int S, const uint8_t* images, const float* drives);
// Task 00089: which steps of the uploaded batch this transformer saw, mask [S, ctx] (1 seen, 0 missed): a missed step is no
// key for the seen steps and has no loss. Cleared by the next batch.
int vmc_set_step_mask(void* h, int S, const uint8_t* mask);
// Task 00128: the universal model's move token as a policy-gradient actor: the move's loss (the token before the move)
// of each frame of the uploaded batch times w [S, ctx], plus ent times the gradient of minus its entropy (sum over the
// used move values, at the cross-entropy's scale). Cleared by the next batch.
int vmc_set_move_weight(void* h, int S, const float* w, float ent);
// Loss and gradients of the current batch (gradients stay on the device; vmc_grads copies them out).
int vmc_loss_and_grads(void* h, const PPOConfig* pc);
// loss_and_grads, the gradient-norm clip and an AdamW step.
int vmc_update(void* h, const PPOConfig* pc);
// The forward pass of S sequences of T <= ctx steps ([S, T, ...] inputs; ep_start may be NULL), with the outputs of
// vm_forward ([S, T, ...]; any may be NULL): z, value, mu = tanh of the mean move, click probability, and with
// pred_actions the predicted next frames (HWC at pred_img) and drives. Waits for the device.
int vmc_forward(void* h, int S, int T, const uint8_t* images, const float* drives, const float* prev_action,
                const float* prev_reward, const uint8_t* ep_start, const float* pred_actions, float* z, float* value,
                float* mu, float* click_p, float* pred_frames, float* pred_drives);
// The universal model (universal = 1), as vm_forward_uni: values [S, T*ntok, V], move_probs [S, T, 3*bins] (either NULL).
int vmc_forward_uni(void* h, int S, int T, const uint8_t* images, const float* drives, const float* actions,
                    const uint8_t* ep_start, float* values, float* move_probs, float* next_body);
// The instinct's candidates (universal model), right after a vmc_forward_uni of S windows: for Mc candidate moves per
// window, cand [S, Mc, 3], each candidate's predicted next hunger [S, Mc], as if it were the move of the window's last
// step. Computes only the move token, against the forward pass's cached keys and values.
int vmc_uni_candidates(void* h, int Mc, const float* cand, float* hunger);
// The ring (task 00062, model.age_key): keys and values kept across ticks. reset for N bodies; step: one new step for all
// bodies, move_probs [N, 3, bins]; the candidates then read the ring; move: the move made [N, 3] goes in.
int vmc_uni_ring_reset(void* h, int N);
int vmc_uni_ring_step(void* h, const uint8_t* images, const float* drives, const uint8_t* ep_start, float* move_probs);
int vmc_uni_ring_move(void* h, const float* moves);
// Task 00089: n ticks missed: the write position moves on, their slots unseen, nothing computed.
int vmc_uni_ring_skip(void* h, int n);
// As vmc_forward_uni, but returning only step t's values, values_t [S, ntok, V] (mode = 0: the expected values; 1: the
// most probable bin's centre; 2: the most probable bin and its more probable neighbour, weighted: task 00038), and the expected next bodies [S, T, 2] (the instinct's lookahead, task 00029).
int vmc_forward_uni_step(void* h, int S, int T, const uint8_t* images, const float* drives, const float* actions,
                         const uint8_t* ep_start, int t, int mode, float* values_t, float* next_body);
// Task 00138 ("draw"): as vmc_forward_uni_step, with step t's values drawn: every value picks one of the humps of its
// predicted chances by the dice u [S, ntok, V] in [0, 1) and takes the sharp value at that hump's peak (mode 2's at the
// most probable bin). A hump's weight is its chance, the sum of its bins' (peak = 0), or its tallest bin's (peak = 1:
// mode 2's ranking), raised to 1 / temp (temp <= 0: the heaviest hump; with peak = 1 that is mode 2). The bodies stay
// the expected ones.
int vmc_forward_uni_draw(void* h, int S, int T, const uint8_t* images, const float* drives, const float* actions,
                         const uint8_t* ep_start, int t, float temp, int peak, const float* u, float* values_t,
                         float* next_body);
// The same rule on given chances prob [n, K] (K at most 64) and dice u [n] -> out [n], without a model (the tests).
int vmc_draw_values(const float* prob, const float* u, long n, int K, float temp, int peak, float* out);
// The universal model's last update: each token's cross-entropy (its mean over the used values of the token it
// predicts) at the last step of the first S0 sequences, out [S0, ntok] (the gate's surprise, task 00028).
int vmc_uni_token_ce(void* h, int S0, float* out);
// The planner's model calls (latent dynamics, dyn_hidden > 0), as vm_encode (v only), vm_imagine and vm_decode.
int vmc_encode(void* h, int n, const uint8_t* images, float* vf_out);
int vmc_imagine(void* h, int n, int H, const float* s0, const float* h0, const float* actions, float* hunger,
                float* latents, float* frames);
int vmc_decode(void* h, int n, const float* latents, float* frames);
// Copies n parameters starting at offset to the host (for example the log-std after an update).
int vmc_get_range(void* h, long offset, long n, float* out);
// Waits for the device and returns the statistics of the last loss_and_grads / update.
int vmc_stats(void* h, VMCStats* out);
int vmc_params(void* h, float* out);   // device -> host copies of n_params floats
int vmc_grads(void* h, float* out);
int vmc_set_params(void* h, const float* params);   // host -> device (resets nothing else)
int vmc_sync(void* h);
// Task 00104, the state (model.state): the reads of the next pass [n, V] (n its frames S T, or the ring's bodies; else
// blank, 0.5); the writes' expected values of the last forward pass or ring step [n, V]; for a carry across two windows:
// the next update pass's gradient at its writes [n, V], its options (loss scale, accumulate onto the last pass, keep the
// gradient at its reads), that gradient [n, V]; and the clip and AdamW step on the gradients as they are.
int vmc_set_uni_state(void* h, int n, const float* st);
int vmc_uni_writes(void* h, int n, float* out);
int vmc_set_uni_carry(void* h, int n, const float* cg);
int vmc_set_uni_train(void* h, float scale, int accumulate, int read_grads);
int vmc_uni_read_grads(void* h, int n, float* out);
int vmc_apply_update(void* h, const PPOConfig* pcp);
// Task 00105, the flexible layer (P holds S + F): F <- lam F - k lr clamp(G / sqrt(v_hat), -1, 1) on the gradients as they are
// (use_grad = 0: F only decays), P changed by F's change; and F (which 0) or Adam's second moment (1) out [n_params].
int vmc_fast_step(void* h, float lam, float k, const PPOConfig* pcp, int use_grad);
int vmc_fast_get(void* h, int which, float* out);
int vmc_tie(void** hs, int n, long off, long len, int mean);   // task 00076: one table for n models

#ifdef __cplusplus
}
#endif
