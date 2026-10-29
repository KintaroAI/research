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
// As vmc_forward_uni, but returning only step t's values, values_t [S, ntok, V] (mode = 0: the expected values; 1: the
// most probable bin's centre; 2: the most probable bin and its more probable neighbour, weighted: task 00038), and the expected next bodies [S, T, 2] (the instinct's lookahead, task 00029).
int vmc_forward_uni_step(void* h, int S, int T, const uint8_t* images, const float* drives, const float* actions,
                         const uint8_t* ep_start, int t, int mode, float* values_t, float* next_body);
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

#ifdef __cplusplus
}
#endif
