// vm_cuda.h — the basic agent's update on the GPU (task 00011).
//
// The same model and parameter layout as vm_agent.cpp (CNN model: CoordConv encoder, causal Transformer, policy
// head, prediction MLP with the U-Net-skip frame decoder and the drives head), for the prediction-only update of the
// basic agent: next frame, next drives and next move, gradient-norm clip and AdamW. All of an update runs on the
// device without host synchronization. Inverse dynamics (task 00013) is supported. Refuses PPO, latent dynamics,
// patch tokens, and models without coord, pred_residual and pred_skip.
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
} VMCStats;

// Creates the model from a config and the flat parameters (n_params floats, the layout of vm_param_layout: offsets
// and sizes of every parameter tensor in ParamId order, n_tensors of them). NULL if the config is not supported or
// the layout differs from the one expected.
void* vmc_create(const VMConfig* cfg, const float* params, long n_params, const long* offsets, const long* sizes,
                 int n_tensors);
void vmc_free(void* h);
// One batch of S sequences of ctx steps, [S, ctx, ...] C-contiguous host arrays (images u8 HWC, next_small u8 HWC
// at pred_img, or the full-size next frames if next_full: then downsampled on the device). prev_reward is the
// command slot. Copied to the device.
int vmc_set_batch(void* h, int S, const uint8_t* images, const float* drives, const float* prev_action,
                  const float* prev_reward, const uint8_t* ep_start, const float* action, const uint8_t* next_small,
                  const float* next_drives, int next_full);
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
