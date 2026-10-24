// vm_agent.h — C API of the visual–motor agent (CPU, OpenMP).
//
// The model (CNN encoder -> causal Transformer -> policy/value/prediction heads),
// its acting path (KV cache + sampling), rollout memory, PPO and AdamW all live in
// C++. Python only simulates the world and calls these functions through ctypes
// (see agent_lib.py). All arrays are C-contiguous; images are uint8 HWC.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int img;            // input image side (64)
    int c1, c2, c3;     // encoder conv channels (k4 s2 p1 each: 64 -> 32 -> 16 -> 8)
    int feat;           // visual feature size
    int d_model;        // Transformer width
    int n_layer;        // Transformer depth
    int n_head;         // attention heads
    int d_ff;           // feed-forward width
    int ctx;            // context window = block length (tokens)
    int pred_img;       // predicted frame side (32)
    int pc1, pc2;       // decoder channels: pc2 x (pred_img/4)^2 -> pc1 x (pred_img/2)^2 -> 3 x pred_img^2
    int pred_hidden;    // decoder MLP hidden size
    int use_prev_reward;// 1 = previous reward is a token input (else that slot is zeroed)
    float logstd_init;  // initial log std of the (dx, dy) Gaussian
    int coord;          // 1 = CoordConv: coordinate channels in the encoder input and decoder (+ spatial broadcast)
    int pb;             // coord=1: size of the latent vector tiled over the first decoder grid
    int pred_residual;  // 1 = predict the change: next frame = current frame (downsampled) + decoder output
    int pred_skip;      // 1 = U-Net skips (encoder conv-3/conv-2 maps of the current frame + tiled action) into the decoder
    int inv_hidden;     // >0: inverse-dynamics head MLP([v_t, v_t+1]) -> executed (dx, dy), hidden size
    int dyn_hidden;     // >0: latent dynamics on the visual feature v: transition v' = v + MLP([v, a]), hunger
                        //     h' = sigmoid(MLP([v, h, a])) and a coarse decoder MLP(v) -> (pred_img/2)^2 x 3 frame
    int one_frame;      // 1 = each token attends only to itself: every prediction uses the current frame only
                        //     (training sequences stay ctx long, for inverse dynamics and latent-chain targets)
    int pi_hidden;      // >0: policy head with one ReLU hidden layer of this size on z (0 = linear head)
    int patch;          // >0: patch tokens (vm-00010). Each step is (img/patch)^2 patch tokens plus one token each for
                        //     the drives, the previous move and the command; each token predicts its own part of the
                        //     next state (patch: its cell of the next frame; drives; move). 0 = the CNN model
} VMConfig;

typedef struct {
    float gamma, lam;           // discount, GAE lambda
    float clip_eps;             // PPO ratio clip
    float vf_coef, ent_coef;    // value-loss / entropy coefficients
    float pred_coef;            // next-frame prediction loss coefficient (0 = off)
    float drive_coef;           // next-drive prediction loss coefficient (0 = off)
    float pred_backbone_scale;  // scale of prediction gradients entering the shared backbone
    float max_grad_norm;        // global grad-norm clip (<=0: off)
    float lr, beta1, beta2, adam_eps, weight_decay;
    float target_kl;            // early-stop epochs when approx KL exceeds this (<=0: off)
    float reward_scale;         // rewards are multiplied by this before GAE
    float vclip_eps;            // value clipping range (used if clip_vloss)
    float pred_changed_weight;  // extra loss weight on pixels that differ between consecutive frames (0 = plain MSE)
    float inv_coef;             // inverse-dynamics loss coefficient (needs inv_hidden > 0; 0 = off)
    float dyn_coef;             // latent dynamics (needs dyn_hidden > 0): imagined vs encoded latent MSE
    float dyn_h_coef;           //   imagined next-hunger MSE
    float dyn_x_coef;           //   coarse decoder MSE (real and imagined latents)
    int epochs, minibatches;    // PPO epochs per update, minibatches per epoch (over sequences)
    int use_ppo;                // 0 = skip policy/value/entropy losses (prediction-only learning)
    int norm_adv;               // normalize advantages per minibatch
    int clip_vloss;             // clipped value loss
    int dyn_horizon;            // imagined steps per chain in the latent-dynamics losses (0 = off)
    int dyn_h_horizon;          // chain steps that train the hunger head (0 = all; 1 = real latents only)
    float move_coef;            // next-move prediction: -log pi(a_t | context) of the move actually made (0 = off)
    int pi_detach;              // 1 = the policy head's gradient stops at its input z: it trains the head only,
                                //     never the Transformer or the encoder
    int pi_own_clip;            // 1 = the policy/value heads' gradient is clipped by its own norm and left out of
                                //     the rest's (with pi_detach, the head then cannot change the rest's update)
} PPOConfig;

typedef struct {
    float pg_loss, v_loss, entropy, approx_kl, clipfrac, explained_var;
    float pred_mse, pred_changed_mse, copy_mse, copy_changed_mse, changed_frac, drive_mse;
    float grad_norm, std_x, std_y, click_p, value_mean, return_mean, adv_mean, adv_std;
    float gn_backbone, gn_policy, gn_pred;   // pre-clip grad norms: shared backbone, policy/value heads, prediction heads
    float inv_mse, inv_r2;                   // inverse dynamics: MSE of (dx, dy) and R^2 vs predicting 0
    float dyn_mse, dyn_hunger_mse, dyn_recon_mse;   // latent dynamics: latent MSE, hunger MSE, decoder MSE
    int epochs_done, updates_done;
    float move_nll;             // next-move prediction: mean -log pi(a_t) of the moves actually made
} PPOStats;

// ---- model lifecycle ----
void* vm_create(const VMConfig* cfg, uint64_t seed);
void* vm_load(const char* path);                 // NULL on failure
int   vm_save(void* agent, const char* path, int with_optimizer);
void  vm_free(void* agent);
long  vm_num_params(void* agent);
void  vm_get_config(void* agent, VMConfig* out);
float* vm_params(void* agent);                   // flat parameter array (num_params)
float* vm_grads(void* agent);                    // flat gradient array (num_params)
int   vm_param_layout(void* agent, int idx, char* name, int name_len, long* offset, long* size); // -1 past end
void  vm_params_changed(void* agent);            // call after writing vm_params() directly
void  vm_set_threads(int n);
int   vm_get_threads(void);

// ---- acting ----
void* vm_ctx_create(void* agent, int n_envs, uint64_t seed);
void  vm_ctx_free(void* ctx);
void  vm_ctx_reset(void* ctx, const uint8_t* mask); // restart contexts of envs with mask[i] != 0 (NULL = all + phase)
// One step for all envs of ctx. Inputs: images [N,img,img,3] u8, drives [N,2], prev_action [N,3],
// prev_reward [N], ep_start [N] (1 = this obs is the first of an episode; NULL = none).
// Contexts restart at episode starts and at every ctx-th committed call (block boundary), mirroring
// the training sequences. Outputs: action [N,3] = (raw dx, raw dy, click in {0,1}), logp [N],
// value [N], optional mu [N,2] (tanh means) and click_p [N]. If rollout != NULL, inputs and outputs
// are stored at step t of the rollout (t % ctx must match the context phase).
// flags: VM_ACT_DETERMINISTIC (mean action, click if p > 0.5), VM_ACT_PEEK (do not advance the context;
// used for bootstrap values).
#define VM_ACT_DETERMINISTIC 1
#define VM_ACT_PEEK 2
void  vm_act(void* agent, void* ctx, const uint8_t* images, const float* drives, const float* prev_action,
             const float* prev_reward, const uint8_t* ep_start, int flags, float* action, float* logp,
             float* value, float* mu, float* click_p, void* rollout, int t);

// ---- rollout memory + PPO ----
void* vm_rollout_create(void* agent, int T, int N);   // T must be a multiple of ctx
void  vm_rollout_free(void* rollout);
// Outcome of step t: reward [N], done [N], next_images [N,img,img,3] (terminal frame when done), next_drives [N,2]
void  vm_rollout_outcome(void* rollout, int t, const float* reward, const uint8_t* done,
                         const uint8_t* next_images, const float* next_drives);
// Store the inputs of step t for all N sequences directly (training from replayed experience): images [N,img,img,3],
// drives [N,2], prev_action [N,3], prev_reward [N], ep_start [N] (may be NULL), action [N,3] raw. logp / value are 0.
void  vm_rollout_store(void* rollout, int t, const uint8_t* images, const float* drives, const float* prev_action,
                       const float* prev_reward, const uint8_t* ep_start, const float* action);
// Replace the action stored for step t (raw [N,3]) by one chosen outside the policy, e.g. by a planner.
// Only for ppo.use_ppo = 0: logp and value of the stored step are not recomputed.
void  vm_rollout_set_action(void* rollout, int t, const float* action);
void  vm_update(void* agent, void* rollout, const float* last_value, const PPOConfig* pc, PPOStats* stats);
// Debug/test: loss + gradients (into vm_grads) for the given rollout sequences, no optimizer step.
// Sequence s covers env (s % N), steps [(s / N) * ctx, (s / N + 1) * ctx). Returns total loss.
float vm_debug_grads(void* agent, void* rollout, const float* last_value, const PPOConfig* pc,
                     const int* seq_ids, int n_seq, PPOStats* stats);

// ---- analysis ----
// Full forward over S sequences of length T (T <= ctx). Inputs are [S,T,...]; ep_start [S,T] marks
// episode starts (position 0 always starts a context). pred_actions [S,T,3] are the actions the
// prediction head is conditioned on (NULL = skip prediction). Any output may be NULL.
// z [S,T,d], value [S,T], mu [S,T,2], click_p [S,T], pred_frames [S,T,pred_img,pred_img,3] (float, HWC),
// pred_drives [S,T,2].
void  vm_forward(void* agent, int S, int T, const uint8_t* images, const float* drives,
                 const float* prev_action, const float* prev_reward, const uint8_t* ep_start,
                 const float* pred_actions, float* z, float* value, float* mu, float* click_p,
                 float* pred_frames, float* pred_drives);

// Attention weights of one sequence of T steps: att [n_layer, n_head, Tt, Tt] with Tt = T (CNN model) or
// T * tokens per step (patch tokens); row = the attending token, zero outside the tokens it attends to.
void  vm_attention(void* agent, int T, const uint8_t* images, const float* drives, const float* prev_action,
                   const float* prev_reward, const uint8_t* ep_start, float* att);

// Encoder features of n frames: conv-3 map [n, c3*(img/8)^2] and visual feature [n, feat] (NULL to skip).
void  vm_encode(void* agent, int n, const uint8_t* images, float* a3, float* vf);
// Inverse-dynamics predictions of the executed (dx, dy) for n frame pairs (img0 -> img1), out [n,2].
void  vm_inverse(void* agent, int n, const uint8_t* img0, const uint8_t* img1, float* out);
// Latent dynamics (dyn_hidden > 0). Free-running imagination: n rollouts of H steps from latents s0 [n,feat]
// (visual features, e.g. from vm_encode) and hunger h0 [n] along raw actions [n,H,3], executed as the world
// does (clipped dx, dy; click if > 0.5). Step k feeds the hunger predicted at step k-1 back in.
// Outputs: hunger [n,H]; optional latents [n,H,feat] and decoded frames [n,H,(pred_img/2)^2*3] (HWC, float).
void  vm_imagine(void* agent, int n, int H, const float* s0, const float* h0, const float* actions, float* hunger,
                 float* latents, float* frames);
// Coarse decoder: frames [n,(pred_img/2)^2*3] (HWC, float) from latents [n,feat].
void  vm_decode(void* agent, int n, const float* latents, float* frames);

#ifdef __cplusplus
}
#endif
