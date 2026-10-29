// vm_agent.h — C API of the visual–motor agent (CPU, OpenMP).
//
// The model (CNN encoder -> causal Transformer -> policy/value/prediction heads),
// its acting path (KV cache + sampling), rollout memory, PPO and AdamW all live in
// C++. Python only simulates the world and calls these functions through ctypes
// (see agent_lib.py). All arrays are C-contiguous; images are uint8 HWC.
#pragma once
#include <stdint.h>
#include <stdlib.h>

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
    int act_token;      // 1 (patch models with pred_residual): one more token per step, a learned query whose output
                        //     the policy and value heads read, trained by its own predictions of the next drives and
                        //     the coarse (8x8) next frame from [its output, executed move] (task 00020)
    int universal;      // 1 (with patch > 0): the universal model (task 00023, ARCHITECTURE.md). Tokens per step: body,
                        //     the patches, the move made; every token a value vector in [0, 1] through one shared input
                        //     map; fully causal; every token predicts the next token's values through one shared head
                        //     (bins two-hot cross-entropy). The CNN / patch heads and their losses do not apply
    int bins;           // universal: bins per predicted value (centres k / (bins - 1))
    int gray;           // universal: 1 = each patch token carries one channel (the frame's first; for env.gray worlds),
                        //     patch^2 values instead of patch^2 * 3 (task 00024)
    int in_bins;        // universal: 1 = bin-embedding input (task 00036): each used value split two-hot over the bins,
                        //     W_in d x (V * bins), a learned embedding per value slot and bin; 0 = W_in d x V on the
                        //     centred values
    int attn_split;     // universal: S > 1 = attention in regions (task 00044): the patch grid's rows cut into S bands,
                        //     each reaching attn_overlap rows past its edges; body and move tokens in every region; a
                        //     token attends to another only if their regions meet. 0 or 1 = off
    int attn_overlap;   // universal with attn_split: rows of patches each band reaches beyond its edges
    int attn_pred_global;   // universal with attn_split: 1 = the token that predicts a global token (the last patch,
                            //     which predicts the move) is in every region too
    int attn_cols;      // universal with attn_split: 1 = the bands are of the patch grid's columns instead of its rows
    int attn_checker;   // universal with attn_split: 1 = checkerboard regions (task 00046): patch (row r, column c) in
                        //     region (r + c) % attn_split, no bands (attn_overlap and attn_cols 0)
    int attn_win;       // universal: W > 0 = local attention (task 00047): two patches attend to each other only within W
                        //     rows and W columns; with uni_sum, a patch also reads every summary and a summary its band
    int uni_sum;        // universal: R > 0 = a summary token after every R rows of patches (task 00047): no values, so
                        //     the token before it has no target; 0 = none
    int uni_pyr;        // universal: L > 0 = a pyramid of summary tokens (task 00048): level l = 1..L, one per 2^l x 2^l
                        //     block of patches, placed right after the block's last patch; 0 = none
    int sum_thumb;      // universal with summaries: 1 = a summary's values are its block's pixels box-averaged to one
                        //     patch, a thumbnail (task 00049); every token then has a target; 0 = no values
    int sum_only;       // universal with uni_pyr and sum_thumb: 1 = the patches are not tokens (task 00054): a step is
                        //     [body, the summaries in the pyramid's order, move]; the patches only make the thumbnails
} VMConfig;

// The universal model's slots per step (tasks 00047, 00048): 0 the body; then the patches row by row, each summary
// token right after the last patch of its block (uni_sum: a band of R rows; uni_pyr: a 2^l x 2^l block at level l, finer
// levels first); nt - 1 the move. A slot: kind (0 body, 1 patch, 2 summary, 3 move), index (the patch, row-major, or
// the summary), level (patches 0) and position on its level's grid (row, column; uni_sum's summaries: band, 0).
typedef struct { int kind, idx, lev, r, c; } VMSlot;
static inline int vm_uni_nsum(const VMConfig* c) {
    if (!c->universal || c->patch <= 0) return 0;
    const int g = c->img / c->patch;
    int n = c->uni_sum > 0 ? g / c->uni_sum : 0;
    for (int l = 1; l <= c->uni_pyr; l++) n += (g >> l) * (g >> l);
    return n;
}
static inline int vm_uni_nt(const VMConfig* c) {
    const int g = c->img / c->patch;
    return (c->sum_only ? 0 : g * g) + 2 + vm_uni_nsum(c);
}
// Fills s [vm_uni_nt] (may be NULL); returns nt.
static inline int vm_uni_layout(const VMConfig* c, VMSlot* s) {
    const int g = c->img / c->patch;
    int n = 0, ns = 0;
#define VM_SLOT_(K, I, L, R, C) do { if (s) { s[n].kind = K; s[n].idx = I; s[n].lev = L; s[n].r = R; s[n].c = C; } n++; } while (0)
    VM_SLOT_(0, 0, 0, 0, 0);
    for (int r = 0; r < g; r++)
        for (int q = 0; q < g; q++) {
            if (!c->sum_only) VM_SLOT_(1, r * g + q, 0, r, q);   // sum_only (task 00054): no patch tokens
            if (c->uni_sum > 0 && q == g - 1 && (r + 1) % c->uni_sum == 0) VM_SLOT_(2, ns++, 1, r / c->uni_sum, 0);
            for (int l = 1; l <= c->uni_pyr; l++)
                if ((r + 1) % (1 << l) == 0 && (q + 1) % (1 << l) == 0) VM_SLOT_(2, ns++, l, r >> l, q >> l);
        }
    VM_SLOT_(3, 0, 0, 0, 0);
#undef VM_SLOT_
    return n;
}
// The pixel block a summary covers (task 00049): origin (by0, bx0) and the pooling factors (fy, fx) that box-average it
// to one patch: uni_sum, a band of R * patch rows by the full width (R x g); uni_pyr, a 2^l * patch square (2^l x 2^l).
static inline void vm_uni_block(const VMConfig* c, const VMSlot* s, int* by0, int* bx0, int* fy, int* fx) {
    const int ps = c->patch, g = c->img / ps;
    if (c->uni_sum > 0) { *by0 = s->r * c->uni_sum * ps; *bx0 = 0; *fy = c->uni_sum; *fx = g; return; }
    const int b = 1 << s->lev;
    *by0 = s->r * b * ps; *bx0 = s->c * b * ps; *fy = b; *fx = b;
}
// The centre of a slot's cells in cell coordinates (task 00052): a patch, its cell; a pyramid summary, its block's
// centre; a row summary, its band's centre. Returns 0 for the body and the move (no position).
static inline int vm_uni_center(const VMConfig* c, const VMSlot* s, float* y, float* x) {
    const int g = c->img / c->patch;
    if (s->kind == 1) { *y = (float)s->r; *x = (float)s->c; return 1; }
    if (s->kind != 2) return 0;
    if (c->uni_sum > 0) { *y = (s->r + 0.5f) * c->uni_sum - 0.5f; *x = 0.5f * (g - 1); return 1; }
    const int b = 1 << s->lev;
    *y = (s->r + 0.5f) * b - 0.5f; *x = (s->c + 0.5f) * b - 0.5f;
    return 1;
}
// Tokens per step without a target: the summaries, unless they carry thumbnails (task 00049)
static inline int vm_uni_untargeted(const VMConfig* c) { return c->sum_thumb ? 0 : vm_uni_nsum(c); }
// The kind of slot k and its index (as vm_uni_layout; slow: for one-off use)
static inline int vm_uni_slot(const VMConfig* c, int k, int* idx) {
    VMSlot* s = (VMSlot*)malloc(sizeof(VMSlot) * vm_uni_nt(c));
    vm_uni_layout(c, s);
    const int kind = s[k].kind;
    *idx = s[k].idx;
    free(s);
    return kind;
}

// The attention regions (task 00044): allow[q * nt + k] = 1 if a token of slot q may attend to one of slot k (slots per
// step: 0 the body, 1..g*g the patches row by row, nt - 1 the move). The body and the move are in every region, and with
// attn_pred_global also the last patch (it predicts the move); attn_checker: regions by (row + column) % S instead of
// bands. attn_win (task 00047, the table directional): two patches meet within W rows and columns; a patch attends to
// every summary, a summary to the patches of its band and to every summary; body and move global. With uni_pyr (task
// 00048): a token reads the summaries of its own or a coarser level whose position is within W of its own, projected to
// that level; a summary reads every patch of its block and no finer summary. Returns nt, or 0 when attention is not
// restricted (allow untouched; allow may be NULL to ask only for nt).
static inline int vm_attn_allow(const VMConfig* c, unsigned char* allow) {
    if (c->universal && c->patch > 0 && c->attn_win > 0) {
        const int nt = vm_uni_nt(c), W = c->attn_win;
        if (!allow) return nt;
        VMSlot* sl = (VMSlot*)malloc(sizeof(VMSlot) * nt);
        vm_uni_layout(c, sl);
        for (int q = 0; q < nt; q++)
            for (int k = 0; k < nt; k++) {
                const VMSlot a = sl[q], b = sl[k];
                int ok = 1;
                if (a.kind == 0 || a.kind == 3 || b.kind == 0 || b.kind == 3) ok = 1;   // body and move: global
                else if (a.kind == 1 && b.kind == 1) ok = abs(a.r - b.r) <= W && abs(a.c - b.c) <= W;
                else if (c->uni_sum > 0) ok = !(a.kind == 2 && b.kind == 1) || b.r / c->uni_sum == a.idx;   // rows
                else if (b.kind == 2)   // the pyramid: a summary of the reader's level or coarser, within W there
                    ok = a.lev <= b.lev && abs((a.r >> (b.lev - a.lev)) - b.r) <= W &&
                         abs((a.c >> (b.lev - a.lev)) - b.c) <= W;
                else ok = (b.r >> a.lev) == a.r && (b.c >> a.lev) == a.c;   // a summary reads its block's patches
                allow[q * nt + k] = (unsigned char)ok;
            }
        free(sl);
        return nt;
    }
    if (!c->universal || c->attn_split <= 1 || c->patch <= 0) return 0;
    const int g = c->img / c->patch, nt = g * g + 2, band = g / c->attn_split, o = c->attn_overlap;
    if (!allow) return nt;
    for (int q = 0; q < nt; q++)
        for (int k = 0; k < nt; k++) {
            int ok = 1;
            const int last = c->attn_pred_global ? nt - 2 : nt - 1;   // slots from here on are global
            if (q > 0 && q < last && k > 0 && k < last && c->attn_checker) {   // checkerboard: the same diagonal class
                ok = ((q - 1) / g + (q - 1) % g) % c->attn_split == ((k - 1) / g + (k - 1) % g) % c->attn_split;
            } else if (q > 0 && q < last && k > 0 && k < last) {   // two regional patches: their rows must share a band
                const int rq = c->attn_cols ? (q - 1) % g : (q - 1) / g;   // row (or column) of each
                const int rk = c->attn_cols ? (k - 1) % g : (k - 1) / g;
                ok = 0;
                for (int s = 0; s < c->attn_split && !ok; s++) {
                    const int lo = s * band - o, hi = (s + 1) * band - 1 + o;
                    ok = rq >= lo && rq <= hi && rk >= lo && rk <= hi;
                }
            }
            allow[q * nt + k] = (unsigned char)ok;
        }
    return nt;
}
// attn_split / attn_overlap / attn_win / uni_sum valid for this config
static inline int vm_attn_valid(const VMConfig* c) {
    if (c->attn_split < 0 || c->attn_overlap < 0 || c->attn_win < 0 || c->uni_sum < 0) return 0;
    if (c->uni_pyr < 0 || c->sum_thumb < 0 || c->sum_thumb > 1) return 0;
    if (c->sum_thumb && vm_uni_nsum(c) == 0) return 0;   // thumbnails need summaries
    if (c->sum_only < 0 || c->sum_only > 1) return 0;
    if (c->sum_only && (!c->uni_pyr || !c->sum_thumb || c->attn_win || c->uni_sum || c->attn_split > 1)) return 0;
    if (c->attn_win > 0 || c->uni_sum > 0 || c->uni_pyr > 0) {   // tasks 00047, 00048: the universal model only
        if (!c->universal || c->patch <= 0 || c->attn_split > 1) return 0;
        if (c->uni_sum > 0 && (c->img / c->patch) % c->uni_sum) return 0;
        if (c->uni_pyr > 0 && (c->uni_sum > 0 || (c->img / c->patch) % (1 << c->uni_pyr))) return 0;
    }
    if (c->attn_pred_global < 0 || c->attn_pred_global > 1 || c->attn_cols < 0 || c->attn_cols > 1) return 0;
    if (c->attn_checker < 0 || c->attn_checker > 1) return 0;
    if (c->attn_split <= 1)
        return c->attn_overlap == 0 && c->attn_pred_global == 0 && c->attn_cols == 0 && c->attn_checker == 0;
    if (!c->universal || c->patch <= 0) return 0;
    const int g = c->img / c->patch;
    if (c->attn_checker) return c->attn_overlap == 0 && c->attn_cols == 0 && c->attn_split <= g;
    return g % c->attn_split == 0 && c->attn_overlap < g / c->attn_split;
}

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
    float drive_changed_weight; // extra weight of the drives loss (and the latent hunger loss) at steps where hunger
                                //     changes by more than drive_change_eps (0 = plain MSE; vm-00013)
    float drive_change_eps;
    float act_coef;             // act token (act_token=1): coefficient of its next-drives + coarse-next-frame losses
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
    float act_drive_mse, act_frame_mse;   // act token: its next-drives MSE and coarse-next-frame MSE
    // universal model: cross-entropy (per used value) and squared error of the expected value, by the kind of token
    // predicted (the next body, a patch, the move)
    float uni_ce_body, uni_ce_patch, uni_ce_move, uni_mse_body, uni_mse_patch, uni_mse_move;
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
// The universal model (universal = 1): S sequences of T <= ctx steps. actions [S,T,3]: the move made at each step (the
// last step's may be a candidate). values [S, T*ntok, V]: every token's expected prediction of the next token's
// values; move_probs [S, T, 3*bins]: the bin probabilities of the move predicted at each step; next_body [S, T, 2]: the
// move tokens' expected prediction of the drives after their step (NULL to skip any).
void  vm_forward_uni(void* agent, int S, int T, const uint8_t* images, const float* drives, const float* actions,
                     const uint8_t* ep_start, float* values, float* move_probs, float* next_body);

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
