/*
GPT-2 Inference in CUDA
Supports both dense and banded sparsity checkpoints.
No runtime masking needed - banded weights in the checkpoint are already sparse.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "llmc/utils.h"
#include "llmc/tokenizer.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void cublasCheck(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;

namespace cg = cooperative_groups;

int SORT_WINDOW = 0;  // Sort layer window size, 0 = disabled

// ----------------------------------------------------------------------------
// Sort layer forward kernel (inference only, no backward needed)

__global__ void sort_forward_kernel(
    float* out, float* sort_att, const float* inp,
    float alpha, float inv_tau,
    int B, int T, int C, int W
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) { return; }

    int t = idx % T;
    const float* x_i = inp + idx * C;
    int w_start = (t - W + 1) > 0 ? (t - W + 1) : 0;
    int w_len = t - w_start + 1;

    extern __shared__ float shared[];
    float* sim = shared + warp.meta_group_rank() * W;

    float scale = rsqrtf((float)C) * inv_tau;
    float maxval = -FLT_MAX;
    for (int w = 0; w < w_len; w++) {
        int j = w_start + w;
        const float* x_j = inp + (idx - t + j) * C;
        float dot = 0.0f;
        for (int c = warp.thread_rank(); c < C; c += warp.size()) {
            dot += x_i[c] * x_j[c];
        }
        dot = cg::reduce(warp, dot, cg::plus<float>{});
        float s = dot * scale;
        if (warp.thread_rank() == 0) { sim[w] = s; maxval = fmaxf(maxval, s); }
    }
    maxval = __shfl_sync(0xffffffff, maxval, 0);

    if (warp.thread_rank() == 0) {
        float sumexp = 0.0f;
        for (int w = 0; w < w_len; w++) { sim[w] = expf(sim[w] - maxval); sumexp += sim[w]; }
        float inv_sum = 1.0f / (sumexp + 1e-8f);
        for (int w = 0; w < w_len; w++) { sim[w] *= inv_sum; }
    }
    warp.sync();

    if (warp.thread_rank() == 0) {
        float* att_dst = sort_att + idx * W;
        for (int w = 0; w < W; w++) { att_dst[w] = (w < w_len) ? sim[w] : 0.0f; }
    }

    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float blend = 0.0f;
        for (int w = 0; w < w_len; w++) {
            int j = w_start + w;
            blend += sim[w] * inp[(idx - t + j) * C + c];
        }
        o[c] = (1.0f - alpha) * x_i[c] + alpha * blend;
    }
}

void sort_forward(float* out, float* sort_att, const float* inp,
                  float alpha, float inv_tau, int B, int T, int C, int W) {
    const int block_size = 512;
    const int num_warps_per_block = block_size / 32;
    const int grid_size = CEIL_DIV(B * T, num_warps_per_block);
    size_t shared_mem = num_warps_per_block * W * sizeof(float);
    sort_forward_kernel<<<grid_size, block_size, shared_mem>>>(
        out, sort_att, inp, alpha, inv_tau, B, T, C, W);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Forward-pass kernels

__device__ inline float4 add_float4(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void encoder_forward_kernel3(float4* out,
                               const int* inp, const float4* wte, const float4* wpe,
                               int B, int T, int C) {
    int C4 = C / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C4;
    if (idx < N) {
        int bt = idx / C4;
        int b = bt / T;
        int t = bt % T;
        int c4 = idx % C4;
        int ix = inp[b * T + t];
        out[b * T * C4 + t * C4 + c4] = add_float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
    }
}

__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N) return;

    const float* x = inp + idx * C;
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if (warp.thread_rank() == 0 && mean != nullptr) mean[idx] = m;

    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if (warp.thread_rank() == 0 && rstd != nullptr) rstd[idx] = s;

    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float n = s * (x[c] - m);
        o[c] = n * weight[c] + bias[c];
    }
}

__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    assert(blockDim.x <= 1024);
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpsInBlock = blockDim.x / 32;

    const float* x = inp + idx * T;
    float maxval = -INFINITY;
    for (int i = tid; i < T; i += blockDim.x) {
        if (i <= idx % T) maxval = fmaxf(maxval, x[i]);
    }
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 16));
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 8));
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 4));
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 2));
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 1));
    if (laneId == 0) shared[warpId] = maxval;
    __syncthreads();
    if (tid == 0) {
        float val = shared[tid];
        for (int i = 1; i < warpsInBlock; i++) val = fmaxf(val, shared[i]);
        shared[0] = val;
    }
    __syncthreads();
    maxval = shared[0];

    float sumval = 0.0f;
    for (int i = tid; i < T; i += blockDim.x) {
        sumval += (i <= idx % T) ? expf((x[i] - maxval) * inv_temperature) : 0.0f;
    }
    sumval += __shfl_xor_sync(0xffffffff, sumval, 16);
    sumval += __shfl_xor_sync(0xffffffff, sumval, 8);
    sumval += __shfl_xor_sync(0xffffffff, sumval, 4);
    sumval += __shfl_xor_sync(0xffffffff, sumval, 2);
    sumval += __shfl_xor_sync(0xffffffff, sumval, 1);
    if (laneId == 0) shared[warpId] = sumval;
    __syncthreads();
    if (tid == 0) {
        float val = shared[tid];
        for (int i = 1; i < warpsInBlock; i++) val += shared[i];
        shared[0] = val;
    }
    __syncthreads();
    sumval = shared[0];

    for (int i = tid; i < T; i += blockDim.x) {
        out[idx * T + i] = (i <= idx % T) ? expf((x[i] - maxval) * inv_temperature) / sumval : 0.0f;
    }
}

__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = inp[idx];
        float cube = 0.044715f * x * x * x;
        out[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + cube)));
    }
}

__global__ void add_bias_kernel(float* out, const float* bias, int N, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int oc = idx % OC;
        out[idx] += bias[oc];
    }
}

// ----------------------------------------------------------------------------
// Wrapper functions

void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N / 4, block_size);
    encoder_forward_kernel3<<<grid_size, block_size>>>((float4*)out, inp, (float4*)wte, (float4*)wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias, int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void matmul_forward(float* out, float* inp, float* weight, float* bias, int B, int T, int C, int OC) {
    int BT = B * T;
    float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, BT, C, &alpha, weight, C, inp, C, &beta, out, OC));
    if (bias != NULL) {
        int block_size = 256;
        int grid_size = CEIL_DIV(BT * OC, block_size);
        add_bias_kernel<<<grid_size, block_size>>>(out, bias, BT * OC, OC);
        cudaCheck(cudaGetLastError());
    }
}

void attention_forward(float* out, float* qkvr, float* att,
                       float* inp, int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH;
    float* q = qkvr + 0 * B * T * C;
    float* k = qkvr + 1 * B * T * C;
    float* v = qkvr + 2 * B * T * C;

    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    float scale = 1.0f / sqrtf(HS);
    float zero = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &scale,
        k, HS, T * HS, q, HS, T * HS, &zero, att, T, T * T, B * NH));

    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel5<<<grid_size, softmax_block_size, shared_mem_size>>>(att, 1.0f, att, B * NH * T, T);
    cudaCheck(cudaGetLastError());

    float one = 1.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one,
        v, HS, T * HS, att, T, T * T, &zero, q, HS, T * HS, B * NH));

    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(q, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward(float* out, const float* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

#define NUM_PARAMETER_TENSORS 16
#define GPT2_EOT 50256

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;

typedef struct {
    float* wte;
    float* wpe;
    float* ln1w;
    float* ln1b;
    float* qkvw;
    float* qkvb;
    float* attprojw;
    float* attprojb;
    float* ln2w;
    float* ln2b;
    float* fcw;
    float* fcb;
    float* fcprojw;
    float* fcprojb;
    float* lnfw;
    float* lnfb;
} ParameterTensors;

typedef struct {
    float* encoded;
    float* ln1;
    float* ln1_mean;
    float* ln1_rstd;
    float* qkv;
    float* atty;
    float* qkvr;
    float* att;
    float* attproj;
    float* residual2;
    float* ln2;
    float* ln2_mean;
    float* ln2_rstd;
    float* fch;
    float* fch_gelu;
    float* fcproj;
    float* residual3;
    float* lnf;
    float* lnf_mean;
    float* lnf_rstd;
    float* output;
    float* sort_out;  // (L, B, T, C) — only when SORT_WINDOW > 0
    float* sort_att;  // (L, B, T, W) — only when SORT_WINDOW > 0
} ActivationTensors;

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    ActivationTensors acts;
    size_t act_sizes[23];
    float* acts_memory;
    size_t num_activations;
    int* inputs;
    int batch_size;
    int seq_len;
    // Sort layer params (loaded from sidecar file)
    float* sort_params_memory;  // GPU: 2*L floats (alpha_raw[L], tau_raw[L])
} GPT2;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    int Vp = config.padded_vocab_size;
    int C = config.channels;
    int maxT = config.max_seq_len;
    int L = config.num_layers;
    param_sizes[0] = Vp * C;
    param_sizes[1] = maxT * C;
    param_sizes[2] = L * C;
    param_sizes[3] = L * C;
    param_sizes[4] = L * (3 * C) * C;
    param_sizes[5] = L * (3 * C);
    param_sizes[6] = L * C * C;
    param_sizes[7] = L * C;
    param_sizes[8] = L * C;
    param_sizes[9] = L * C;
    param_sizes[10] = L * (4 * C) * C;
    param_sizes[11] = L * (4 * C);
    param_sizes[12] = L * C * (4 * C);
    param_sizes[13] = L * C;
    param_sizes[14] = C;
    param_sizes[15] = C;
}

float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    float* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters * sizeof(float)));

    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* ptr = params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = ptr;
        ptr += param_sizes[i];
    }
    return params_memory;
}

void fill_in_activation_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    int C = config.channels;
    int NH = config.num_heads;
    int L = config.num_layers;
    int Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C;
    act_sizes[1] = L * B * T * C;
    act_sizes[2] = L * B * T;
    act_sizes[3] = L * B * T;
    act_sizes[4] = L * B * T * 3 * C;
    act_sizes[5] = L * B * T * C;
    act_sizes[6] = L * B * T * 3 * C;
    act_sizes[7] = L * B * NH * T * T;
    act_sizes[8] = L * B * T * C;
    act_sizes[9] = L * B * T * C;
    act_sizes[10] = L * B * T * C;
    act_sizes[11] = L * B * T;
    act_sizes[12] = L * B * T;
    act_sizes[13] = L * B * T * 4 * C;
    act_sizes[14] = L * B * T * 4 * C;
    act_sizes[15] = L * B * T * C;
    act_sizes[16] = L * B * T * C;
    act_sizes[17] = B * T * C;
    act_sizes[18] = B * T;
    act_sizes[19] = B * T;
    act_sizes[20] = B * T * Vp;
    act_sizes[21] = (SORT_WINDOW > 0) ? L * B * T * C : 0;            // sort_out
    act_sizes[22] = (SORT_WINDOW > 0) ? (size_t)L * B * T * SORT_WINDOW : 0; // sort_att
}

#define NUM_ACT_TENSORS 23

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (int i = 0; i < NUM_ACT_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, num_activations * sizeof(float)));

    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->qkvr, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3,
        &acts->lnf, &acts->lnf_mean, &acts->lnf_rstd, &acts->output,
        &acts->sort_out, &acts->sort_att
    };
    float* ptr = acts_memory;
    for (int i = 0; i < NUM_ACT_TENSORS; i++) {
        *(ptrs[i]) = ptr;
        ptr += act_sizes[i];
    }
    return acts_memory;
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) { fprintf(stderr, "Bad version in model file\n"); exit(1); }

    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    fill_in_parameter_sizes(model->param_sizes, model->config);

    size_t num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) num_parameters += model->param_sizes[i];
    model->num_parameters = num_parameters;

    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);

    float* params_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_cpu, sizeof(float), num_parameters, model_file);
    cudaCheck(cudaMemcpy(model->params_memory, params_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    free(params_cpu);
    fcloseCheck(model_file);

    model->acts_memory = NULL;
    model->inputs = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->sort_params_memory = NULL;
}

void gpt2_load_sort_params(GPT2 *model, const char* checkpoint_path) {
    // Load sort layer params from sidecar file (<checkpoint>.sort)
    char sort_path[512];
    snprintf(sort_path, sizeof(sort_path), "%s.sort", checkpoint_path);
    FILE* f = fopen(sort_path, "rb");
    if (f == NULL) {
        fprintf(stderr, "Error: sort layer enabled (-r %d) but sidecar file '%s' not found.\n", SORT_WINDOW, sort_path);
        fprintf(stderr, "Train with -r %d first to generate the sidecar.\n", SORT_WINDOW);
        exit(1);
    }
    int sort_header[2];
    freadCheck(sort_header, sizeof(int), 2, f);
    int file_window = sort_header[0];
    int file_L = sort_header[1];
    int L = model->config.num_layers;
    if (file_L != L) {
        fprintf(stderr, "Error: sort sidecar has L=%d but model has L=%d\n", file_L, L);
        exit(1);
    }
    if (file_window != SORT_WINDOW) {
        printf("Warning: sort sidecar has window=%d, CLI has -r %d. Using sidecar value.\n", file_window, SORT_WINDOW);
        SORT_WINDOW = file_window;
    }
    float* sort_cpu = (float*)mallocCheck(2 * L * sizeof(float));
    freadCheck(sort_cpu, sizeof(float), 2 * L, f);
    fclose(f);
    cudaCheck(cudaMalloc((void**)&model->sort_params_memory, 2 * L * sizeof(float)));
    cudaCheck(cudaMemcpy(model->sort_params_memory, sort_cpu, 2 * L * sizeof(float), cudaMemcpyHostToDevice));
    printf("Sort params loaded from %s (window=%d, %d params)\n", sort_path, SORT_WINDOW, 2 * L);
    // Print alpha values for visibility
    for (int l = 0; l < L && l < 3; l++) {
        float alpha = 1.0f / (1.0f + expf(-sort_cpu[l]));
        printf("  layer %d: alpha=%.4f (raw=%.4f), tau=%.4f (raw=%.4f)\n",
               l, alpha, sort_cpu[l], expf(sort_cpu[L + l]), sort_cpu[L + l]);
    }
    if (L > 3) printf("  ... (%d more layers)\n", L - 3);
    free(sort_cpu);
}

void gpt2_forward(GPT2 *model, int* inputs, int B, int T) {
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    GPT2Config config = model->config;
    int V = config.vocab_size;
    int Vp = config.padded_vocab_size;
    int L = config.num_layers;
    int NH = config.num_heads;
    int C = config.channels;

    if (B != model->batch_size || T != model->seq_len) {
        if (model->acts_memory != NULL) cudaCheck(cudaFree(model->acts_memory));
        if (model->inputs != NULL) cudaCheck(cudaFree(model->inputs));

        model->batch_size = B;
        model->seq_len = T;
        fill_in_activation_sizes(model->act_sizes, B, T, config);
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
    }

    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);

    // Bulk-copy sort params to CPU
    float* sort_params_cpu = NULL;
    if (SORT_WINDOW > 0 && model->sort_params_memory != NULL) {
        sort_params_cpu = (float*)mallocCheck(2 * L * sizeof(float));
        cudaCheck(cudaMemcpy(sort_params_cpu, model->sort_params_memory, 2 * L * sizeof(float), cudaMemcpyDeviceToHost));
    }

    for (int l = 0; l < L; l++) {
        float* residual;
        if (SORT_WINDOW > 0) {
            residual = (l == 0) ? acts.encoded : acts.sort_out + (l-1) * B * T * C;
        } else {
            residual = (l == 0) ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        }

        float* ln1_out = acts.ln1 + l * B * T * C;
        float* ln1_mean = acts.ln1_mean + l * B * T;
        float* ln1_rstd = acts.ln1_rstd + l * B * T;
        layernorm_forward(ln1_out, ln1_mean, ln1_rstd, residual, params.ln1w + l*C, params.ln1b + l*C, B, T, C);

        float* qkv = acts.qkv + l * B * T * 3 * C;
        matmul_forward(qkv, ln1_out, params.qkvw + l*3*C*C, params.qkvb + l*3*C, B, T, C, 3*C);

        float* atty = acts.atty + l * B * T * C;
        float* qkvr = acts.qkvr + l * B * T * 3 * C;
        float* att = acts.att + l * B * NH * T * T;
        attention_forward(atty, qkvr, att, qkv, B, T, C, NH);

        float* attproj = acts.attproj + l * B * T * C;
        matmul_forward(attproj, atty, params.attprojw + l*C*C, params.attprojb + l*C, B, T, C, C);

        float* residual2 = acts.residual2 + l * B * T * C;
        residual_forward(residual2, residual, attproj, B*T*C);

        float* ln2_out = acts.ln2 + l * B * T * C;
        float* ln2_mean = acts.ln2_mean + l * B * T;
        float* ln2_rstd = acts.ln2_rstd + l * B * T;
        layernorm_forward(ln2_out, ln2_mean, ln2_rstd, residual2, params.ln2w + l*C, params.ln2b + l*C, B, T, C);

        float* fch = acts.fch + l * B * T * 4 * C;
        matmul_forward(fch, ln2_out, params.fcw + l*4*C*C, params.fcb + l*4*C, B, T, C, 4*C);

        float* fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
        gelu_forward(fch_gelu, fch, B*T*4*C);

        float* fcproj = acts.fcproj + l * B * T * C;
        matmul_forward(fcproj, fch_gelu, params.fcprojw + l*C*4*C, params.fcprojb + l*C, B, T, 4*C, C);

        float* residual3 = acts.residual3 + l * B * T * C;
        residual_forward(residual3, residual2, fcproj, B*T*C);

        if (SORT_WINDOW > 0 && sort_params_cpu != NULL) {
            float* l_sort_out = acts.sort_out + l * B * T * C;
            float* l_sort_att = acts.sort_att + l * B * T * SORT_WINDOW;
            float alpha_raw = sort_params_cpu[l];
            float tau_raw = sort_params_cpu[L + l];
            float alpha = 1.0f / (1.0f + expf(-alpha_raw));
            float inv_tau = expf(-tau_raw);
            sort_forward(l_sort_out, l_sort_att, residual3, alpha, inv_tau, B, T, C, SORT_WINDOW);
        }
    }

    if (sort_params_cpu != NULL) { free(sort_params_cpu); }

    float* residual;
    if (SORT_WINDOW > 0) {
        residual = acts.sort_out + (L-1) * B * T * C;
    } else {
        residual = acts.residual3 + (L-1) * B * T * C;
    }
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);
}

void gpt2_free(GPT2 *model) {
    if (model->params_memory != NULL) cudaCheck(cudaFree(model->params_memory));
    if (model->acts_memory != NULL) cudaCheck(cudaFree(model->acts_memory));
    if (model->inputs != NULL) cudaCheck(cudaFree(model->inputs));
    if (model->sort_params_memory != NULL) cudaCheck(cudaFree(model->sort_params_memory));
}

// ----------------------------------------------------------------------------
// Sampling

float random_f32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) / (float)0xFFFFFFFFFFFFFFFFull;
}

int sample_softmax(const float* logits, int n, float coin) {
    double maxval = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (logits[i] > maxval) maxval = logits[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += exp(logits[i] - maxval);
    }
    double cumsum = 0.0;
    for (int i = 0; i < n; i++) {
        cumsum += exp(logits[i] - maxval) / sum;
        if (coin < cumsum) return i;
    }
    return n - 1;
}

// ----------------------------------------------------------------------------
// Main

void error_usage() {
    fprintf(stderr, "Usage: ./generate [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -e <string> model checkpoint path (default = checkpoint.bin)\n");
    fprintf(stderr, "  -n <int>    number of tokens to generate (default = 256)\n");
    fprintf(stderr, "  -p <string> prompt text (default = empty)\n");
    fprintf(stderr, "  -s <int>    random seed (default = time)\n");
    fprintf(stderr, "  -r <int>    sort layer window size (default = 0, disabled)\n");
    exit(1);
}

#ifndef TESTING
int main(int argc, char *argv[]) {
    const char* model_path = "checkpoint.bin";
    int num_tokens = 256;
    const char* prompt = "";
    unsigned long long rng_state = 0;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) error_usage();
        if (argv[i][0] != '-') error_usage();
        if (strlen(argv[i]) != 2) error_usage();
        if (argv[i][1] == 'e') model_path = argv[i+1];
        else if (argv[i][1] == 'n') num_tokens = atoi(argv[i+1]);
        else if (argv[i][1] == 'p') prompt = argv[i+1];
        else if (argv[i][1] == 's') rng_state = atoll(argv[i+1]);
        else if (argv[i][1] == 'r') SORT_WINDOW = atoi(argv[i+1]);
        else error_usage();
    }

    if (rng_state == 0) rng_state = (unsigned long long)time(NULL);

    printf("=== GPT-2 Inference ===\n");
    printf("Model: %s\n", model_path);
    printf("Tokens: %d\n", num_tokens);
    printf("Seed: %llu\n", rng_state);
    printf("---\n");

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cublasCheck(cublasCreate(&cublas_handle));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, model_path);
    if (SORT_WINDOW > 0) {
        gpt2_load_sort_params(&model, model_path);
    }
    int T = model.config.max_seq_len;
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;

    printf("Model loaded: %zu parameters\n", model.num_parameters);
    printf("Vocab: %d, MaxSeq: %d, Channels: %d\n", V, T, model.config.channels);
    printf("---\n");

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    int* gen_tokens = (int*)mallocCheck(T * sizeof(int));
    float* cpu_logits = (float*)mallocCheck(V * sizeof(float));

    for (int i = 0; i < T; i++) gen_tokens[i] = GPT2_EOT;

    printf("Generating:\n");
    for (int t = 1; t < num_tokens && t < T; t++) {
        gpt2_forward(&model, gen_tokens, 1, T);

        float* logits = model.acts.output + (t - 1) * Vp;
        cudaCheck(cudaMemcpy(cpu_logits, logits, V * sizeof(float), cudaMemcpyDeviceToHost));

        float coin = random_f32(&rng_state);
        int next_token = sample_softmax(cpu_logits, V, coin);
        gen_tokens[t] = next_token;

        if (tokenizer.init_ok) {
            const char* token_str = tokenizer_decode(&tokenizer, next_token);
            printf("%s", token_str);
        } else {
            printf("%d ", next_token);
        }
        fflush(stdout);
    }
    printf("\n---\n");

    free(gen_tokens);
    free(cpu_logits);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));

    return 0;
}
#endif
