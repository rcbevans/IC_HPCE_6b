#include "cuda_miner.cu.h"

namespace bitecoin
{

__global__ void firstCudaRun(const bigint_t x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, unsigned long long time, curandState *d_state)
{
    extern __shared__ uint32_t sharedMem[];
    uint32_t *localIndices = &sharedMem[0];
    uint32_t *localProofs = &sharedMem[maxIndices];

    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    int globalID = (blockID * blockDim.x) + threadID;

    if (threadID == 0)
    {
    	curand_init (time, globalID, threadID, d_state);

        uint32_t curr = 0;
        for (unsigned j = 0; j < maxIndices; j++)
        {
            curr += 1 + (4*blockID) + (uint32_t(curand(d_state)) & 268,435,455);
            localIndices[j] = curr;
        }
    }

    __syncthreads();

    bigint_t fph = x;
    fph.limbs[0] = localIndices[threadID];
    bigint_t point = CudaFastPoolHash(d_hashConstant, hashSteps, fph);

    cuda_wide_copy(8, &localProofs[threadID * 8], point.limbs);

    __syncthreads();

    if (threadID == 0)
    {
        for (unsigned j = 1; j < maxIndices; j++)
        {
            cuda_wide_xor(8, &localProofs[0], &localProofs[0], &localProofs[j * 8]);
        }

        cuda_wide_copy(8, &d_ParallelProofs[blockID * 8], &localProofs[0]);
    }
}

__global__ void cudaIteration(const bigint_t x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, curandState *d_state)
{
    extern __shared__ uint32_t sharedMem[];
    uint32_t *localIndices = &sharedMem[0];
    uint32_t *localProofs = &sharedMem[maxIndices];

    int threadID = threadIdx.x;
    int blockID = blockIdx.x;

    if (threadID == 0)
    {
        uint32_t curr = 0;
        for (unsigned j = 0; j < maxIndices; j++)
        {
            curr += 1 + (4*blockID) + (uint32_t(curand(d_state)) & 268,435,455);
            localIndices[j] = curr;
        }
    }

    __syncthreads();

    bigint_t fph = x;
    fph.limbs[0] = localIndices[threadID];
    bigint_t point = CudaFastPoolHash(d_hashConstant, hashSteps, fph);

    cuda_wide_copy(8, &localProofs[threadID * 8], point.limbs);

    __syncthreads();

    if (threadID == 0)
    {
        for (unsigned j = 1; j < maxIndices; j++)
        {
            cuda_wide_xor(8, &localProofs[0], &localProofs[0], &localProofs[j * 8]);
        }

        if (cuda_wide_compare(8, &localProofs[0], &d_ParallelProofs[blockID * 8]) < 0)
        {
            cuda_wide_copy(maxIndices, &d_ParallelSolutions[blockID * maxIndices], &localIndices[0]);
            cuda_wide_copy(8, &d_ParallelProofs[blockID * 8], &localProofs[0]);
        }
    }
}

__global__ void cudaReduce(const uint32_t maxIndices, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs)
{
    int threadID = threadIdx.x;
    int blockWidth = blockDim.x;

    for (unsigned toDo = blockWidth; blockWidth <= 1; blockWidth >>= 1)
    {
        if (threadID < toDo)
        {
            if (cuda_wide_compare(8, &d_ParallelProofs[(threadID * 8) + (toDo * 8)], &d_ParallelProofs[threadID * 8]) < 0)
            {
                cuda_wide_copy(8, &d_ParallelProofs[threadID * 8], &d_ParallelProofs[(threadID * 8) + (toDo * 8)]);
                cuda_wide_copy(maxIndices, &d_ParallelSolutions[threadID * maxIndices], &d_ParallelSolutions[(threadID * maxIndices) + (toDo * maxIndices)]);
            }
        }
        __syncthreads();
    }
}

void initialiseGPUArray(unsigned cudaBlockCount, const uint32_t maxIndices, const uint32_t hashSteps, const bigint_t &x, const uint32_t *d_hashConstant, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, curandState *d_state)
{
    dim3 grid(cudaBlockCount);
    dim3 threads(maxIndices);

    firstCudaRun <<< grid, threads, sizeof(uint32_t)*maxIndices + sizeof(uint32_t)*maxIndices * 8 >>> (x, d_hashConstant, maxIndices, hashSteps, d_ParallelSolutions, d_ParallelProofs, time(NULL), d_state);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();
}

void cudaMiningRun(unsigned cudaBlockCount, const uint32_t maxIndices, const uint32_t hashSteps, const bigint_t &x, const uint32_t *d_hashConstant, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs,
                   curandState *d_state)
{
    dim3 grid(cudaBlockCount);
    dim3 threads(maxIndices);

    cudaIteration <<< grid, threads, sizeof(uint32_t)*maxIndices + sizeof(uint32_t)*maxIndices * 8  >>> (x, d_hashConstant, maxIndices, hashSteps, d_ParallelSolutions, d_ParallelProofs, d_state);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();
}

void cudaParallelReduce(unsigned cudaBlockCount, const uint32_t maxIndices, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, uint32_t *gpuBestSolution, uint32_t *gpuBestProof)
{
    dim3 grid(1);
    dim3 threads(cudaBlockCount / 2);

    cudaReduce <<< grid, threads >>>(maxIndices, d_ParallelSolutions, d_ParallelProofs);

    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpyAsync(gpuBestProof, d_ParallelProofs, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpyAsync(gpuBestSolution, d_ParallelSolutions, sizeof(uint32_t)*maxIndices, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
}

}//End of Bitecoin Namespace