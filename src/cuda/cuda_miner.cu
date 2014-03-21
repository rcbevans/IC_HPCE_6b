#include "cuda_miner.cu.h"

namespace bitecoin
{

// __global__ void firstCudaRun(const bigint_t x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, unsigned long long time, curandState *d_state, uint32_t randomizer)
// {
//     extern __shared__ uint32_t sharedMem[];
//     uint32_t *localIndices = &sharedMem[0];
//     uint32_t *localProofs = &sharedMem[maxIndices];

//     int threadID = threadIdx.x;
//     int blockID = blockIdx.x;
//     int globalID = (blockID * blockDim.x) + threadID;

//     if (threadID == 0)
//     {
//      curand_init (time, globalID, threadID, d_state);

//         uint32_t curr = (4*blockID) + (uint32_t(curand(d_state)) & randomizer);
//         for (unsigned j = 0; j < maxIndices; j++)
//         {
//             curr += 1 + (4*blockID) + (uint32_t(curand(d_state)) & randomizer);
//             localIndices[j] = curr;
//         }
//     }

//     __syncthreads();

//     bigint_t fph = x;
//     fph.limbs[0] = localIndices[threadID];
//     bigint_t point = CudaFastPoolHash(d_hashConstant, hashSteps, fph);

//     cuda_wide_copy(8, &localProofs[threadID * 8], point.limbs);

//     __syncthreads();

//     if (threadID == 0)
//     {
//         for (unsigned j = 1; j < maxIndices; j++)
//         {
//             cuda_wide_xor(8, &localProofs[0], &localProofs[0], &localProofs[j * 8]);
//         }

//         cuda_wide_copy(8, &d_ParallelProofs[blockID * 8], &localProofs[0]);
//     }
// }

// __global__ void cudaIteration(const bigint_t x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, curandState *d_state, uint32_t randomizer)
// {
//     extern __shared__ uint32_t sharedMem[];
//     uint32_t *localIndices = &sharedMem[0];
//     uint32_t *localProofs = &sharedMem[maxIndices];

//     int threadID = threadIdx.x;
//     int blockID = blockIdx.x;

//     if (threadID == 0)
//     {
//         uint32_t curr = (4*blockID) + (uint32_t(curand(d_state)) & randomizer);
//         for (unsigned j = 0; j < maxIndices; j++)
//         {
//             curr += 1 + (uint32_t(curand(d_state)) & randomizer);
//             localIndices[j] = curr;
//         }
//     }

//     __syncthreads();

//     bigint_t fph = x;
//     fph.limbs[0] = localIndices[threadID];
//     bigint_t point = CudaFastPoolHash(d_hashConstant, hashSteps, fph);

//     cuda_wide_copy(8, &localProofs[threadID * 8], point.limbs);

//     __syncthreads();

//     if (threadID == 0)
//     {
//         for (unsigned j = 1; j < maxIndices; j++)
//         {
//             cuda_wide_xor(8, &localProofs[0], &localProofs[0], &localProofs[j * 8]);
//         }

//         if (cuda_wide_compare(8, &localProofs[0], &d_ParallelProofs[blockID * 8]) < 0)
//         {
//             cuda_wide_copy(maxIndices, &d_ParallelSolutions[blockID * maxIndices], &localIndices[0]);
//             cuda_wide_copy(8, &d_ParallelProofs[blockID * 8], &localProofs[0]);
//         }
//     }
// }

// __global__ void cudaReduce(const uint32_t maxIndices, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs)
// {
//     int threadID = threadIdx.x;
//     int blockWidth = blockDim.x;

//     for (unsigned toDo = blockWidth; blockWidth <= 1; blockWidth >>= 1)
//     {
//         if (threadID < toDo)
//         {
//             if (cuda_wide_compare(8, &d_ParallelProofs[(threadID * 8) + (toDo * 8)], &d_ParallelProofs[threadID * 8]) < 0)
//             {
//                 cuda_wide_copy(8, &d_ParallelProofs[threadID * 8], &d_ParallelProofs[(threadID * 8) + (toDo * 8)]);
//                 cuda_wide_copy(maxIndices, &d_ParallelSolutions[threadID * maxIndices], &d_ParallelSolutions[(threadID * maxIndices) + (toDo * maxIndices)]);
//             }
//         }
//         __syncthreads();
//     }
// }

// __global__ void cudaInitial(uint32_t *d_ParallelIndices, uint32_t *d_ParallelProofs, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions, const bigint_t x, const bigint_t nLessOne, const uint32_t *d_hashConstant, const uint32_t hashSteps, const unsigned baseNum)
// {
//     int globalID = blockIdx.x * blockDim.x + threadIdx.x;

//     uint32_t index = baseNum + globalID;

//     d_ParallelIndices[globalID] = index;

//     bigint_t proof = cudaHash(d_hashConstant,
//                                       hashSteps,
//                                       index,
//                                       x,
//                                       nLessOne);

//     cuda_wide_copy(8, &d_ParallelProofs[globalID * 8], proof.limbs);
// }

__global__ void cudaInitial(uint32_t *d_ParallelBestProofs)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    bigint_t ones;
    cuda_wide_ones(8, ones.limbs);
    cuda_wide_copy(8, &d_ParallelBestProofs[globalID * 8], ones.limbs);
}

__global__ void cudaIteration(uint32_t *d_ParallelIndices, uint32_t *d_ParallelProofs, const bigint_t x, const uint32_t *d_hashConstant, const uint32_t hashSteps, const unsigned baseNum, const unsigned offset)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t index = baseNum + offset + globalID;

    bigint_t proof = cudaHash(d_hashConstant,
                              hashSteps,
                              index,
                              x);

    cuda_wide_copy(8, &d_ParallelProofs[globalID * 8], proof.limbs);
    d_ParallelIndices[globalID] = index;
}

__global__ void cudaWideCrossHash(uint32_t *d_ParallelIndices, uint32_t *d_ParallelProofs, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions, const unsigned cudaTotalSize, const uint32_t maxIndices)
{
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned compare = globalID + 1; compare < blockDim.x; compare++)
    {
        bigint_t crossHash;
        cuda_wide_xor(8, crossHash.limbs, &d_ParallelProofs[globalID * 8], &d_ParallelProofs[compare * 8]);

        if (cuda_wide_compare(8, crossHash.limbs, &d_ParallelBestProofs[globalID * 8]) < 0)
        {
            cuda_wide_copy(8, &d_ParallelBestProofs[globalID * 8], crossHash.limbs);
            d_ParallelBestSolutions[globalID * maxIndices] = d_ParallelIndices[globalID * maxIndices];
            d_ParallelBestSolutions[(globalID * maxIndices) + 1] = d_ParallelIndices[compare * maxIndices];
        }
    }
}

__global__ void cudaReduce(const unsigned cudaDim, const uint32_t maxIndices, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions)
{

    int threadID = threadIdx.x;
    // int blockID = blockIdx.x;
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < cudaDim; i++)
    {
        if (cuda_wide_compare(8, &d_ParallelBestProofs[(globalID * 8) + (i * cudaDim * 8)], &d_ParallelBestProofs[globalID * 8]) < 0)
        {
            cuda_wide_copy(8, &d_ParallelBestProofs[globalID * 8], &d_ParallelBestProofs[(globalID * 8) + (i * cudaDim * 8)]);
            cuda_wide_copy(maxIndices, &d_ParallelBestSolutions[globalID * 8], &d_ParallelBestSolutions[globalID + (i * cudaDim)]);
        }
    }

    for (unsigned toDo = cudaDim; toDo <= 1; toDo >>= 1)
    {
        if (threadID < toDo)
        {
            if (cuda_wide_compare(8, &d_ParallelBestProofs[(threadID * 8) + (toDo * 8)], &d_ParallelBestProofs[threadID * 8]) < 0)
            {
                cuda_wide_copy(8, &d_ParallelBestProofs[threadID * 8], &d_ParallelBestProofs[(threadID * 8) + (toDo * 8)]);
                cuda_wide_copy(1, &d_ParallelBestSolutions[threadID], &d_ParallelBestSolutions[threadID + toDo]);
            }
        }
        __syncthreads();
    }
}

void cudaInit(const unsigned cudaDim, uint32_t *d_ParallelBestProofs)
{
    dim3 grid(cudaDim);
    dim3 threads(cudaDim);

    cudaInitial <<< grid, threads >>> (d_ParallelBestProofs);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();
}

void cudaIteration(uint32_t *d_ParallelIndices, uint32_t *d_ParallelProofs, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions, const bigint_t x, const uint32_t *d_hashConstant, const uint32_t hashSteps, const unsigned baseNum, const unsigned cudaDim, const unsigned offset, const unsigned cudaTotalSize, const uint32_t maxIndices)
{
    dim3 grid(cudaDim);
    dim3 threads(cudaDim);

    cudaIteration <<< grid, threads >>> (d_ParallelIndices, d_ParallelProofs, x, d_hashConstant, hashSteps, baseNum, offset);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();

    cudaWideCrossHash <<< grid, threads >>> (d_ParallelIndices, d_ParallelProofs, d_ParallelBestProofs, d_ParallelBestSolutions, cudaTotalSize, maxIndices);

    getLastCudaError("Kernel execution failed");

    cudaDeviceSynchronize();
}

void cudaParallelReduce(const unsigned cudaDim, const uint32_t maxIndices, uint32_t *d_ParallelBestProofs, uint32_t *d_ParallelBestSolutions, uint32_t *gpuBestSolution, uint32_t *gpuBestProof)
{
    dim3 grid(1);
    dim3 threads(cudaDim);

    cudaReduce <<< grid, threads >>>(cudaDim, maxIndices, d_ParallelBestProofs, d_ParallelBestSolutions);

    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpyAsync(gpuBestProof, d_ParallelBestProofs, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpyAsync(gpuBestSolution, d_ParallelBestSolutions, sizeof(uint32_t) * maxIndices, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
}

}//End of Bitecoin Namespace