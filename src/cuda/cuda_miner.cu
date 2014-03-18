#include "cuda_miner.cu.h"

namespace bitecoin{

__global__ void runCudaMining(const bigint_t x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, unsigned long long time)
{
	extern __shared__ uint32_t localProofs[];

	int threadID = threadIdx.x;
	int blockID = blockIdx.x;
	int globalID = (blockID * blockDim.x) + threadID;

	curandState state;
	curand_init (time, globalID, threadID, &state);

	if (threadID == 0)
	{
		uint32_t curr = (uint32_t(curand(&state)) & 8191);
        for (unsigned j = 0; j < maxIndices; j++)
        {
            curr += 1 + (uint32_t(curand(&state)) & 524287);
            d_ParallelSolutions[(blockID * maxIndices) + j] = curr;
            // cuPrintf("Thread %d curr: %d", threadID, curr);
        }
	}

	__syncthreads();

    bigint_t fph = x;
    fph.limbs[0] = d_ParallelSolutions[globalID];

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

	void cudaMiningRun(unsigned cudaBlockCount, const bigint_t &bestProof, bigint_t &gpuBestProof, uint32_t *gpuBestSolution, const bigint_t &x, const uint32_t *d_hashConstant, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, uint32_t *GPUparallelProofs)
	{

		dim3 grid(cudaBlockCount);
		dim3 threads(maxIndices);

		runCudaMining<<< grid, threads, sizeof(uint32_t)*maxIndices*8 >>>(x, d_hashConstant, maxIndices, hashSteps, d_ParallelSolutions, d_ParallelProofs, (unsigned long long)time(NULL));

		getLastCudaError("Kernel execution failed");

		checkCudaErrors(cudaMemcpy(GPUparallelProofs, d_ParallelProofs, sizeof(uint32_t)*cudaBlockCount*8, cudaMemcpyDeviceToHost));

		int BestSolution = -1;
		for(int block = 0; block < cudaBlockCount; block++)
		{
			if(cuda_wide_compare(8, &GPUparallelProofs[block * 8], gpuBestProof.limbs) < 0)
			{
				BestSolution = block;
				cuda_wide_copy(8, gpuBestProof.limbs, &GPUparallelProofs[block * 8]);
			}
		}
		if(BestSolution != -1)
		{
			checkCudaErrors(cudaMemcpy(gpuBestSolution, d_ParallelSolutions + (BestSolution*maxIndices), sizeof(uint32_t)*maxIndices, cudaMemcpyDeviceToHost));
		}
	}

}//End of Bitecoin Namespace