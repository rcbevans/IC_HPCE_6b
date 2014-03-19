#include "bitecoin_protocol.hpp"
#include "bitecoin_endpoint_client.hpp"

// Provies a basic (non-cryptographic) hash function
#include "contrib/fnv.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>

#include <thread>

// CUDA runtime
#include <cuda_runtime.h>
#include <curand_kernel.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

namespace bitecoin
{

extern void initialiseGPUArray(unsigned cudaBlockCount, const uint32_t maxIndices, const uint32_t hashSteps, const bigint_t &x, const uint32_t *d_hashConstant, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, curandState *d_state);

extern void cudaMiningRun(unsigned cudaBlockCount, const uint32_t maxIndices, const uint32_t hashSteps, const bigint_t &x, const uint32_t *d_hashConstant, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, curandState *d_state);

extern void cudaParallelReduce(unsigned cudaBlockCount, const uint32_t maxIndices, uint32_t *d_ParallelSolutions, uint32_t *d_ParallelProofs, uint32_t *gpuBestSolution, uint32_t *gpuBestProof);

class cudaEndpointClient : public EndpointClient
{
    //using EndpointClient::EndpointClient; //C++11 only!

protected:
    uint32_t *d_hashConstant, *d_ParallelProofs, *gpuParallelProofs;
    unsigned CUDA_PARALLEL_COUNT;

public:

    explicit cudaEndpointClient(std::string clientId,
                                std::string minerId,
                                std::unique_ptr<Connection> &conn,
                                std::shared_ptr<ILog> &log,
                                uint32_t *d_hashConstant,
                                uint32_t *d_ParallelProofs,
                                unsigned cuda_jobs) : EndpointClient(clientId,
                                            minerId,
                                            conn,
                                            log)
    {
        this->d_hashConstant = d_hashConstant;
        this->d_ParallelProofs = d_ParallelProofs;
        this->CUDA_PARALLEL_COUNT = cuda_jobs;
    };

    void MakeBid(
        const std::shared_ptr<Packet_ServerBeginRound> roundInfo,   // Information about this particular round
        const std::shared_ptr<Packet_ServerRequestBid> request,     // The specific request we received
        double period,                                                                          // How long this bidding period will last
        double skewEstimate,                                                                // An estimate of the time difference between us and the server (positive -> we are ahead)
        std::vector<uint32_t> &solution,                                                // Our vector of indices describing the solution
        uint32_t *pProof                                                                        // Will contain the "proof", which is just the value
    )
    {
        double tSafetyMargin = 0.5; // accounts for uncertainty in network conditions
        /* This is when the server has said all bids must be produced by, plus the
            adjustment for clock skew, and the safety margin
        */
        double tFinish = request->timeStampReceiveBids * 1e-9 + skewEstimate - tSafetyMargin;

        Log(Log_Verbose, "MakeBid - start, total period=%lg.", period);

        /*
            We will use this to track the best solution we have created so far.
        */
        std::vector<uint32_t> bestSolution(roundInfo->maxIndices);
        std::vector<uint32_t> gpuBestSolution(roundInfo->maxIndices);
        bigint_t bestProof, gpuBestProof; //uint32_t [8];
        //set bestProof.limbs = 1's
        wide_ones(BIGINT_WORDS, bestProof.limbs);

        double worst = pow(2.0, BIGINT_LENGTH * 8); // This is the worst possible score

        // Incorporate the existing block chain data - in a real system this is the
        // list of transactions we are signing. This is the FNV hash:
        // http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        hash::fnv<64> hasher;
        uint64_t chainHash = hasher((const char *)&roundInfo->chainData[0], roundInfo->chainData.size());

        bigint_t x;
        wide_x_init(&x.limbs[0], uint32_t(0), roundInfo->roundId, roundInfo->roundSalt, chainHash);

        std::vector<uint32_t> indices(roundInfo->maxIndices);

        //Define GPU shit
        uint32_t *d_ParallelSolutions;
        checkCudaErrors(cudaMalloc((void **)&d_ParallelSolutions, sizeof(uint32_t)*CUDA_PARALLEL_COUNT * roundInfo->maxIndices));

        checkCudaErrors(cudaMemcpy(d_hashConstant, &roundInfo->c[0], sizeof(uint32_t) * 4, cudaMemcpyHostToDevice));

        curandState *d_state;
        checkCudaErrors(cudaMalloc((void **)&d_state, sizeof(curandState)));

        unsigned nTrials = 0;

        nTrials += CUDA_PARALLEL_COUNT;

        initialiseGPUArray(CUDA_PARALLEL_COUNT, roundInfo->maxIndices, roundInfo->hashSteps, x, d_hashConstant, d_ParallelSolutions, d_ParallelProofs, d_state);

        do
        {
            nTrials += CUDA_PARALLEL_COUNT;

            cudaMiningRun(CUDA_PARALLEL_COUNT, roundInfo->maxIndices, roundInfo->hashSteps, x, d_hashConstant, d_ParallelSolutions, d_ParallelProofs, d_state);

        }
        while ((tFinish - now() * 1e-9) > 0);

        cudaParallelReduce(CUDA_PARALLEL_COUNT, roundInfo->maxIndices, d_ParallelSolutions, d_ParallelProofs, &gpuBestSolution[0], gpuBestProof.limbs);

        checkCudaErrors(cudaFree(d_ParallelSolutions));

        bestSolution = gpuBestSolution;
        wide_copy(BIGINT_WORDS, bestProof.limbs, gpuBestProof.limbs);

        solution = bestSolution;
        wide_copy(BIGINT_WORDS, pProof, bestProof.limbs);

        Log(Log_Verbose, "MakeBid - finish. Total trials %d", nTrials);

    }

};
};//End namespace bitecoin;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "bitecoin_client client_id logLevel connectionType [arg1 [arg2 ...]]\n");
        exit(1);
    }

    // We handle errors at the point of read/write
    signal(SIGPIPE, SIG_IGN);   // Just look at error codes

    try
    {
        std::string clientId = argv[1];
        std::string minerId = "Spunkmonkey's Miner";

        // Control how much is being output.
        // Higher numbers give you more info
        int logLevel = atoi(argv[2]);
        fprintf(stderr, "LogLevel = %s -> %d\n", argv[2], logLevel);

        std::vector<std::string> spec;
        for (int i = 3; i < argc; i++)
        {
            spec.push_back(argv[i]);
        }

        std::shared_ptr<bitecoin::ILog> logDest = std::make_shared<bitecoin::LogDest>(clientId, logLevel);
        logDest->Log(bitecoin::Log_Info, "Created log.");

        std::unique_ptr<bitecoin::Connection> connection {bitecoin::OpenConnection(spec)};

        unsigned CUDA_PARALLEL_COUNT = 768;

        uint32_t *d_hashConstant, *d_ParallelProofs;

        checkCudaErrors(cudaMalloc((void **)&d_hashConstant, sizeof(uint32_t) * 4));
        checkCudaErrors(cudaMalloc((void **)&d_ParallelProofs, sizeof(uint32_t)*CUDA_PARALLEL_COUNT * 8));

        bitecoin::cudaEndpointClient endpoint(clientId, minerId, connection, logDest, d_hashConstant, d_ParallelProofs, CUDA_PARALLEL_COUNT);
        endpoint.Run();

    }
    catch (std::string &msg)
    {
        std::cerr << "Caught error string : " << msg << std::endl;
        return 1;
    }
    catch (std::exception &e)
    {
        std::cerr << "Caught exception : " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception." << std::endl;
        return 1;
    }

    return 0;
}