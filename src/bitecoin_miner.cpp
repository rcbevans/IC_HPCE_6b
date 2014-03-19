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

#include "tbb/tbb.h"

#define DIRTY_MAGIC_NUMBER 0xBA2E8BA3

namespace bitecoin
{

class cudaEndpointClient : public EndpointClient
{
    //using EndpointClient::EndpointClient; //C++11 only!

public:

    explicit cudaEndpointClient(std::string clientId,
                                std::string minerId,
                                std::unique_ptr<Connection> &conn,
                                std::shared_ptr<ILog> &log) : EndpointClient(clientId,
                                            minerId,
                                            conn,
                                            log) {};

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

        double worst = pow(2.0, BIGINT_LENGTH * 8); // This is the worst possible score

        // Incorporate the existing block chain data - in a real system this is the
        // list of transactions we are signing. This is the FNV hash:
        // http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        hash::fnv<64> hasher;
        uint64_t chainHash = hasher((const char *)&roundInfo->chainData[0], roundInfo->chainData.size());

        bigint_t x;
        wide_x_init(&x.limbs[0], uint32_t(0), roundInfo->roundId, roundInfo->roundSalt, chainHash);

        unsigned PARALLEL_COUNT = 32;

        uint32_t *parallel_Indices = (uint32_t *)malloc(sizeof(uint32_t) * roundInfo->maxIndices * PARALLEL_COUNT);
        uint32_t *parallel_Proofs = (uint32_t *)malloc(sizeof(uint32_t) * 8 * PARALLEL_COUNT);

        srand(now());

        //Initial Setup
        unsigned nTrials = 0;

        nTrials += PARALLEL_COUNT;

        auto tbbInitial = [ = ](unsigned i)
        {
            uint32_t curr = (rand() & 8191);
            for (unsigned j = 0; j < roundInfo->maxIndices; j++)
            {
                curr += 1 + (rand() & 524287);
                parallel_Indices[(i * roundInfo->maxIndices) + j] = curr;
            }

            bigint_t proof = FastHashReference(roundInfo.get(), roundInfo->maxIndices, &parallel_Indices[i * roundInfo->maxIndices], x);

            wide_copy(8, &parallel_Proofs[i * 8], proof.limbs);
        };

        tbb::parallel_for<unsigned>(0, PARALLEL_COUNT, tbbInitial);

        do
        {
            nTrials += PARALLEL_COUNT;

            auto tbbIteration = [ = ](unsigned i)
            {
                uint32_t localSolution[roundInfo->maxIndices];
                uint32_t curr = 4*i + (rand() & 8191);
                for (unsigned j = 0; j < roundInfo->maxIndices; j++)
                {
                    curr += 1 + (rand() & 524287);
                    localSolution[j] = curr;
                }

                bigint_t proof = FastHashReference(roundInfo.get(), roundInfo->maxIndices, &localSolution[0], x);

                if (wide_compare(BIGINT_WORDS, proof.limbs, &parallel_Proofs[i * 8]) < 0)
                {
                    wide_copy(roundInfo->maxIndices, &parallel_Indices[i * roundInfo->maxIndices], &localSolution[0]);
                    wide_copy(8, &parallel_Proofs[i * 8], proof.limbs);
                }
            };

            tbb::parallel_for<unsigned>(0, PARALLEL_COUNT, tbbIteration);

        }
        while ((tFinish - now() * 1e-9) > 0);

        //Do the parallel reduction to get our best solution
        for (int toDo = PARALLEL_COUNT / 2; toDo >= 1; toDo >>= 1)
        {
            auto tbbReduce = [ = ](unsigned i)
            {
                if (wide_compare(BIGINT_WORDS, &parallel_Proofs[(i * 8) + (toDo * 8)], &parallel_Proofs[i * 8]) < 0)
                {
                    wide_copy(8, &parallel_Proofs[i * 8], &parallel_Proofs[(i * 8) + (toDo * 8)]);
                    wide_copy(roundInfo->maxIndices, &parallel_Indices[i * roundInfo->maxIndices], &parallel_Indices[(i * roundInfo->maxIndices) + (toDo * roundInfo->maxIndices)]);
                }
            };

            tbb::parallel_for<unsigned>(0, toDo, tbbReduce);
        }

        wide_copy(BIGINT_WORDS, pProof, &parallel_Proofs[0]);
        wide_copy(roundInfo->maxIndices, &bestSolution[0], &parallel_Indices[0]);
        solution = bestSolution;

        double score = wide_as_double(BIGINT_WORDS, pProof);
        Log(Log_Verbose, "    Found best, nTrials=%d, score=%lg, ratio=%lg.", nTrials, score, worst / score);

        free(parallel_Proofs);
        free(parallel_Indices);

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

        bitecoin::cudaEndpointClient endpoint(clientId, minerId, connection, logDest);
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