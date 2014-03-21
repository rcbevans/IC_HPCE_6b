HPC - Coursework 6
===========================================================================================================================================================
Richard Bennett rmb209 Richard Evans rce10

Running Instructions
===========================================================================================================================================================

The coursework source code is contained within src/

The cuda helper function files are contained within src/cuda/

The code variations are contained within bitecoin_miner.cpp (containing our tbb implementation), cuda_miner.cpp (containing the cuda implementation) and tbb_cuda_miner.cpp (containing the cuda and tbb implementation combined)

Each variation can be run both locally and connected to the exchange by running
make -B XXX_miner_connect_local OR   make -B XXX_miner_connect_exchange




===========================================================================================================================================================
HPCE Coursework 6
Bitecoining Mining

Introduction:

The aim of this coursework was mine bitecoins using the high performance computing tools taught to us throughout this course.

The original solution:

The original version of the bitecoin_client used a very slow, single core strategy to attempt to find combinations of indices which when hashed and xor'ed together would produce a proof as close to zero as possible.  A vector the size of maxIndices was populated with random monotonically increasing values and the hash process run to produce a proof.  If the proof was better than the currently stored best, it was replaced along with the current best solution.

There were many problems with this implementation and optimisations which were applied in our final version to maximise both the throughput of trials possible and the quality of the final result produced.



Our Solution:
==============

The solution we have used aims to reduce the amount of calculation required per iteration compared to the original code thereby boosting the number of trials possible. 

Our miner was built over several iterations, initially a single core miner was create and then the code and algorithm was improved to increase the number of trials from 35,000 to 400,000 trials. These performance increases can from removing redundant calculation outside of the loop, such as the calculation of the chainHash.  We made use of Intel's Amplifier XE profiling suite to identify hotspots in the code for poor performance and sought to make improvements by improving memory access patterns and keeping important information active.  We identified the wide_mul function as taking the vast majority of execttion cycles and so investigated the use of the karatsuba algorithm for multiplication.  Having implemented this algorithm, however, we found it was in fact slower.  We also identified optimisation in the wide_add where if carry was found to be 0, the input array could be passed directly through to the output for all remaining words.  Again, we found this was detrimental to performance due to the branch misprediction overhead.  We further optimised poolhashstep and directly set the limbs of the bigint_t being processed, rather than running multiple add instructions to achieve the same result.

In the second iteration we built a TBB based miner which enabled us to move from 400,000 trials to 700,000. The TBB version took advantage of the natural parallelism of independent trials to compute possible solutions completely in parallel.  Initially, we then ran a check on all of these possible proofs to find the best every iteration but this was a massive performance hit.  We realised that only an individual thread needs to keep track of its own best solution, with the best solution overall only being found at the end of a round when a submisison needs to be made.  We then realised that we could make use of parallel reduction to find the best solution from all threads in log(n) time, rather than n with linear comparisons.

In the third iteration a cuda miner was built so we could independtly test our code running on the GPU. This took the working TBB version of the code and ported it directly to Cuda to work on a larger scale running on the GPU.

In the fourth and final iteration we combined the cuda and TBB miners together to utilise the power of the CPU and GPU together.  This involved taking the TBB version of the code and invoking a thread which did exactly the same as the TBB version, but exectuting on the GPU as many times as possible before time is up.  Once the GPU thread was spawned, the main thread would continue with TBB exection until time was up.  At this point the GPU thread would then join, and a second thread spawned to exectute the Parallel reduce on the GPU memory buffers.  In the meantime, the CPU data would be reduced in parallel using TBB yielding the best solution found by the CPU.  The second GPU thread would then join and a comparison made between the GPU's best solution and that of the CPU.  The best result would be chosen and submitted for that round.

Once the cuda-TBB version was implemented we were able to achieve hash throughput of ~140,000,000/3s, however, we noticed that given the amount of trials being completed, we were obtaining only marginally better results than even the original single cpu thread version.  This is because for every proof calculated - the hard work - we were only comparing it to one hash value to seek a collision.  We realised that we could, for every block of proofs generated, run crosscomparisons to maximise, in parallel, the number of collisions possible, yielding much improved results ~e+54, rather that e+69.  To do this, each thread in the global space would choose an index and calculate its hash.  A second cuda kernel then runs which iterates over the other proofs produced in the previous step, xoring and seeking the best collision possible.  The step size is varied to maximize the number of potential proofs which can be checked, improving our chances of hitting a low score for the result.


Testing:
==========

To aid in our testing we created a local test server which keeps the round length, roundSalt, chainData size and hash steps constant in order for us to test the number of trials able to be produced in each round. This enabled us work on code and algorithmic improvements to quantifiably increase the number of trials run per round. The miner was also frequently tested on the exchange server to ensure that we were not submitting our results late and getting an "OVERDUE Error"



Work Distribution
==========



Parallel Reduction on Cuda -nVidia
http://www.google.com/url?q=http%3A%2F%2Fdeveloper.download.nvidia.com%2Fcompute%2Fcuda%2F1.1-Beta%2Fx86_website%2Fprojects%2Freduction%2Fdoc%2Freduction.pdf&sa=D&sntz=1&usg=AFQjCNEBdYr-Njtm2M5wt9qO6RBSqAXBhA