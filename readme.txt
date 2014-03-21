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





Our Solution:

The solution we have used aims to reduce the amount of calculation required per iteration compared to the original code thereby boosting the number of trials possible. 

Our miner was built over several iterations, initially a single core miner was create and then the code and algorithm was improved to increase the number of trials from 35,000 to 400,000 trials. 

In the second iteration we built a TBB based miner which enabled us to move from 400,000 trials to 700,000. 

In the third iteration a cuda miner was built so we could independtly test our code running on the GPU. 

In the fourth and final iteration we combined the cuda and TBB miners together to utilise the power of the CPU and GPU together. 



Strategy:
==========



Testing:
==========

To aid in our testing we created a local test server which keeps the round length, roundSalt, chainData size and hash steps constant in order for us to test the number of trials able to be produced in each round. This enabled us work on code and algorithmic improvements to quantifiably increase the number of trials run per round. The miner was also frequently tested on the exchange server to ensure that we were not submitting our results late and getting an "OVERDUE Error"





Work Distribution
==========
