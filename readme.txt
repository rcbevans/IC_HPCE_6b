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


To aid in our testing we created a local test server which keeps the round length constant in order for us to test the number of trials able to be produced each round. 

===========================================================================================================================================================
HPCE Coursework 6
Bitecoining Mining

Introduction:

The aim of this coursework was mine bitecoins using the high performance computing tools taught to us throughout this course.

The original solution:





Our Solution:

The solution we have used aims to reduce the amount of calculation required per iteration compared to the original code thereby boosting the number of trials possible. 

Our initial code based improvements running on a single core enabled us to move from 35000 trials in 3 seconds to 400,000 trials




Strategy:
==========


Testing:
==========

To test our solution we created a local_test_server which held all 


Work Distribution
==========
