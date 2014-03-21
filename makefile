SHELL=/bin/bash

CPPFLAGS += -std=c++11 -W -Wall  -g
CPPFLAGS += -O3
CPPFLAGS += -I include -I include/cudaInc/
CUDAINCLUDE += -I include -I include/cudaInc/

LDFLAGS += -L /opt/cuda/lib64/

LDLIBS += -ltbb -lrt -lcudart -lcuda

# For your makefile, add TBB and OpenCL as appropriate

src/cuda/cuda_miner.o :
	nvcc -c -arch=sm_20 -use_fast_math --optimize 3 $(CUDAINCLUDE) src/cuda/cuda_miner.cu -o src/cuda/cuda_miner.o

src/cuda_miner : src/cuda/cuda_miner.o
	g++ -o src/cuda_miner $(CPPFLAGS) src/cuda_miner.cpp src/cuda/cuda_miner.o $(LDFLAGS) $(LDLIBS)
	rm src/cuda/cuda_miner.o

src/cuda_tbb_miner : src/cuda/cuda_miner.o
	g++ -o src/cuda_tbb_miner $(CPPFLAGS) src/cuda_tbb_miner.cpp src/cuda/cuda_miner.o $(LDFLAGS) $(LDLIBS)
	rm src/cuda/cuda_miner.o

# Launch client and server connected by pipes
launch_pipes : src/bitecoin_server src/bitecoin_client
	-rm .fifo_rev
	mkfifo .fifo_rev
	# One direction via pipe, other via fifo
	src/bitecoin_client client1 3 file .fifo_rev - | (src/bitecoin_server server1 3 file - .fifo_rev &> /dev/null)

# Launch an "infinite" server, that will always relaunch
launch_infinite_server : src/bitecoin_server
	while [ 1 ]; do \
		src/bitecoin_server server1-Richy-Rich 3 tcp-server 4000; \
		sleep 15s; \
	done;

# Launch an "infinite" server, that will always relaunch
launch_infinite_test_server : src/bitecoin_test_server
	while [ 1 ]; do \
		src/bitecoin_test_server server1-Richy-Rich 3 tcp-server 4000; \
		sleep 15s; \
	done;

# Launch a client connected to a local server
connect_local : src/bitecoin_client
	src/bitecoin_client Richy-Rich 3 tcp-client localhost 4000
	
# Launch a client connected to a shared exchange
connect_exchange : src/bitecoin_client
	src/bitecoin_client Richy-Rich 3 tcp-client $(EXCHANGE_ADDR) $(EXCHANGE_PORT)

# Launch a client connected to a local server
miner_connect_local : src/bitecoin_miner
	src/bitecoin_miner Richy-Rich 3 tcp-client localhost 4000
	
# Launch a client connected to a shared exchange
miner_connect_exchange : src/bitecoin_miner
	src/bitecoin_miner Richy-Rich 3 tcp-client $(EXCHANGE_ADDR) $(EXCHANGE_PORT)

# Launch a client connected to a local server
cuda_miner_connect_local : src/cuda_miner
	src/cuda_miner Richy-Rich 3 tcp-client localhost 4000
	
# Launch a client connected to a shared exchange
cuda_miner_connect_exchange : src/cuda_miner
	src/cuda_miner Richy-Rich 3 tcp-client $(EXCHANGE_ADDR) $(EXCHANGE_PORT)

# Launch a client connected to a local server
tbb_cuda_miner_connect_local : src/cuda_tbb_miner
	src/cuda_tbb_miner Richy-Rich 3 tcp-client localhost 4000
	
# Launch a client connected to a shared exchange
tbb_cuda_miner_connect_exchange : src/cuda_tbb_miner
	src/cuda_tbb_miner Richy-Rich 3 tcp-client $(EXCHANGE_ADDR) $(EXCHANGE_PORT)