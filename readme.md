Overview
========

The idea of this coursework is to develop an accelerator for a
fake crypt-currency (see bit-coin or doge-coin for some "real"
currencies). Your goal is to maximise the number of coins
you can mine within the exercise.

There will be three macro rounds of "mining", where all
eligible miners will compete for a fixed pool of
coins. Within each round there will be multiple distinct
micro-rounds, and in each micro-round someone will
win a coin. In later stages the number of coins available
will increase. The maximum that any one miner can win per
macro-round is 80% of the coins - once they hit that limit,
I'm afraid they'll be kicked out to give the others a go
(though they will obviously be rewarded in the marks).

In order to enter a round, the miner must be submitted
before the round cutoff time and date. Once a miner
has entered a specific round it cannot be updated or
modified, but it will be able to take part in all
other remaining rounds. The cutoff dates for the rounds
are:

 - Thursday 20th, 23:59. Coin weighting 1.
 - Friday 21st, 23:59. Coin weighting 2.
 - Sunday 23rd, 23:59. Coin weighting 3.
 
The coursework will be assesed according to three
criterias:

 - 50% : Performance. How many coins did you win?
		 The reward process is non-linear, to avoid a
         complete winner takes all scenario but a very
		 large amount of the marks is determined soley
		 by how fast your miner is.
 
 - 40% : Applying techniques from the course. So
         I need to see examples of where you have
		 tried to used things like TBB or OpenCL, and where you
         have tried to analyse dependencies or transform
         iteration spaces.

 - 10% : Execution and compilation. How much did I
         have to modify code to make it work?

Your miners will compete with each other in
real-time on Amazon EC2 g2.2xlarge instances. The
specifications are freely available, so you can look at the relative
power of the GPU and the CPU. TBB and OpenCL 1.1 will both be
available.

Running the executables
=======================

There are two main executables in the distribution, built
from source files of the same name:

- bitecoin_client : A basic, but slow, client, which can connect to an exchange

- bitecoin_server : Implements a simply single-client server

There is also a hidden "exchange" server which implements
multi-threaded networking, and so allows multiple clients
to complete. The code is not available to you, but you can
connect to an instance running on a machine I control (look
in blackboard for the IP address and port).

Both executables take arguments of the form:

    ./bitecoin_XXXX chosen-identifier log-level connection-type [connection-args]

chosen-identifier : The name you are using to identify a
  particular server (exchange) or a client. The name must
  be unique, and the exchange server will not allow multiple
  people with the same id.

log-level : An integer parameter saying how much debug output
  you want to see. 0 is fatal errors, 5 is all debugging information.

connection-type : indicates how you want to connect, and
  can be: `tcp-client`, `tcp-server`, or `file`.
  
  tcp-client connects to an existing server and takes as parameters
  the host address, and the port number.
  
  tcp-server tells the executable to create a listening port, and
  wait for someone else to connect.
  
  file requires a pair of file names to be given, with "-" for standard
  files. The first file is the input stream, the second is the output
  stream.

For example, to set up a server and connect a client over tcp, you
could do:

    ./bitecoin_server MyServer 4 tcp-server 4000

in one terminal, then:

    ./bitecoin_client MyClient 4 tcp-client localhost 4000

in another terminal on the same machine. You should see the
server start up, then once the client connects they'll start
running the protocol and the client will "mine".

Or if you know that there is a server or exchange running
at $ADDRESS on $PORT, connect your client to it with:

    ./bitecoin_client MyClient 4 tcp-client $ADDRESS $PORT
    
For a valid exchange address and port look in the blackboard description.
I decided putting it in github was not a good idea. The exchanges
may occasionally go down, in which case your clients will get
kicked out, but they should come back eventually. There should
be some active miners in the exchange at all times, though none
of them are very fast.

You may also wish to run client and server connected via their
stdin and stdout, though it is a little more verbose. In bash, you
can do it via:

    mkfifo .reverse
	./bitecoin_client MyClient 2 file .reverse - | (./bitecoin_server MyServer 2 file - .reverse &> /dev/null)

Note that you'll also get interleaved logging if you don't
surpress the server output, and choose to run them in the
same terminal.

Once the client is running, you'll see it is printing information
about the transactions it is taking part in, such as
when it is waiting for a round to start, when it is
trying to find a good hash, and how the round eventually
finished. You can of course look at the protocol information
more deeply in your program.

Looking at the code
===================

The code is written in a rambling style, and there has been
no attempt to design it well. It is a mis-mash of different
parts, written at different points over about six months since
I first thought of this coursework. It is reasonably robust (I did
a lot of fuzz testing on the exchange/client interaction), but is certainly
not production quality, and shouldn't be seen as an example of
good design. Even in academia I would improve the code, but
I've left it as is to provide more of a challenge.

Much of the code is dedicated to communication and networking,
and from your point of view the important parts are likely to be:

  - bitecoin_endpoint_client.hpp : This handles the state
    machine for the client, and is somewhat responsible for
	strategy.

  - bitecoin_hashing.hpp : This contains the reference
    implementation for the hashing strategy, used by the
	slow client for hashing, and the server for verification.

If you follow the code, you'll be able to see the path
the client takes by pairing it up with the log messages.
Following through with a debugger is highly recommended
(though not if you're connected to an exchange, as you'll
get kicked off due to timeouts).

The hash operation
============

The server/exhange will initiate mining micro-rounds, and for each
micro-round specifies a set of round parameters and the time alotted
to that round. Clients must return the best solution they
can within the time limit of the round, by trying random
solutions and keeping the one with the best proof.

In this case the "solution" is a vector of one or more
indices, in ascending order, and the "proof" is a 256
bit integer. There is a hash function which does the
operation:

    bitecoin::HashReference(roundParameters,solution) -> proof

and the proof with the lowest numeric value is more likely to
win. So if we have solution1 and solution2, and:

    bitecoin::HashReference(roundParameters,solution1) < bitecoin::HashReference(roundParameters,solution2)

then solution 1 would be more likely to win out of the two.

The hash function is somewhat expensive to execute, so the aim
is to try as many solutions as possible in order to find the smallest
hash, while staying within the time limits of the round. If you
exceed the time limit of the round, then the your bid will be
discounted. Note that it is the time limit as determined by the
server - any delays due to networking are the clients problem.
In the final evaluation, everyone will have identical machines and
networking.

If you return an incorrect hash then you will be kicked off the
server as a potentially malicious actor.

The hash function itself can be determined from the code, as it is
less than 100 lines of code. Note that it is deliberately flawed
(cryptographically), and contains opportunities to demonstrate
the skills you have learnt in this course (and elsewhere).

Getting started
===============

I would suggest starting by cloning `src/bitecoin_client.cpp` into
a file `src/bitecoin_miner.cpp`, and creating a new class which
inherits from `bitecoin::EndpointClient`. Then override
`bitecoin::EndpointClient::MakeBid`, as that is where all the number
crunching happens. Start just by copying the existing `MakeBid`, then
think about how to optimise it.

Some suggestions:

- Look carefully at the data dependencies. Work out who is producing
  what, and how it is used.
  
- Work out what can be done in parallel (obviously).

- Try to isolate constant data from variable. 

- Avoid recalculating.

- Try to identify batches of computation, including different
  types of parallel batches.

- You may wish to adapt to the period of the round. Some rounds are
  short, some are long. It may be worth ignoring certain rounds as
  not worth worrying about, or focussing just on a particular length
  of round

- Within a given round, think about the optimal allocation of compute
  time to different parts.
  
- There may be calculations which can be re-ordered or even partially
  removed.

Submission
==========

The submission will consist of a zip file containing:

 - The source files needed to compile your miner. The
   miner can be based on a modified version of `src/bitecoin_client.cpp`,
   and the main source file should be called `src/bitecoin_miner.cpp`.
 
 - A readme.txt or readme.pdf which explains your approach
   or strategy, and how you split the work up between the
   pair.
   
When you submit determines which round you are in. If
you submit before a deadline, you can freely modify
the submission up to the deadline, at which point I
will take a snapshot, and that will be your final
submission.

The OS will be the Amazon-provided linux for GPU instances, and
the compiler g++ 4.7. I will enable SSE to the maximum level
supported on the processor. You will be firewalled off from everything
apart from the bitecoin exchange, so no trying to spin up worker
instances. Also, consider the Imperial ICT computing and networking
policies to extend to the Amazon instances - yes, we all know you can
jump a chroot jail: let's just assume you can do it and move on.

I will control compilation (though you can include your own makefiles),
and the strategy will be to compile `src/bitecoin_miner.cpp`. Any
extra cpp files should be named `src/bitecoin_miner_XXXX.cpp`, or
`src/miner/XXXX.cpp` for some value of XXXX - they will get compiled
in at the same time.

Your miner will be launched with `src` as the working directory, so
you can pick up any open cl kernels relative to this path (yes, I know,
this is terrible practise).

Note that I will go and fix any miners which are obviously failing
for some reason, or which do not quite compile. However, this
will lead to some reduction in marks in the execution
and compilation part. I will not fix performance problems.

