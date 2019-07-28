# GNN-LAMBDA

Dataserver is originally is a push-based ASPIRE implementation. This project is a cleaned up version of gift (forked on July 06, 2016). Implemented streaming-like processing as in Tornado (SIGMOD'16) paper. Now the main logic of the engine has been completely simplified, and we are trying to integrate it with AWS Lambda threads. Ultimate goal is to achieve "*Affordable AI*" using the cheap scalability of serverless computing.


## [D] Data Server (Graph Server)

A cluster of EC2 instances form up the dataserver. Build the following dependencies and the system on **ALL** your AWS-EC2 instances (Ubuntu seems to be the easiest) under the **same** location.

### D.1 Dependencies

Make sure you have installed the following packages by `apt`:

- `build-essential`
- `libboost-all-dev`
- `dsh`
- `libtool`
- `pkg-config`
- `autoconf`
- `ant`
- `openjdk-8-jdk`

Assume the system directory is `~/gnn-lambda/`. Create a folder called `installs` in this directory where we put dependency source codes, and then make a `out` folder under `installs` to serve as ZeroMQ & Metis install location.

Folder tree should look like:

    /home/<USER>/gnn-lambda/
        |- installs/
            |- zeromq-x.x.x/
            |- ...
            |- out/
        |- ...

> Keep all dependency source directories after installation.

#### D.1.1 ZeroMQ

Get ZeroMQ 4.1.4 from: [https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz](https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz)

    $ wget https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz
    $ tar xvf zeromq-4.1.4.tar.gz

Go inside the `zeromq-4.1.4/` directory and compile it from source:

    $ ./configure --prefix=/home/<USER>/gnn-lambda/installs/out --with-libsodium=no
    $ make install
    $ sudo ldconfig

**Then** move `build/zmq.hpp` into `installs/out/include/` to replace it.

#### D.1.2 ZooKeeper

Get ZooKeeper 3.4.6 from: [https://github.com/apache/zookeeper/archive/release-3.4.6.tar.gz](https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz)

    $ wget https://github.com/apache/zookeeper/archive/release-3.4.6.tar.gz
    $ tar xvf release-3.4.6.tar.gz

Go inside the `zookeeper-release-3.4.6/src/c/` directory and make the following **change** to the file `configure.ac`:

    line 37: delete "AM_PATH_CPPUNIT(1.10.2)"; replace with "PKG_CHECK_MODULES(CPPUNIT, cppunit >= 1.10.2)"

Under the `zookeeper-release-3.4.6/` directory, run:

    $ ant deb

#### D.1.3 Metis

Get Metis 5.1.0 from: [http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz](https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz)

    $ wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    $ tar xvf metis-5.1.0.tar.gz

> Change `metis-5.1.0/include/metis.h` to reflect the partition data size being used.

Go inside the `metis-5.1.0/` directory and compile it from source:

    $ make config prefix=/home/<USER>/gnn-lambda/installs/out
    $ make install

### D.2 Building the Dataserver

**Ensure all dependencies have been installed correctly** before building ASPIRE.

Go to the `gnn-lambda/build` directory, then run the following for a release build (`-O3`):

    $ make mode=release [Benchmark-Name]  # Specify benchmark name for individual build, OR omit it to build all benchs

Or the following for a debug build (enabling `-g` and Address Sanitizer):

    $ make mode=debug [Benchmark-Name]    # Specify benchmark name for individual build, OR omit it to build all benchs

Clean the build by:

    $ make clean

### D.3 Preparing Input Graph

First we should have a text graph file with vertices numbered from 0 and each line representing an edge. Example content of a text graph file `test_graph.txt`:

    # Example with 6 vertices and 5 edges
    0 1
    0 2
    1 3
    2 4
    3 5

Go into the `inputs/` directory. Compile the utilities by:

    $ make

Then convert the graph to partitioned binary by:

    $ ./prepare <Path-to-Text-File> <Undirected? (0/1)> <Num-Vertices> <Num-Partitions>

This will create a `data/` folder under `input/`, where there is the binary graph file `*.bsnap` and a folder `parts_<Num-Partitions>/` containing partitioning infos inside.

Make sure you put the things in `data/` folder on all machines and under the **same** location, and the `ec2run.sh` script points to it correctly.

### D.4 Running the System

Make sure you have setup password-less `ssh` login among all your EC2 instances.

#### D.4.1 Setup Config Files

Under `run/` directory, **setup the content of `dshmachines` file**. Every line is in the form `username@ip` representing one of your nodes.

Setup the content of `zoo.basic` file for configuring ZooKeeper. Basically you need:

    line 2: set dataDir=/home/<USER>/zktmp/zooDataDir

Then **setup the content of `cserverinfo` file** with a line in the following format:

    <Dataserver-Port> <Coordinationserver-IP> <Coordinationserver-Port>

#### D.4.2 Running with Your Input

Use `utils/b+r` script for easy build and run across machines (make sure that the coordination server and the weight server have already been started and listening on you):

    Usage: ./b+r [Dataset]
        Dataset:        small(*), fb
    "(*)" means default.

Use `clear_out` script to clean all log files and output files, and reset GVID to 1:

    ./clear_out


## [C] Coordination Server

### C.1 Dependencies

Make sure the following packages have been installed:

- `libboost-all-dev`
- `libopenblas-dev`

#### C.1.1 AWS C++ SDK

Get AWS C++ SDK from:

    $ git clone https://github.com/aws/aws-sdk-cpp.git

Go inside the `aws-sdk-cpp/` repo, make a build directory and conduct an out-of-source build:

    $ mkdir build
    $ cmake .. -DCMAKE_BUILD_TYPE=Release -BUILD_ONLY=lambda
    $ make
    $ sudo make install

#### C.1.2 AWS Lambda C++ Runtime

Get AWS C++ Lambda runtime from:

    $ git clone https://github.com/awslabs/aws-lambda-cpp.git

Go inside the `aws-lambda-cpp/` repo, make a build directory and conduct an out-of-source build:

    $ mkdir build
    $ cmake .. -DCMAKE_BUILD_TYPE=Release
    $ make
    $ sudo make install

### C.2 Build the Coordserver

Under `src/coordination-server/` folder, do:

    $ ./bld

### C.3 Run the Coordserver

Under `src/coordination-server/` folder, configure a proper `run` script. Then do:
    
    $ ./run


## [W] Weight Server

### W.1 Build the Weightserver

Under `src/weight-server/` folder, do:

    $ ./bld

### W.2 Run the Weightserver

Under `src/weight-server/` folder, configure a proper `run` script. Then do:
    
    $ ./run


## [Appendix] Managing EC2 Servers

### Install AWS-CLI and Configure the Variables

On Ubuntu:

    $ sudo apt install awscli

On ArchLinux:

    $ sudo pacman -S aws-cli

On MacOS:

    $ brew install awscli

Configure AWS variables:

    $ aws configure
    ...     # Enter the required fields

### For [D]: Python `ec2_manager` Module Manual

#### Preparations

In project root directory `gnn-lambda/`, create a Python3 virtual environment:

    $ python3 -m venv <VM-Name>

Source the environment:

    $ source <VM-Name>/bin/activate

Install `boto3` module:

    $ pip3 install boto3

Setup EC2 servers using `ec2_manager` module:

    $ python3 -m ec2_manager setup --user=<AWS-Username> --key=<Path-to-SSH-Private-Key> --tag=<Tag-Name>   # `--tag=aspire-server`

Check your servers are ready by:

    $ python3 -m ec2_manager all check

#### Module Usage

Basic syntax:

    $ python3 -m ec2_manager <Node> <Option> [Arguments]

For `Node`:

- You can specify a node ID, e.g. `0`
- Using `all` to loop through all nodes to apply the following operation

For `Option`:

- `ssh`: ssh to the node
- `id`: get node id string
- `prip`: get node private IP address
- `pubip`: get node public IP address
- `send`: send files to `./` path of the node
- `get`: get files from the node
- `check`: check the state of the node
- `start`, `stop`, `reboot`, `reset`

### For [C]:

Trying to integrate the script into `ec2_manager` module... PENDING
