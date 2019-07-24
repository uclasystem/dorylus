# GNN-LAMBDA

Originally is a push-based ASPIRE implementation. This project is a cleaned up version of gift (forked on July 06, 2016). Implemented streaming-like processing as in Tornado (SIGMOD'16) paper. Now the main logic of the engine has been completely simplified, and
we are trying to integrate it with AWS Lambda threads. Ultimate goal is to achieve "*Affordable AI*" using the cheap scalability of serverless computing.

Build the following dependencies and the system on **ALL** your AWS-EC2 instances (Ubuntu seems to be the easiest) under the **same** location.

## Dependences

### Preparations

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
        |- run/
            |- ec2.run
            |- dshmachines
            |- zoo.basic
            |- ...
        |- build/
            |- Makefile
            |- zmq.hpp
            |- ...
        |- ...

> Keep all dependency source directories after installation.

### ZeroMQ

Get ZeroMQ 4.1.4 from: [https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz](https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz)

    $ wget https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz
    $ tar xvf zeromq-4.1.4.tar.gz

Go inside the `zeromq-4.1.4/` directory and compile it from source:

    $ ./configure --prefix=/home/<USER>/gnn-lambda/installs/out --with-libsodium=no
    $ make install
    $ sudo ldconfig

**Then** move `build/zmq.hpp` into `installs/out/include/` to replace it.

### ZooKeeper

Get ZooKeeper 3.4.6 from: [https://github.com/apache/zookeeper/archive/release-3.4.6.tar.gz](https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz)

    $ wget https://github.com/apache/zookeeper/archive/release-3.4.6.tar.gz
    $ tar xvf release-3.4.6.tar.gz

Go inside the `zookeeper-release-3.4.6/src/c/` directory and make the following **change** to the file `configure.ac`:

    line 37: delete "AM_PATH_CPPUNIT(1.10.2)"; replace with "PKG_CHECK_MODULES(CPPUNIT, cppunit >= 1.10.2)"

Under the `zookeeper-release-3.4.6/` directory, run:

    $ ant deb

### Metis

Get Metis 5.1.0 from: [http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz](https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz)

    $ wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    $ tar xvf metis-5.1.0.tar.gz

> Change `metis-5.1.0/include/metis.h` to reflect the partition data size being used.

Go inside the `metis-5.1.0/` directory and compile it from source:

    $ make config prefix=/home/<USER>/gnn-lambda/installs/out
    $ make install

## Building ASPIRE

**Ensure all dependencies have been installed correctly** before building ASPIRE.

Go to the `gnn-lambda/build` directory, then run the following for a release build (`-O3`):

    $ make mode=release [Benchmark-Name]  # Specify benchmark name for individual build, OR omit it to build all benchs

Or the following for a debug build (enabling `-g` and Address Sanitizer):

    $ make mode=debug [Benchmark-Name]    # Specify benchmark name for individual build, OR omit it to build all benchs

Clean the build by:

    $ make clean

## Preparing Input Graph

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

## Running ASPIRE

### SSH Password-less Login

Make sure you have setup password-less `ssh` login among all your EC2 instances.

### Setup Config Files

Under `run/` directory, **setup the content of `dshmachines` file**. Every line is in the form `username@ip` representing one of your nodes.

Setup the content of `zoo.basic` file for configuring ZooKeeper. Basically you need:

    line 2: set dataDir=/home/<USER>/zktmp/zooDataDir

### Running with Your Input

Use `utils/b+r` script for easy build and run across machines:

    Usage: ./b+r [Dataset] [Feature-File] [Layer-File]
        Dataset:        small(*), fb
        Feature-File:   (*), path_to_file
        Layer-File:     (*), path_to_file
    "(*)" means default.

Use `clear_out` script to clean all log files and output files, and reset GVID to 1:

    ./clear_out
