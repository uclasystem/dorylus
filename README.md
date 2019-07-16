Push-based ASPIRE implementation. This project is a cleaned up version of gift (forked on July 06, 2016). Implemented streaming-like processing as in Tornado (SIGMOD'16) paper.

Build the following dependencies and ASPIRE on **ALL** your AWS-EC2 instances (Ubuntu seems to be the easiest) under the **same** location.

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

Assume the ASPIRE directory is `~/aspire-streaming/`. Create a folder called `installs` in this directory where we put dependency source codes, and then make a `out` folder under `installs` to serve as ZeroMQ & Metis install location.

Folder tree should look like:

    /home/<USER>/aspire-streaming/
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

    wget https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz
    tar xvf zeromq-4.1.4.tar.gz

Go inside the `zeromq-4.1.4/` directory and compile it from source:

    ./configure --prefix=/home/<USER>/aspire-streaming/installs/out --with-libsodium=no
    make install
    sudo ldconfig

**Then** move `build/zmq.hpp` into `installs/out/include/` to replace it.

### ZooKeeper

Get ZooKeeper 3.4.6 from: [https://github.com/apache/zookeeper/archive/release-3.4.6.tar.gz](https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz)

    wget https://github.com/apache/zookeeper/archive/release-3.4.6.tar.gz
    tar xvf release-3.4.6.tar.gz

Go inside the `zookeeper-release-3.4.6/src/c/` directory and make the following **change** to the file `configure.ac`:

    line 37: delete "AM_PATH_CPPUNIT(1.10.2)"; replace with "PKG_CHECK_MODULES(CPPUNIT, cppunit >= 1.10.2)"

Under the `zookeeper-release-3.4.6/` directory, run:

    ant deb

### Metis

Get Metis 5.1.0 from: [http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz](https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz)

    wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
    tar xvf metis-5.1.0.tar.gz

> Change `metis-5.1.0/include/metis.h` to reflect the partition data size being used.

Go inside the `metis-5.1.0/` directory and compile it from source:

    make config prefix=/home/<USER>/aspire-streaming/installs/out
    make install

## Building ASPIRE

**Ensure all dependencies have been installed correctly** before building ASPIRE.

Go to the `aspire-streaming/build` directory, then run the following for a release build:

    make

Or the following for a debug build (enabling `-g` and Address Sanitizer):

    make

## Preparing Input Graph

### Converting to Binary

Go into the `inputs/` directory.

- Compile `snapToBinarySnap.cpp` & `binarySnapReader.cpp`. These will allow you to convert text files into binary to be usable by the system, and the snapReader will help verify the correctness of the binary data.  
  
### Partitioning the Data

Once Metis has been setup, compile the partitioner with:

    g++ -I../installs/out/include -L../installs/out/lib partitioner.cpp -o partitioner -lmetis

Now run the `paritioner.sh` script with the binary graph file, the number of vertices, and the number of machine nodes.

Make sure the data is put on all machines and under the **same** location.

## Running ASPIRE

### SSH Password-less Login

Make sure you have setup password-less `ssh` login among all your EC2 instances.

### Setup Config Files

Under `run/` directory, setup the content of `dshmachines` file. Every line is in the form `username@ip` representing one of your nodes.

Setup the content of `zoo.basic` file for configuring ZooKeeper. Basically you need:

    line 2: set dataDir=/home/<USER>/zktmp/zooDataDir

Open `ec2run.sh` and:

    line 3: set user=<USER>     # Your username

### Running with Your Input

Set variable `IP` around line 129 of `ec2run.sh` to the path of your partitioned input data (Make sure the path is the same for all nodes). Set `IK` and `SRC` accordingly.

Run ASPIRE from the current node's shell:

    ./ec2run.sh

and it will start ZooKeeper at all your nodes and do the work (basiclly by utilizing the `dsh` remote shell command).
