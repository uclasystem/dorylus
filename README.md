Push based ASPIRE implementation. This project is a cleaned up version of gift (forked on July 06, 2016).  
Implemented streaming-like processing as in Tornado (SIGMOD'16) paper.  

# Build Instructions:  
run from an ec2 instance (Ubuntu seems to be the easiest)

First you will need to install the following libraries:

	sudo apt install dsh libboost-all-dev
## DEPENDENCIES  
ZeroMQ  
ZooKeeper  

create an installs folder in root aspire directory for library dependencies:

	mkdir installs
	cd installs

make another directory inside installs called "out". This is where we will install the libraries we need


#### ZeroMQ Setup
Make sure that libtool pkg-config build-essential and autoconf are installed  
get ZeroMQ 4.1.4: https://archive.org/download/zeromq_4.1.4/zeromq-4.1.4.tar.gz  
extract the tarball in the installs directory  
inside the zeromq-4.1.4 directory, run the following commands:

	./configure --prefix=path/to/out --with-libsodium=no
	make install
	sudo ldconfig

finally, in the installs/out directory include should contain several zmq header files. Move zmq.hpp from the aspire-streaming/build directory into the include folder

#### ZooKeeper Setup
Make sure that ant and openjdk-8-jdk are installed  
get Zookeeper 3.4.6: https://github.com/apache/zookeeper/archive/release-3.4.6.tar.gz  
extract the tarball in the installs directory  

First, go into the src/c directory and make the following change to the file "configure.ac":

	line 37: delete "AM_PATH_CPPUNIT(1.10.2)"; replace with "PKG_CHECK_MODULES(CPPUNIT, cppunit >= 1.10.2)"

inside the zookeeper-release directory run the following command:

	ant deb
		** you may have to set JAVA_HOME if it has not been set

make sure to keep the zookeeper directory even after installation


## BUILDING ASPIRE
Once both major dependencies have been installed correctly we can go to the aspire-streaming root directory. Move into the "build" folder and open the Makefile. There are two values that need to be updated to reflect the values on your system:

	ZMQ_LIBPATH, ZMQ_INCPATH
	
	Make sure that each points to the correct placed (path/to/out/include, path/to/out/lib, etc)

finally run make  

# Preparing Input:

### Converting to Binary
inside the input directory there will be several source files.  
First, compile the snapToBinarySnap.cpp and binarySnapReader.cpp files. 
These will allow you to convert text files into binary to be usable by the system,
and the snapReader will help verify the correctness of the binary data.  
  
### Partitioning the Data
Once you have the data as a binary edge list, it needs to be partitioned. 
First we need to set up Metis:  
Extract the tarball in the installs directory as with ZeroMQ. 
Once it has been extracted, change metis.h in the include directory to 
reflect the data size being used. 
Then run the following commands in the top level metis directory:

	make config prefix=/path/to/out
	make install

Once metis has been setup, compile the partitioner with the following command:

	gcc -I/path/to/out/include -L/path/to/out/lib partitioner.cpp -o partitioner -lmetis

Now run the paritioner.sh script with the binary graph file, the number 
of vertices, and the number of machiens which will be in the cluster.  

Make sure the data is on all machines and has been properly setup and it should 
be runnable.
