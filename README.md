Push based ASPIRE implementation. This project is a cleaned up version of gift (forked on July 06, 2016).
Implemented streaming-like processing as in Tornado (SIGMOD'16) paper.

build instructions:
run from an ec2 instance (Ubuntu seems to be the easiest)

First you will need to install the following libraries:
	sudo apt install dsh libboost-all-dev

## DEPENDENCIES
ZeroMQ

ZooKeeper

create an installs folder in root aspire directory for library dependencies
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

finally, in the installs/out directory include should contain several zmq header files. Move zmq.hpp from the top level aspire directory into the include folder

#### ZooKeeper Setup
Make sure that ant and openjdk-8-jdk are installed
get Zookeeper 3.4.6: https://github.com/apache/zookeeper/archive/release-3.4.6.tar.gz

extract the tarball in the installs directory
inside the zookeeper-release directory run the following command:
	ant compile_jute
		** you may have to set JAVA_HOME if it has not been set

Next go into the src/c directory and make the following change to the file "configure.ac"
	line 37: delete "AM_PATH_CPPUNIT(1.10.2)"; replace with "PKG_CHECK_MODULES(CPPUNIT, cppunit >= 1.10.2)"

Once the file is modified, the system is ready to build. In the same directory (src/c) run the following commands:
	ACLOCAL="aclocal -I /usr/share/aclocal" autoreconf -if
		** for this command the directory path following -I should be the location of your "aclocal"
	./configure --prefix=path/to/out
	make
	make install



##### BUILDING ASPIRE #####
Once both major dependencies have been installed correctly we can go to the aspire-streaming root directory. Move into the "build" folder and open the Makefile. There are four values that need to be updated to reflect the values on your system:
	ZK_LIBPATH, ZK_INCPATH
	ZMQ_LIBPATH, ZMQ_INCPATH
	
	Make sure that each points to the correct placed (path/to/out/include, path/to/out/lib, etc)

	finally run make
