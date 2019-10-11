
cmake_minimum_required(VERSION 2.8)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE Tensorflow_LINK_FLAGS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE Tensorflow_COMPILE_FLAGS)

add_library(gnnengine SHARED export_gnnengine.cpp)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${Tensorflow_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed ${Tensorflow_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(export_operators SHARED export_operators.cpp)

# build the gradient operation which is used in inner_product_grad.py
# to register it
include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(export_operators_grad SHARED export_operators_grad.cpp)