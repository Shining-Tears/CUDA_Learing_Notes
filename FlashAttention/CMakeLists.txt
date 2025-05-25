cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
set(CMAKE_PREFIX_PATH "/home/wenze/source/FlashAttention/libtorch/share/cmake/Torch;${CMAKE_PREFIX_PATH}")
#find_package(CUDAToolkit)
project(flash-atten-main CXX CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(example-app example-app.cpp)
# target_link_libraries(example-app "${TORCH_LIBRARIES}")
# set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

add_executable(flash-atten-main main.cpp flash-attention-v1.cu)
target_include_directories(flash-atten-main PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(flash-atten-main PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET flash-atten-main PROPERTY CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "Debug") 
    target_compile_options(flash-atten-main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
endif()
