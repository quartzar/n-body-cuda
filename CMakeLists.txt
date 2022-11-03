cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_STANDARD 17)
project(orbiterv6 CUDA CXX)

set(OpenGlLinkers -lglfw -lGLU -lGL -lglut -lGLEW)


set(PROJECT_SOURCES
        NbKernel_N2.cu NbodyRenderer.cpp NbSystemCUDA.cu)

set(PROJECT_HEADERS
        CONSTANTS.h
        NbKernel_N2.cuh NbodyRenderer.h NbSystemCUDA.cuh )


#############################################################################
add_executable(orbiterv6 ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(orbiterv6 ${OpenGlLinkers})

#add_compile_options(-G)

set_target_properties(orbiterv6 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
