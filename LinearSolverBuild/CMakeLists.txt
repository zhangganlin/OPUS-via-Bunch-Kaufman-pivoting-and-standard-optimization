cmake_minimum_required(VERSION 3.5)

project(LinearSolver)

# set(CXX_FLAGS -o3)

enable_testing()

set(CMAKE_CXX_STANDARD 14)

# find_package(Ceres REQUIRED)


add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")


set(TEST_LINEARSOLVER_SOURCES
    ../test/test_LinearSolver.cpp
    ../src/LinearSolver.cpp
)

add_executable(test_linearsolver ${TEST_LINEARSOLVER_SOURCES})


target_include_directories(test_linearsolver
    PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
)


target_link_libraries (test_linearsolver 
    PRIVATE
)