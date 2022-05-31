cmake_minimum_required(VERSION 3.5)

project(OPUS)

# set(CXX_FLAGS -O3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

enable_testing()

set(CMAKE_CXX_STANDARD 14)

# find_package(Ceres REQUIRED)
find_package (Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")

set(SOURCES
    test/demo.cpp
    src/opus.cpp
    src/randomlhs.cpp
    src/surrogate.cpp
    src/LinearSolver.cpp
    src/blockbk.cpp
    src/test_utils.cpp
)

set(TEST_SURROGATE_SOURCES
    test/test_surrogate.cpp
    src/opus.cpp
    src/randomlhs.cpp
    src/surrogate.cpp
    src/LinearSolver.cpp
    src/blockbk.cpp
    src/test_utils.cpp
)

set(TEST_LINEARSOLVER_SOURCES
    test/test_LinearSolver.cpp
    src/LinearSolver.cpp
    src/test_utils.cpp
    src/blockbk.cpp
)

set(TEST_EVAL_SURR_SOURCES
    test/test_eval_surrogate.cpp
    src/LinearSolver.cpp
    src/surrogate.cpp
    src/test_utils.cpp
    src/blockbk.cpp
)

set(TEST_BLOCK_BK_SOURCES
    test/test_blockbk.cpp
    src/blockbk.cpp
    src/test_utils.cpp
)


set(TEST_MATRIX_UPDATE_SOURCES
    test/test_matrix_update.cpp
    src/test_utils.cpp
)

include_directories(
    ${CERES_INCLUDE_DIR}
)

add_executable(demo ${SOURCES})
add_executable(test_surrogate ${TEST_SURROGATE_SOURCES})
add_executable(test_linearsolver ${TEST_LINEARSOLVER_SOURCES})
add_executable(test_eval_surr ${TEST_EVAL_SURR_SOURCES})
add_executable(test_block_bk ${TEST_BLOCK_BK_SOURCES})
add_executable(test_matrix_update ${TEST_MATRIX_UPDATE_SOURCES})

target_include_directories(demo
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CERES_INCLUDE_DIRS}
)

target_include_directories(test_surrogate
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CERES_INCLUDE_DIRS}
)

target_include_directories(test_linearsolver
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(test_eval_surr
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(test_block_bk
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(test_matrix_update
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)



target_link_libraries (demo 
    PRIVATE
    Eigen3::Eigen 
    ${CERES_LIBRARIES}
)

target_link_libraries (test_surrogate 
    PRIVATE
    Eigen3::Eigen
    ${CERES_LIBRARIES}
)

target_link_libraries (test_linearsolver 
    PRIVATE
    Eigen3::Eigen
    ${CERES_LIBRARIES}
)

target_link_libraries (test_eval_surr
    PRIVATE
    Eigen3::Eigen
    ${CERES_LIBRARIES}
)

target_link_libraries (test_block_bk
    PRIVATE
    Eigen3::Eigen
    ${CERES_LIBRARIES}
)