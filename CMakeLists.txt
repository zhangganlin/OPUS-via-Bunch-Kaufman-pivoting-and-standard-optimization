cmake_minimum_required(VERSION 3.5)

project(OPUS)

# set(CXX_FLAGS -o3)

enable_testing()

set(CMAKE_CXX_STANDARD 14)

# find_package(Ceres REQUIRED)
find_package (Eigen3 REQUIRED)

add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")

set(SOURCES
    test/demo.cpp
    src/opus.cpp
    src/randomlhs.cpp
    src/surrogate.cpp
)

set(TEST_SURROGATE_SOURCES
    test/test_surrogate.cpp
    src/opus.cpp
    src/randomlhs.cpp
    src/surrogate.cpp
)


add_executable(demo ${SOURCES})
add_executable(test_surrogate ${TEST_SURROGATE_SOURCES})


target_link_libraries (demo Eigen3::Eigen)
target_link_libraries (test_surrogate Eigen3::Eigen)


# include_directories(
#     "/usr/include/eigen3"
# )


target_include_directories(demo
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(test_surrogate
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)