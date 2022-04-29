cmake_minimum_required (VERSION 3.0)
project (surrogate)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
add_executable (surrogate surrogate.cpp)
target_link_libraries (surrogate Eigen3::Eigen)