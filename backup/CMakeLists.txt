cmake_minimum_required(VERSION 3.10)
project(BEVPoolV2)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Include the FetchContent module
include(FetchContent)

# Declare Eigen3 via FetchContent
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.3.9
)

# Make Eigen available to the project
FetchContent_MakeAvailable(eigen)

# Add executable for the test code
add_executable(test_bev_pool_v2
    bev_pool_v2/test.cpp 
)

# # Add executable for the test code
# add_executable(test_bev_pool_prep
#     bev_pool_prep/test.cpp 
# )

target_include_directories(test_bev_pool_prep PRIVATE ${eigen_SOURCE_DIR})

file(GLOB BIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/bev_pool_prep/*.bin")
file(COPY ${BIN_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(GLOB BIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/bev_pool_v2/*.bin")
file(COPY ${BIN_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})