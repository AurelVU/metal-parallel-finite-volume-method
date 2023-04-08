cmake_minimum_required(VERSION 3.21)
project(test_cpp_metal)

set(CMAKE_CXX_STANDARD 17)

# Library definition
add_executable(METAL_CPP ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_executable(METAL_CPP_2 ${CMAKE_CURRENT_SOURCE_DIR}/main2.cpp)
target_compile_features(METAL_CPP_2 PRIVATE cxx_std_17)

find_package(Vulkan REQUIRED FATAL_ERROR)

add_custom_command(
        TARGET METAL_CPP
        COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/kernel.metal -o kernel.air
)

add_custom_command(
        TARGET METAL_CPP
        COMMAND xcrun -sdk macosx metallib kernel.air -o default.metallib
)

# Metal cpp headers
target_include_directories(METAL_CPP PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp")

target_include_directories(METAL_CPP_2 PUBLIC /Users/vladimirusakov/VulkanSDK/1.3.236.0/macOS/include/)

target_link_libraries(METAL_CPP_2 ${VULKAN_LIB_LIST})

target_link_directories(METAL_CPP_2 PUBLIC /Users/vladimirusakov/VulkanSDK/1.3.236.0/macOS/lib/)
target_link_libraries(METAL_CPP_2 vulkan)

# Metal cpp library (linker)
target_link_libraries(METAL_CPP
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore")
