cmake_minimum_required(VERSION 3.10)
project(robot_inference_hw_interface CXX)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# POLICY_V3: 固定使用 V3 观测维度 (15帧×47=705)
set(POLICY_V3_VALUE 1)

set(LOCAL_LIBTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch)
set(LOCAL_PYTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/python_torch/torch)
set(TORCH_PREFIX_HINTS)

if(EXISTS "${LOCAL_LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
    list(APPEND TORCH_PREFIX_HINTS "${LOCAL_LIBTORCH_DIR}")
endif()

if(EXISTS "${LOCAL_PYTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
    list(APPEND TORCH_PREFIX_HINTS "${LOCAL_PYTORCH_DIR}/share/cmake")
endif()

find_program(PYTHON3_EXECUTABLE python3)
if(PYTHON3_EXECUTABLE)
    execute_process(
        COMMAND "${PYTHON3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
        RESULT_VARIABLE PYTORCH_PREFIX_RESULT
        OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX
        ERROR_QUIET
    )
    if(PYTORCH_PREFIX_RESULT EQUAL 0 AND EXISTS "${PYTORCH_CMAKE_PREFIX}/Torch/TorchConfig.cmake")
        list(APPEND TORCH_PREFIX_HINTS "${PYTORCH_CMAKE_PREFIX}")
    endif()
endif()

if(TORCH_PREFIX_HINTS)
    set(CMAKE_PREFIX_PATH ${TORCH_PREFIX_HINTS} ${CMAKE_PREFIX_PATH})
endif()

find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch/PyTorch C++ runtime not found. Put the CPU LibTorch SDK at src/inference/third_party/libtorch, install Python torch, install torch into src/inference/third_party/python_torch, or configure with -DCMAKE_PREFIX_PATH=/path/to/libtorch.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../motors
                 ${CMAKE_CURRENT_BINARY_DIR}/motors_build)

add_library(imu_a100_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../imu/a100/src/drivers/serial_port.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../imu/a100/src/protocol/frame/imu_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../imu/a100/src/drivers/imu_reader.cpp
)
target_include_directories(imu_a100_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../imu/a100/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../imu/a100/src/drivers
    ${CMAKE_CURRENT_SOURCE_DIR}/../imu/a100/src/protocol/frame
)
target_link_libraries(imu_a100_core PUBLIC m pthread)

add_library(robot_interface STATIC
    src/robot_interface.cpp
    src/torch_policy_runner.cpp
)
target_compile_definitions(robot_interface PUBLIC POLICY_V3=${POLICY_V3_VALUE})
target_include_directories(robot_interface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../motors/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../motors/src/drivers/myact
    ${CMAKE_CURRENT_SOURCE_DIR}/../motors/src/protocol/ethercat
    ${CMAKE_CURRENT_SOURCE_DIR}/../imu/a100/include
)
target_link_libraries(robot_interface
    PUBLIC
    MYACTUA_EtherCat
    imu_a100_core
    ${TORCH_LIBRARIES}
    pthread
)


add_executable(imu_only_test
    examples/imu_test.cpp
)
target_link_libraries(imu_only_test PRIVATE robot_interface)

add_executable(motors_test
    examples/motors_test.cpp
)
target_link_libraries(motors_test PRIVATE robot_interface)

add_executable(policy_test
    examples/policy_test.cpp
)
target_link_libraries(policy_test PRIVATE robot_interface)

add_executable(standing_policy_test
    examples/standing_policy_test.cpp
)
target_link_libraries(standing_policy_test PRIVATE robot_interface)

add_executable(motors_test_single
    examples/motors_test_single.cpp
)
target_link_libraries(motors_test_single PRIVATE robot_interface)

add_executable(hold_test
    examples/hold_test.cpp
)
target_link_libraries(hold_test PRIVATE robot_interface)

# 不依赖 EtherCAT/Torch 的脚踝 FK/IK/Jacobian 主机测试。
add_executable(ankle_jacobian_host_test
    tests/ankle_jacobian_host_test.cpp
)
target_include_directories(ankle_jacobian_host_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_test(NAME ankle_jacobian_host_test COMMAND ankle_jacobian_host_test)

# 不依赖 EtherCAT/Torch 的电机位置/速度单位往返测试。
add_executable(motor_units_host_test
    tests/motor_units_host_test.cpp
)
target_include_directories(motor_units_host_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../motors/include
)
add_test(NAME motor_units_host_test COMMAND motor_units_host_test)
