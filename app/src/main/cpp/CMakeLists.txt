cmake_minimum_required(VERSION 3.22.1)
project(localimage_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(localimage_runtime SHARED
    native_runtime.cpp
    vulkan/vulkan_context.cpp
    vulkan/vulkan_buffer.cpp
    vulkan/vulkan_memory.cpp
    vulkan/vulkan_shader.cpp
    vulkan/vulkan_pipeline.cpp
    vulkan/vulkan_compute.cpp
    vulkan/vulkan_tensor.cpp
    safetensors/json_value.cpp
    safetensors/safe_tensor_file.cpp
    runtime/model_hash.cpp
    runtime/device_info.cpp
    runtime/cache_key.cpp
    runtime/runtime_cache.cpp
    tensor/tensor.cpp
    operators/cpu_operators.cpp
    graph/graph_runtime.cpp
    models/model_detector.cpp
    models/model_family.cpp
    models/model_package_inspector.cpp
    models/component_catalog.cpp
    sd15/sd15_runtime.cpp
    diffusion/diffusion_runtime.cpp
    runtime/ir/localimage_ir.cpp
    runtime/hardware_planner.cpp
    runtime/neural_runtime.cpp
    runtime/resolution_policy.cpp
    runtime/weight_store.cpp
    limodel/limodel.cpp
    transformer/transformer_runtime.cpp
    converter/safetensors_to_limodel.cpp
    npu/npu_backend.cpp
    npu/qnn/qnn_types.cpp
    npu/qnn/qnn_context.cpp
    npu/qnn/qnn_tensor.cpp
    npu/qnn/qnn_op_map.cpp
    npu/qnn/qnn_graph_executor.cpp
    sdxl/sdxl_runtime.cpp
    tokenizer/bpe_tokenizer.cpp
    image/png_writer.cpp
)

target_include_directories(localimage_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vulkan
    ${CMAKE_CURRENT_SOURCE_DIR}/safetensors
)

target_compile_options(localimage_runtime PRIVATE
    -std=c++20
    -fno-exceptions
    -fno-rtti
    -Wall -Wextra -Wpedantic
)

target_link_libraries(localimage_runtime
    android
    log
    vulkan
)

# Neural compute shaders are compiled to SPIR-V at build time when glslc is available.
# The runtime never treats missing binaries as GPU support; such operators remain
# explicitly UNSUPPORTED and may be selected for a CPU fallback by the planner.
find_program(LOCALIMAGE_GLSLC glslc)
if(LOCALIMAGE_GLSLC)
    set(LI_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/nn)
    set(LI_ELEMENTWISE_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../shaders/elementwise)
    set(LI_SHADER_OUT ${CMAKE_CURRENT_BINARY_DIR}/vulkan/generated)
    file(MAKE_DIRECTORY ${LI_SHADER_OUT})
    set(LI_SHADER_HEADERS)
    foreach(shader add mul silu gelu sub div exp sqrt rsqrt clamp)
        add_custom_command(
            OUTPUT ${LI_SHADER_OUT}/${shader}_spirv.h
            COMMAND ${LOCALIMAGE_GLSLC} -O ${LI_ELEMENTWISE_SHADER_DIR}/${shader}.comp -o ${LI_SHADER_OUT}/${shader}.spv
            COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8 python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../../../tools/spv_to_header.py ${LI_SHADER_OUT}/${shader}.spv ${LI_SHADER_OUT}/${shader}_spirv.h li_${shader}
            DEPENDS ${LI_ELEMENTWISE_SHADER_DIR}/${shader}.comp ${CMAKE_CURRENT_SOURCE_DIR}/../../../../tools/spv_to_header.py
            VERBATIM)
        list(APPEND LI_SHADER_HEADERS ${LI_SHADER_OUT}/${shader}_spirv.h)
    endforeach()
    foreach(shader matmul softmax layernorm rmsnorm groupnorm conv2d attention rope transpose slice broadcast concat upsample)
        add_custom_command(
            OUTPUT ${LI_SHADER_OUT}/${shader}_spirv.h
            COMMAND ${LOCALIMAGE_GLSLC} -O ${LI_SHADER_DIR}/${shader}.comp -o ${LI_SHADER_OUT}/${shader}.spv
            COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8 python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../../../tools/spv_to_header.py ${LI_SHADER_OUT}/${shader}.spv ${LI_SHADER_OUT}/${shader}_spirv.h ${shader}
            DEPENDS ${LI_SHADER_DIR}/${shader}.comp ${CMAKE_CURRENT_SOURCE_DIR}/../../../../tools/spv_to_header.py
            VERBATIM)
        list(APPEND LI_SHADER_HEADERS ${LI_SHADER_OUT}/${shader}_spirv.h)
    endforeach()
    add_custom_target(localimage_neural_shaders DEPENDS ${LI_SHADER_HEADERS})
    add_dependencies(localimage_runtime localimage_neural_shaders)
    target_include_directories(localimage_runtime PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/vulkan ${CMAKE_CURRENT_BINARY_DIR}/vulkan/generated)
else()
    message(STATUS "LocalImage: glslc not found; neural Vulkan shader binaries will remain explicitly unavailable until generated.")
endif()

# ============================================================================
# NPU / QNN (Qualcomm Hexagon DSP) backend
# ============================================================================
#
# Enable with: -DLOCALIMAGE_QNN=ON -DQNN_SDK_ROOT=/path/to/qnn/sdk
#
# The NPU code is always compiled in (stubbed when SDK unavailable).
# Define LOCALIMAGE_NO_NPU to completely exclude NPU code (reduces binary size).
#
# Minimum target: Snapdragon 8 Gen3 (Hexagon DSP v75)
# ============================================================================

option(LOCALIMAGE_QNN "Enable Qualcomm QNN HTP NPU backend" OFF)
option(LOCALIMAGE_NO_NPU "Exclude all NPU code from build" OFF)

if(LOCALIMAGE_NO_NPU)
    target_compile_definitions(localimage_runtime PRIVATE LOCALIMAGE_NO_NPU)
    message(STATUS "LocalImage: NPU backend excluded from build (LOCALIMAGE_NO_NPU)")
else()
    if(LOCALIMAGE_QNN)
        if(NOT QNN_SDK_ROOT)
            # Try common default locations
            set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")
            if(NOT QNN_SDK_ROOT)
                set(QNN_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../qnn_sdk")
            endif()
        endif()

        if(EXISTS "${QNN_SDK_ROOT}/include/QNN")
            target_include_directories(localimage_runtime PRIVATE
                ${QNN_SDK_ROOT}/include
            )
            target_compile_definitions(localimage_runtime PRIVATE LOCALIMAGE_QNN)

            # Link QNN libraries dynamically (loaded at runtime via dlopen)
            # The actual backend/skel libraries are loaded by qnn_context.cpp
            target_link_libraries(localimage_runtime dl)

            message(STATUS "LocalImage: QNN HTP NPU backend ENABLED (SDK: ${QNN_SDK_ROOT})")
            message(STATUS "LocalImage: Minimum target: Snapdragon 8 Gen3 (Hexagon v75)")
        else()
            message(WARNING "LocalImage: QNN SDK not found at ${QNN_SDK_ROOT} — NPU backend will be stubbed")
            set(LOCALIMAGE_QNN OFF)
        endif()
    else()
        message(STATUS "LocalImage: NPU code compiled in but QNN SDK not enabled (use -DLOCALIMAGE_QNN=ON)")
        message(STATUS "LocalImage: NPU will report unavailable at runtime until SDK is linked.")
    endif()
endif()
