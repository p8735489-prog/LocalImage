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
