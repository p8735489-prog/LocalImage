#include "../app/src/main/cpp/vulkan/add_shader_spirv.h"
#include "../app/src/main/cpp/vulkan/mul_shader_spirv.h"
#include "../app/src/main/cpp/vulkan/silu_shader_spirv.h"
int main(){return (localimage::vulkan::shader::add_words>0&&localimage::vulkan::shader::mul_words>0&&localimage::vulkan::shader::silu_words>0)?0:1;}
