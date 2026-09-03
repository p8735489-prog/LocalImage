#pragma once
#include "../tensor/tensor.h"
#include <string>
namespace localimage::image { bool writePng(const tensor::Tensor& rgb,const std::string& path,std::string& error); }
