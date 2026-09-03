#pragma once
#include "../limodel/limodel.h"
#include "../models/model_detector.h"
namespace localimage::converter {
class SafeTensorsConverter { public: bool convert(const safetensors::SafeTensorFile&,const std::string&,std::string&) const; };
}
