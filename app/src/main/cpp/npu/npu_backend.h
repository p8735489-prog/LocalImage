#pragma once
#include <string>
#include <vector>
namespace localimage::npu {
enum class Backend { None, QNN, NNAPI };
struct Capabilities { bool available=false; Backend backend=Backend::None; std::string device; std::vector<std::string> supportedOps; };
class BackendProbe { public: static Capabilities detect(); };
}
