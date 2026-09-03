#include "npu_backend.h"
#if __has_include(<android/hardware_buffer.h>)
#include <android/hardware_buffer.h>
#endif
namespace localimage::npu {
Capabilities BackendProbe::detect(){Capabilities c;c.available=false;c.backend=Backend::None;c.device="NPU capability probe: no vendor NPU runtime is linked";return c;}
}
