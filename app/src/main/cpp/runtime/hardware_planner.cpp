#include "hardware_planner.h"
#include <algorithm>
namespace localimage::runtime {
bool HardwarePlanner::build(const ir::Graph& g,const DeviceCapabilities& c,bool preferNpu,ExecutionPlan& out,std::string& e) const {
    std::vector<uint32_t> order;
    if(!g.validate(e)||!g.topological(order,e)) return false;
    out.steps.clear(); out.steps.reserve(order.size());
    for(uint32_t id:order){
        const auto& n=g.nodes()[id];
        if(n.op==ir::Op::Input){out.steps.push_back({id,BackendKind::CPU});continue;}
        BackendKind selected=BackendKind::CPU;
        const bool npuOp=std::find(c.npuOps.begin(),c.npuOps.end(),ir::opName(n.op))!=c.npuOps.end();
        const bool vkOp=std::find(c.vulkanOps.begin(),c.vulkanOps.end(),ir::opName(n.op))!=c.vulkanOps.end();
        if(preferNpu&&c.npu&&npuOp) selected=BackendKind::NPU;
        else if(c.vulkan&&vkOp) selected=BackendKind::Vulkan;
        else selected=BackendKind::CPU;
        out.steps.push_back({id,selected});
    }
    return true;
}
}
