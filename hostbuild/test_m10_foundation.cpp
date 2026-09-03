#include "../app/src/main/cpp/runtime/ir/localimage_ir.h"
#include "../app/src/main/cpp/runtime/hardware_planner.h"
#include <iostream>
using namespace localimage;
int main(){ir::Graph g; ir::TensorSpec a{"a",tensor::TensorShape({2,2}),tensor::TensorStride::contiguous(tensor::TensorShape({2,2})),tensor::TensorDType::F32,true}; auto v=g.addValue(a); auto n=g.addNode(ir::Op::Input,{},"input");std::string e;if(!g.setOutputs(n,{v},e)||!g.validate(e)){std::cerr<<e;return 1;}runtime::ExecutionPlan p;if(!runtime::HardwarePlanner().build(g,{},false,p,e)){std::cerr<<e;return 2;}std::cout<<"M10 IR foundation OK\n";return 0;}
