#include <iostream>
#include <string>
#include <vector>
#include "../app/src/main/cpp/runtime/ir/localimage_ir.h"
int main(){
 using namespace localimage::tensor; using namespace localimage::ir; std::string e;
 Graph g; TensorShape sh({2,2}); TensorSpec s{"x",sh,TensorStride::contiguous(sh),TensorDType::F32,true};
 auto x=g.addValue(s); auto y=g.addValue({"y",sh,TensorStride::contiguous(sh),TensorDType::F32,true}); auto z=g.addValue({"z",sh,TensorStride::contiguous(sh),TensorDType::F32,true});
 auto i=g.addNode(Op::Input,{},"input"); if(!g.setOutputs(i,{x},e))return 1; auto n=g.addNode(Op::Add,{x,x},"add"); if(!g.setOutputs(n,{y},e))return 2; auto n2=g.addNode(Op::MatMul,{y,y},"matmul"); if(!g.setOutputs(n2,{z},e))return 3;
 if(!g.validate(e)){std::cerr<<e;return 4;} std::vector<uint32_t> order; if(!g.topological(order,e)||order.size()!=3||order[0]!=i||order[1]!=n||order[2]!=n2)return 5;
 Graph dup; auto d1=dup.addValue({"same",sh,TensorStride::contiguous(sh),TensorDType::F32,true}); auto d2=dup.addValue({"same",sh,TensorStride::contiguous(sh),TensorDType::F32,true}); auto di=dup.addNode(Op::Input,{},"input"); dup.setOutputs(di,{d1},e); if(dup.validate(e))return 6;
 std::cout<<"IR Graph validation: PASS\nTopological execution order: PASS\nShape inference: PASS\nDuplicate value-name rejection: PASS\n"; return 0;
}