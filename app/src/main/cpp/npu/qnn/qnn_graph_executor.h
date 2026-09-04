#pragma once

#include "qnn_types.h"
#include "qnn_context.h"
#include "qnn_op_map.h"
#include "qnn_tensor.h"
#include "../../runtime/ir/localimage_ir.h"
#include "../../tensor/tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#ifdef LOCALIMAGE_QNN
#include "QNN/QnnGraph.h"
#include "QNN/QnnTensor.h"
#endif

namespace localimage {
namespace npu {
namespace qnn {

// Single operation execution result
struct NpuOpResult {
    bool ok = false;
    std::string error;
    tensor::Tensor output;
};

// QnnGraphExecutor: builds and executes QNN graphs
// Supports two modes:
// 1. Per-op execution (simple, low overhead for small graphs)
// 2. Subgraph execution (fused ops, better performance for larger subgraphs)
class QnnGraphExecutor {
public:
    QnnGraphExecutor();
    ~QnnGraphExecutor();

    QnnGraphExecutor(const QnnGraphExecutor&) = delete;
    QnnGraphExecutor& operator=(const QnnGraphExecutor&) = delete;

    // Initialize with QNN context
    bool initialize(QnnContext& context, std::string& error);

    // Check if executor is ready
    bool isReady() const { return ready_; }

    // Execute a single operation on NPU
    // This is the per-op mode: builds a tiny graph with one op, executes, returns
    bool executeSingleOp(ir::Op op,
                         const ir::Attributes& attr,
                         const std::vector<tensor::Tensor>& inputs,
                         tensor::Tensor& output,
                         std::string& error);

    // Execute a subgraph (multiple fused ops)
    // Takes a list of nodes and builds a single QNN graph
    bool executeSubgraph(const std::vector<ir::Node>& nodes,
                         const std::unordered_map<std::string, tensor::Tensor>& inputs,
                         std::unordered_map<std::string, tensor::Tensor>& outputs,
                         std::string& error);

    // Clear cached resources
    void clear();

private:
    bool ready_ = false;
    QnnContext* context_ = nullptr;
    OpMapper opMapper_;
    WeightPrepacker weightPrepacker_;

    // Cache of pre-built single-op graphs (key = op name)
    // Avoids rebuilding the graph for repeated ops
    struct SingleOpCache {
        std::string opName;
        QnnDataType dtype = QnnDataType::Float16;
        uint32_t inputCount = 0;
#ifdef LOCALIMAGE_QNN
        QnnGraph_Handle_t graphHandle = nullptr;
#endif
    };
    std::unordered_map<std::string, SingleOpCache> opCache_;

    // Build a single-op QNN graph
    bool buildSingleOpGraph(ir::Op op,
                            const ir::Attributes& attr,
                            const std::vector<QnnTensorInfo>& inputInfos,
                            const QnnTensorInfo& outputInfo,
                            std::string& error);

    // Execute a built single-op graph
    bool executeSingleOpGraph(const std::vector<QnnTensorInfo>& inputs,
                              QnnTensorInfo& output,
                              std::string& error);

    // Allocate and populate input tensors for QNN
    bool prepareQnnInputs(const std::vector<tensor::Tensor>& localInputs,
                          std::vector<QnnTensorInfo>& qnnInputs,
                          std::vector<void*>& tempAllocations,
                          std::string& error);

    // Read output from QNN back to LocalImage tensor
    bool readQnnOutput(const QnnTensorInfo& qnnOutput,
                       const tensor::TensorShape& expectedShape,
                       tensor::TensorDType expectedDtype,
                       tensor::Tensor& output,
                       std::string& error);

    // Free temporary allocations
    void cleanupTempAllocations(std::vector<void*>& allocs);

    // Infer output shape for an op
    bool inferOutputShape(ir::Op op,
                          const ir::Attributes& attr,
                          const std::vector<tensor::Tensor>& inputs,
                          tensor::TensorShape& outShape,
                          tensor::TensorDType& outDtype,
                          std::string& error);
};

} // namespace qnn
} // namespace npu
} // namespace localimage
