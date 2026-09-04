# LocalImage NPU 转接头架构文档

## 概述

LocalImage 的 NPU 加速采用 **"运行时转接头" (Runtime Adapter)** 架构：

- **模型格式不变**：继续使用 limodel 专用格式（分 shard + mmap 零拷贝）
- **运行时映射**：执行时通过转接头将 LocalImage IR 算子按需映射到 NPU 指令
- **不做模型全转换**：不需要离线把模型转成 NPU 专用二进制（如 QNN .cpp/.bin）
- **三级 Fallback**：NPU → Vulkan GPU → CPU，算子级细粒度回退

**目标平台**：骁龙 8 Gen3 及以上（Hexagon DSP v75+）
**底层 SDK**：Qualcomm QNN HTP Backend

---

## 架构分层

```
┌─────────────────────────────────────────────────┐
│                limodel 模型文件                  │  ← 零拷贝 mmap
│  (manifest.json + weights.index + *.bin shards)  │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              LocalImage IR (27种算子)             │  ← 统一中间表示
│     Graph / Node / Op / Attributes / Tensor      │
└──────────┬───────────────────┬──────────────────┘
           │                   │
┌──────────▼──────┐  ┌─────────▼──────────┐
│  NpuBackend     │  │  VulkanBackend     │  ← 转接头入口
│  (graph层)      │  │  (已有)            │
└──────┬──────────┘  └─────────┬──────────┘
       │                       │
┌──────▼───────────────────────▼──────────────────┐
│              NPU → Vulkan → CPU Fallback          │
│         (graph_runtime.cpp 三级分派)              │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│         QnnGraphExecutor (NPU执行引擎)            │
│  ┌─────────────┬──────────────┬──────────────┐   │
│  │  OpMapper   │ QnnTensorAdapter │ 权重缓存  │   │
│  │ (算子映射)   │  (布局转换)    │ (Prepack)  │   │
│  └─────────────┴──────────────┴──────────────┘   │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│            QnnContext (设备管理)                  │
│  后端加载 / 设备创建 / 内存分配 / DSP版本检测      │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│        Qualcomm QNN SDK (libQnnHtp.so)           │
│     HTP v75 / v79 / v81 skel (DSP版本选择)        │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│     Hexagon DSP NPU (骁龙 8 Gen3 / Elite / Elite Gen5) │
└─────────────────────────────────────────────────┘
```

---

## 文件清单

### 新增文件 (npu/qnn/)

| 文件 | 作用 |
|------|------|
| `qnn_types.h` / `.cpp` | 公共类型定义：DSP版本、QNN数据类型、布局、算子类型 |
| `qnn_context.h` / `.cpp` | QNN 上下文管理：设备初始化、后端加载、内存分配、全局单例 |
| `qnn_tensor.h` / `.cpp` | Tensor 适配层：LocalImage Tensor ↔ QNN Tensor 转换、布局重排、权重预打包 |
| `qnn_op_map.h` / `.cpp` | 算子映射层：LocalImage IR Op → QNN Op、属性参数构建 |
| `qnn_graph_executor.h` / `.cpp` | 图执行器：单算子执行、子图融合执行、形状推理 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `npu/npu_backend.h` | 扩展枚举：QNN_Htp / QNN_Gpu / NNAPI，增加 DSP 版本和能力字段 |
| `npu/npu_backend.cpp` | 真实 SoC/DSP 检测 + QNN 后端探测 + 最低版本校验(v75) |
| `graph/graph_runtime.h` | 新增 `NpuBackend` 类，`Graph::execute` 增加 `preferNpu` 参数 |
| `graph/graph_runtime.cpp` | 实现 `NpuBackend`，三级执行路径：NPU → Vulkan → CPU |
| `CMakeLists.txt` | 新增 QNN 构建选项 (`LOCALIMAGE_QNN`) + SDK 路径配置 |

---

## 核心设计

### 1. 转接头原理 (OpMapper)

转接头的核心是 **LocalImage IR Op → QNN Op 的运行时映射**。

- **1:1 直接映射**（大部分算子）：
  - `Add → QNN Add`
  - `MatMul → QNN MatMul`
  - `Conv2D → QNN Conv2D`
  - `Softmax → QNN Softmax`
  - 等等共 25+ 种算子

- **分解映射**（一个IR算子拆成多个QNN算子）：
  - `Linear → MatMul + Add`
  - `GroupNorm → slice + mean + sub + mul + rsqrt + ...`

- **不支持的算子**：返回 "unsupported" 错误，触发回退

### 2. 三级 Fallback 机制

`Graph::execute()` 中每个算子按优先级尝试：

```
当前算子
  │
  ├─ 尝试 NPU ── 成功 → usedNpu=true
  │    │
  │    └─ 失败 ── 如果是"不支持的算子" → 继续
  │         └─ 如果是"执行错误" → 直接报错返回
  │
  ├─ 尝试 Vulkan ── 成功 → usedVulkan=true
  │    │
  │    └─ 失败 ── 如果是"不支持的算子" → 继续
  │         └─ 如果是"执行错误" → 直接报错返回
  │
  └─ 执行 CPU ── 成功 → usedCpu=true
       └─ 失败 → 报错返回
```

**重要原则**：只有明确的 "operator unavailable" 才触发回退。真实的硬件错误（设备丢失、内存不足、超时等）必须向上抛出，不能静默回退。

### 3. Tensor 零拷贝策略

`QnnTensorAdapter` 负责 Tensor 格式转换，遵循 **"零拷贝优先，按需转换"** 原则：

- **零拷贝**（最快）：当 LocalImage Tensor 是 F16/F32、NCHW 布局、连续内存时，直接把指针传给 QNN
- **布局转换**（次优）：权重或激活需要 block 格式（NCHW_C8/C16）时，在运行时重排
- **权重预打包缓存**（最优）：权重的布局转换结果缓存到磁盘，下次加载直接 mmap

### 4. 权重预打包缓存 (WeightPrepacker)

模型权重在首次加载时做一次 NPU 友好的布局重排，结果缓存到磁盘：

- **Cache Key** = `model_hash + dsp_version + tensor_name_hash`
- **缓存格式**：原始二进制（可直接 mmap）
- **缓存目录**：应用 cache 目录下的 `npu_weights/`
- **过期策略**：model hash 变化时自动失效

这确保了：
- 第一次加载慢（需要重排）
- 第二次及以后加载 = 零拷贝，和原始模型一样快

### 5. DSP 版本检测与最低要求

`BackendProbe::detect()` 执行完整的检测流程：

```
1. 读系统属性 → SoC 型号 (SM8650 / SM8750 / ...)
2. SoC → DSP 版本映射 (v75 / v79 / v81)
3. 最低版本校验 (≥ v75 = 8 Gen3)
   ↓ 不满足 → available=false，返回原因
4. 加载 QNN HTP 后端 + 对应版本的 skel 库
5. 创建设备 + 上下文
6. 探测支持的算子列表
7. 返回完整 Capabilities
```

**骁龙 8 Gen2 (v73)**：检测到但因低于最低要求而拒绝，提示用户设备不支持。

---

## 构建配置

### 默认构建（无 QNN SDK）

```bash
./gradlew assembleDebug
```

- NPU 代码编译进 .so
- 运行时 NPU 报告 "不可用"（QNN SDK not compiled in）
- 不影响 CPU / Vulkan 功能

### 启用 QNN NPU

```bash
# 设置 QNN SDK 路径
export QNN_SDK_ROOT=/path/to/qcom/qnn/sdk

# 构建时传入 CMake 参数
./gradlew assembleDebug \
  -DLOCALIMAGE_CMAKE_ARGS="-DLOCALIMAGE_QNN=ON -DQNN_SDK_ROOT=$QNN_SDK_ROOT"
```

或者在 `app/build.gradle.kts` 中配置：

```kotlin
externalNativeBuild {
    cmake {
        cppFlags += listOf("-std=c++20", "-fno-exceptions", "-fno-rtti")
        arguments += listOf(
            "-DLOCALIMAGE_QNN=ON",
            "-DQNN_SDK_ROOT=/path/to/qnn/sdk"
        )
    }
}
```

### 完全排除 NPU 代码（最小体积）

```cmake
-DLOCALIMAGE_NO_NPU=ON
```

---

## 条件编译宏

| 宏 | 含义 | NPU 行为 |
|----|------|----------|
| `LOCALIMAGE_QNN` | QNN SDK 可用，完全启用 | 真实 NPU 执行 |
| *(默认)* | 无 SDK，代码仍编译 | Stub 实现，运行时报不可用 |
| `LOCALIMAGE_NO_NPU` | 完全排除 NPU 代码 | 编译时移除所有 NPU 相关代码 |

这样设计的好处：
- 开发者没有 QNN SDK 也能正常编译项目
- CI 构建不需要 QNN SDK
- 真机上有 SDK 时自动点亮 NPU
- 包体积敏感的构建可以用 `LOCALIMAGE_NO_NPU` 移除

---

## 算子支持清单

### 直接映射 (1:1)

| LocalImage Op | QNN Op | HTP 支持 |
|---|---|---|
| Add | ElementWise Add | ✅ |
| Sub | ElementWise Sub | ✅ |
| Mul | ElementWise Mul | ✅ |
| Div | ElementWise Div | ✅ |
| MatMul | MatMul | ✅ |
| BatchedMatMul | BatchMatMul | ✅ |
| Conv2D | Conv2D | ✅ |
| Softmax | Softmax | ✅ |
| LayerNorm | LayerNorm | ✅ |
| RMSNorm | RMSNorm | ✅ |
| GroupNorm | GroupNorm | ✅ |
| SiLU | SiLU | ✅ |
| GELU | GELU | ✅ |
| Exp | Exp | ✅ |
| Sqrt | Sqrt | ✅ |
| Rsqrt | Rsqrt | ✅ |
| Clamp | Clamp | ✅ |
| Reshape | Reshape | ✅ (零开销) |
| Transpose | Transpose | ✅ |
| Slice | Slice | ✅ |
| Concat | Concat | ✅ |
| Broadcast | Broadcast | ✅ |
| Upsample | Resize (Nearest) | ✅ |

### 分解映射

| LocalImage Op | QNN 分解 |
|---|---|
| Linear | MatMul + Add |
| RoPE | Slice × 2 + Sin + Cos + Mul × 2 + Add |
| Attention | MatMul × 3 + Softmax + 辅助 |

### 不支持 / 回退 CPU

- 自定义激活函数（非标准）
- 复杂动态形状算子
- 非 4D 的特殊卷积变体

---

## 性能预期

相对于 Vulkan GPU，Hexagon NPU (8 Gen3+) 的优势：

| 维度 | Vulkan GPU | NPU (HTP) | 提升 |
|------|-----------|-----------|------|
| MatMul (大矩阵) | 基准 | ~2-4x | ✅ |
| Conv2D | 基准 | ~1.5-3x | ✅ |
| 功耗 | 基准 | ~30-50% 更低 | ✅ |
| 小算子延迟 | 好 | 一般 (RPC开销) | ⚠️ |
| 显存带宽 | 高 | 中（共享内存） | ⚠️ |

**最佳实践**：
- 大算子（MatMul/Conv2D/Attention）跑 NPU
- 小算子（逐元素运算）留在 GPU resident 链上
- 用子图融合减少 NPU dispatch 次数

---

## 后续优化方向

1. **子图融合执行**：把连续多个 NPU 算子合并成一个 QNN Graph，减少 dispatch 开销
2. **INT8 量化**：利用 NPU 的 INT8 性能优势，需要校准工具
3. **DMA 管道**：权重预加载 + 计算重叠
4. **NPU resident**：类似 Vulkan resident，中间张量留在 NPU 内存
5. **多 NPU 核心调度**：8 Elite 以上多 Tensor Core 并行
6. **动态形状优化**：减少 QNN graph rebuild 开销

---

## 验证步骤

接入 QNN SDK 后的验证流程：

```
1. 编译 APK (LOCALIMAGE_QNN=ON)
2. 安装到 8 Gen3 及以上设备
3. 运行 NPU capability probe
   → 预期: available=true, DSP=v75+, backend=QNN_Htp
4. 运行单算子测试 (MatMul / Conv2D)
   → 预期: 输出正确，性能优于 Vulkan
5. 运行完整 SDXL UNet
   → 预期: NPU+Vulkan+CPU 混合执行，整体速度提升
6. 验证 fallback
   → 不支持的算子正确回退到 Vulkan/CPU
```
