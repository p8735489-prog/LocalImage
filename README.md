# LocalImage

Android 本地 Neural Runtime / AI 绘画项目。

## 当前真实状态

- Tensor / SafeTensors / LocalImage IR：已实现
- CPU Reference Operators：已实现
- Graph validation / topological execution：已实现
- Vulkan Context / 基础 Compute：已实现
- SDXL / SD3 / FLUX / Anima E2E：未声明支持
- NPU / QNN：只有真实 SDK 与 capability 可用时才启用
- 模型支持以真实权重加载、Graph 执行和输出验证为准，不以 Detector 或 UI 为准

## Android 构建

AGP 9.2.1 / Gradle 9.4.1 / Kotlin 2.2.10 / compileSdk 36 / minSdk 29 / NDK 29.0.14206865 / CMake 3.22.1 / arm64-v8a。

```bash
./gradlew assembleDebug
./gradlew assembleRelease
```

## GitHub Actions

每次 `main` push、PR 和手动触发都会构建 APK 并上传 Actions Artifact。

推送版本 tag：

```bash
git tag v0.10.3
git push origin v0.10.3
```

`v*` tag 会自动创建 GitHub Release，并上传 debug APK、unsigned release APK 和 SHA256 校验文件。

未配置正式签名密钥时，release APK 保持 unsigned；debug APK 可用于测试。正式签名密钥禁止提交到仓库。

## Runtime 原则

Vulkan failure 与 unsupported 严格区分。GPU 执行失败不会静默改写成 CPU 成功。
