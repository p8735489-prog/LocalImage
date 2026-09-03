package com.haobai.localimage.runtime

import com.haobai.localimage.NativeRuntime

enum class ModelArchitecture(val id: Int, val displayName: String, val recommendedWidth: Int, val recommendedHeight: Int) {
    SD15(0, "SD 1.5", 512, 512),
    SD2(1, "SD 2.x", 768, 768),
    SDXL(2, "SDXL", 1024, 1024),
    SD3(3, "SD 3", 1080, 1080),
    SD35(4, "SD 3.5", 1080, 1080),
    FLUX(5, "FLUX", 1080, 1080),
    ANIMA(6, "Anima", 1080, 1080),
    UNKNOWN(7, "Unknown", 512, 512);

    companion object {
        fun fromId(id: Int): ModelArchitecture = entries.firstOrNull { it.id == id } ?: UNKNOWN
    }
}

enum class RuntimeDType(val id: Int) { F32(0), F16(1), BF16(2) }

enum class AspectRatio(val label: String, val w: Int, val h: Int) {
    SQUARE("1:1", 1, 1),
    PORTRAIT_34("3:4", 3, 4),
    LANDSCAPE_43("4:3", 4, 3),
    LANDSCAPE_169("16:9", 16, 9),
    PORTRAIT_916("9:16", 9, 16)
}

data class GenerationRequest(
    val prompt: String,
    val negativePrompt: String = "",
    val width: Int? = null,
    val height: Int? = null,
    val aspectRatio: AspectRatio? = null,
    val steps: Int? = null,
    val cfgScale: Float? = null,
    val seed: Long? = null,
    val scheduler: String? = null,
    val batchSize: Int = 1,
    val denoiseStrength: Float? = null,
    val modelId: String
)

data class ResolvedResolution(
    val requestedWidth: Int,
    val requestedHeight: Int,
    /** Actual dimensions passed to the diffusion graph. */
    val width: Int,
    val height: Int,
    /** VAE-safe final image dimensions. */
    val outputWidth: Int,
    val outputHeight: Int,
    val latentWidth: Int,
    val latentHeight: Int,
    val alignment: Int,
    val estimatedPeakBytes: Long,
    val memoryBudgetBytes: Long,
    val adjusted: Boolean,
    val memoryLimited: Boolean,
    val reason: String,
    val warning: String?
)

object ResolutionRuntime {
    fun resolve(
        architecture: ModelArchitecture,
        width: Int = 0,
        height: Int = 0,
        dtype: RuntimeDType = RuntimeDType.F16,
        backend: Int = 0,
        weightBytes: Long = 0,
        availableCpuBytes: Long = 0,
        availableGpuBytes: Long = 0,
        availableNpuBytes: Long = 0
    ): ResolvedResolution {
        require(width >= 0 && height >= 0) { "width/height must be non-negative" }
        require(weightBytes >= 0 && availableCpuBytes >= 0 && availableGpuBytes >= 0 && availableNpuBytes >= 0)
        val raw = NativeRuntime.nativeResolveResolution(
            architecture.id, width, height, dtype.id, backend, weightBytes,
            availableCpuBytes, availableGpuBytes, availableNpuBytes
        )
        val map = raw.split(';').associate {
            val p = it.indexOf('=')
            if (p < 0) it to "" else it.substring(0, p) to it.substring(p + 1)
        }
        fun size(key: String): Pair<Int, Int> {
            val value = map[key] ?: error("Runtime response missing $key")
            val x = value.indexOf('x')
            require(x > 0) { "Invalid runtime resolution: $value" }
            return value.substring(0, x).toInt() to value.substring(x + 1).toInt()
        }
        val requested = size("requested")
        val execution = size("resolved")
        val output = size("output")
        val latent = size("latent")
        return ResolvedResolution(
            requested.first, requested.second,
            execution.first, execution.second,
            output.first, output.second,
            latent.first, latent.second,
            map.getValue("alignment").toInt(),
            map.getValue("estimatedPeakBytes").toLong(),
            map.getValue("memoryBudgetBytes").toLong(),
            map.getValue("adjusted") == "true",
            map.getValue("memoryLimited") == "true",
            map.getValue("reason"),
            map["warning"]?.takeIf { it.isNotBlank() }
        )
    }
}
