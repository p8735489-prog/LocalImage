package com.haobai.localimage

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DeveloperMode
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Smartphone
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.haobai.localimage.runtime.AspectRatio
import com.haobai.localimage.runtime.ModelArchitecture
import com.haobai.localimage.runtime.ResolutionRuntime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

// ============================================================
// Pixel / Retro Color Palette
// ============================================================

// 像素风配色：霓虹紫 + 电光青 + 像素粉
private val PixelPrimary = Color(0xFF8B5CF6)      // 霓虹紫
private val PixelOnPrimary = Color(0xFFFFFFFF)
private val PixelPrimaryContainer = Color(0xFF2D1B69) // 深紫容器
private val PixelOnPrimaryContainer = Color(0xFFE9D5FF)

private val PixelSecondary = Color(0xFF06B6D4)     // 电光青
private val PixelOnSecondary = Color(0xFF000000)
private val PixelSecondaryContainer = Color(0xFF083344)
private val PixelOnSecondaryContainer = Color(0xFFA5F3FC)

private val PixelTertiary = Color(0xFFF472B6)      // 像素粉
private val PixelTertiaryContainer = Color(0xFF500724)

private val PixelSurface = Color(0xFF0F0A1F)       // 深紫黑底
private val PixelSurfaceVariant = Color(0xFF1E1B4B) // 靛蓝灰
private val PixelOnSurface = Color(0xFFE0E7FF)
private val PixelOnSurfaceVariant = Color(0xFF94A3B8)
private val PixelOutline = Color(0xFF6366F1)
private val PixelOutlineVariant = Color(0xFF4338CA)

private val PixelError = Color(0xFFF87171)
private val PixelSuccess = Color(0xFF34D399)
private val PixelWarning = Color(0xFFFBBF24)

// 亮色主题
private val PixelLightPrimary = Color(0xFF7C3AED)
private val PixelLightSurface = Color(0xFFFAFAFF)
private val PixelLightOnSurface = Color(0xFF1E1B4B)

private val LightPixelColors = lightColorScheme(
    primary = PixelLightPrimary,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFEDE9FE),
    onPrimaryContainer = Color(0xFF4C1D95),
    secondary = PixelSecondary,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFCFFAFE),
    onSecondaryContainer = Color(0xFF083344),
    tertiary = PixelTertiary,
    tertiaryContainer = Color(0xFFFCE7F3),
    surface = PixelLightSurface,
    onSurface = PixelLightOnSurface,
    surfaceVariant = Color(0xFFEDE9FE),
    onSurfaceVariant = Color(0xFF4338CA),
    outline = Color(0xFFA78BFA),
    outlineVariant = Color(0xFFDDD6FE),
    error = Color(0xFFEF4444),
)

private val DarkPixelColors = darkColorScheme(
    primary = PixelPrimary,
    onPrimary = PixelOnPrimary,
    primaryContainer = PixelPrimaryContainer,
    onPrimaryContainer = PixelOnPrimaryContainer,
    secondary = PixelSecondary,
    onSecondary = PixelOnSecondary,
    secondaryContainer = PixelSecondaryContainer,
    onSecondaryContainer = PixelOnSecondaryContainer,
    tertiary = PixelTertiary,
    tertiaryContainer = PixelTertiaryContainer,
    surface = PixelSurface,
    onSurface = PixelOnSurface,
    surfaceVariant = PixelSurfaceVariant,
    onSurfaceVariant = PixelOnSurfaceVariant,
    outline = PixelOutline,
    outlineVariant = PixelOutlineVariant,
    error = PixelError,
    onError = Color(0xFF000000),
)

// 像素风格形状 — 硬切角 + 略微圆角，模拟像素感
private val PixelShapeSmall = RoundedCornerShape(4.dp)
private val PixelShapeMedium = RoundedCornerShape(8.dp)
private val PixelShapeLarge = RoundedCornerShape(12.dp)
private val PixelShapeXL = RoundedCornerShape(16.dp)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { LocalImageApp() }
    }
}

@Composable
private fun LocalImageTheme(content: @Composable () -> Unit) {
    val context = LocalContext.current
    val dark = androidx.compose.foundation.isSystemInDarkTheme()
    // 使用自定义像素配色方案，深色模式下效果最佳
    val colors = if (dark) DarkPixelColors else LightPixelColors
    MaterialTheme(
        colorScheme = colors,
        typography = MaterialTheme.typography.let {
            it.copy(
                displayLarge = it.displayLarge.copy(fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold),
                headlineLarge = it.headlineLarge.copy(fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold),
                headlineMedium = it.headlineMedium.copy(fontWeight = FontWeight.SemiBold),
                titleLarge = it.titleLarge.copy(fontWeight = FontWeight.SemiBold),
                titleMedium = it.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                bodyMedium = it.bodyMedium.copy(fontSize = 14.sp),
                labelLarge = it.labelLarge.copy(fontWeight = FontWeight.SemiBold),
            )
        },
        shapes = MaterialTheme.shapes.let {
            it.copy(
                small = PixelShapeSmall,
                medium = PixelShapeMedium,
                large = PixelShapeLarge,
                extraLarge = PixelShapeXL,
            )
        },
        content = content
    )
}

// 像素风格装饰 — 顶部渐变条
@Composable
private fun PixelGlowHeader(modifier: Modifier = Modifier) {
    Box(
        modifier
            .height(3.dp)
            .fillMaxWidth()
            .background(
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.primary,
                        MaterialTheme.colorScheme.secondary,
                        MaterialTheme.colorScheme.tertiary,
                        MaterialTheme.colorScheme.primary,
                    )
                )
            )
    )
}

// 像素卡片 — 带发光边框效果
@Composable
private fun PixelCard(
    modifier: Modifier = Modifier,
    glow: Boolean = false,
    content: @Composable () -> Unit
) {
    val baseColor = MaterialTheme.colorScheme.surfaceVariant
    val glowColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)

    Card(
        modifier = modifier
            .then(
                if (glow) Modifier.border(
                    width = 1.dp,
                    color = glowColor,
                    shape = MaterialTheme.shapes.large
                ) else Modifier
            ),
        shape = MaterialTheme.shapes.large,
        colors = CardDefaults.cardColors(
            containerColor = baseColor,
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = if (glow) 4.dp else 2.dp),
        content = {
            Column(
                Modifier
                    .fillMaxWidth()
                    .padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
                content = { content() }
            )
        }
    )
}

// 像素按钮样式
@Composable
private fun PixelButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    content: @Composable () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = modifier,
        enabled = enabled,
        shape = MaterialTheme.shapes.medium,
        colors = ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.primary,
            contentColor = MaterialTheme.colorScheme.onPrimary,
        ),
        elevation = ButtonDefaults.buttonElevation(
            defaultElevation = 2.dp,
            pressedElevation = 0.dp,
        ),
        content = { content() }
    )
}

// 状态徽章
@Composable
private fun StatusBadge(text: String, success: Boolean = true) {
    val bg = if (success) PixelSuccess.copy(alpha = 0.15f) else PixelError.copy(alpha = 0.15f)
    val fg = if (success) PixelSuccess else PixelError
    AssistChip(
        onClick = {},
        label = { Text(text, fontWeight = FontWeight.SemiBold, fontSize = 12.sp) },
        shape = MaterialTheme.shapes.small,
        colors = androidx.compose.material3.AssistChipDefaults.assistChipColors(
            containerColor = bg,
            labelColor = fg,
        ),
        border = BorderStroke(1.dp, fg.copy(alpha = 0.5f))
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LocalImageApp() {
    LocalImageTheme {
        var page by remember { mutableIntStateOf(0) }
        var mainTab by remember { mutableIntStateOf(0) }
        var handle by remember { mutableLongStateOf(0L) }
        var fileName by remember { mutableStateOf<String?>(null) }
        var fileSize by remember { mutableLongStateOf(-1L) }
        var tensorCount by remember { mutableLongStateOf(0L) }
        var tensorInfo by remember { mutableStateOf("") }
        var modelHash by remember { mutableStateOf("") }
        var cacheKey by remember { mutableStateOf("") }
        var architectureInfo by remember { mutableStateOf("") }
        var resolutionInfo by remember { mutableStateOf("") }
        var status by remember { mutableStateOf("尚未选择模型") }
        var runtimeInfo by remember { mutableStateOf("尚未查询 Runtime") }
        var gpuTest by remember { mutableStateOf("") }
        var gpuTesting by remember { mutableStateOf(false) }
        var tensorTest by remember { mutableStateOf("") }
        var npuInfo by remember { mutableStateOf("") }
        var npuTesting by remember { mutableStateOf(false) }
        var loading by remember { mutableStateOf(false) }
        var prompt by remember { mutableStateOf("") }
        var negativePrompt by remember { mutableStateOf("") }
        var advanced by remember { mutableStateOf(false) }
        var aspect by remember { mutableStateOf(AspectRatio.SQUARE) }
        var custom by remember { mutableStateOf(false) }
        var customWidth by remember { mutableStateOf("1024") }
        var customHeight by remember { mutableStateOf("1024") }
        var steps by remember { mutableIntStateOf(20) }
        var cfg by remember { mutableStateOf(7f) }
        var seed by remember { mutableStateOf<Long?>(null) }
        var scheduler by remember { mutableStateOf("Euler") }
        var generateMessage by remember { mutableStateOf<String?>(null) }
        val scope = rememberCoroutineScope()
        val context = LocalContext.current

        val architecture = remember(architectureInfo) {
            ModelArchitecture.fromId(architectureInfo.substringBefore('|').toIntOrNull() ?: 7)
        }

        fun closeModel() {
            if (handle != 0L) NativeRuntime.nativeCloseSafeTensor(handle)
            handle = 0L
            fileName = null
            fileSize = -1L
            tensorCount = 0L
            tensorInfo = ""
            modelHash = ""
            cacheKey = ""
            architectureInfo = ""
            resolutionInfo = ""
            status = "尚未选择模型"
        }

        fun applyRecommendedResolution() {
            custom = false
            aspect = AspectRatio.SQUARE
            val model = architecture
            customWidth = model.recommendedWidth.toString()
            customHeight = model.recommendedHeight.toString()
        }

        DisposableEffect(Unit) {
            onDispose { if (handle != 0L) NativeRuntime.nativeCloseSafeTensor(handle) }
        }
        BackHandler(enabled = page != 0) { page = 0 }

        val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri == null) return@rememberLauncherForActivityResult
            loading = true
            try {
                try {
                    context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                } catch (_: SecurityException) {}
                if (handle != 0L) NativeRuntime.nativeCloseSafeTensor(handle)
                val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: error("无法打开模型文件")
                val fd = pfd.detachFd()
                handle = NativeRuntime.nativeOpenSafeTensor(fd)
                fileName = queryDisplayName(context, uri) ?: "model.safetensors"
                fileSize = NativeRuntime.nativeGetFileSize(handle)
                tensorCount = NativeRuntime.nativeGetTensorCount(handle)
                tensorInfo = if (tensorCount > 0) NativeRuntime.nativeGetTensorInfo(handle, 0) else "No tensors"
                modelHash = NativeRuntime.nativeGetModelHash(handle)
                cacheKey = NativeRuntime.nativeGetCacheKey(handle)
                architectureInfo = NativeRuntime.nativeGetModelArchitecture(handle)
                val detected = ModelArchitecture.fromId(architectureInfo.substringBefore('|').toIntOrNull() ?: 7)
                customWidth = detected.recommendedWidth.toString()
                customHeight = detected.recommendedHeight.toString()
                resolutionInfo = resolveResolution(context, detected, detected.recommendedWidth, detected.recommendedHeight, fileSize)
                status = "✓ SafeTensors / mmap 已验证"
                page = 0
            } catch (t: Throwable) {
                if (handle != 0L) NativeRuntime.nativeCloseSafeTensor(handle)
                handle = 0L
                status = "✕ ${t.message ?: "模型加载失败"}"
            } finally {
                loading = false
            }
        }

        Surface(Modifier.fillMaxSize()) {
            Scaffold(
                contentWindowInsets = WindowInsets.safeDrawing,
                topBar = {
                    Column {
                        CenterAlignedTopAppBar(
                            title = {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(
                                        "LocalImage",
                                        fontWeight = FontWeight.Bold,
                                        fontFamily = FontFamily.Monospace,
                                        letterSpacing = 0.5.sp
                                    )
                                    Spacer(Modifier.width(8.dp))
                                    Box(
                                        Modifier
                                            .size(8.dp)
                                            .background(
                                                color = PixelSuccess,
                                                shape = RoundedCornerShape(2.dp)
                                            )
                                    )
                                }
                            },
                            colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                                containerColor = MaterialTheme.colorScheme.surface,
                                titleContentColor = MaterialTheme.colorScheme.onSurface,
                            ),
                            actions = {
                                IconButton(onClick = { page = 2 }) {
                                    Icon(Icons.Default.Settings, contentDescription = "Runtime 设置")
                                }
                            }
                        )
                        PixelGlowHeader()
                    }
                },
                bottomBar = {
                    NavigationBar(
                        windowInsets = WindowInsets.navigationBars,
                        containerColor = MaterialTheme.colorScheme.surface,
                    ) {
                        NavigationBarItem(
                            selected = page == 0,
                            onClick = { page = 0 },
                            icon = { Icon(Icons.Default.AutoAwesome, null) },
                            label = { Text("生成", fontSize = 11.sp, fontWeight = FontWeight.Medium) }
                        )
                        NavigationBarItem(
                            selected = page == 1,
                            onClick = { page = 1 },
                            icon = { Icon(Icons.Default.Memory, null) },
                            label = { Text("模型", fontSize = 11.sp, fontWeight = FontWeight.Medium) }
                        )
                        NavigationBarItem(
                            selected = page == 2,
                            onClick = { page = 2 },
                            icon = { Icon(Icons.Default.Psychology, null) },
                            label = { Text("NPU", fontSize = 11.sp, fontWeight = FontWeight.Medium) }
                        )
                        NavigationBarItem(
                            selected = page == 3,
                            onClick = { page = 3 },
                            icon = { Icon(Icons.Default.DeveloperMode, null) },
                            label = { Text("Runtime", fontSize = 11.sp, fontWeight = FontWeight.Medium) }
                        )
                    }
                }
            ) { padding ->
                when (page) {
                    0 -> GenerationPage(
                        mainTab, { mainTab = it }, fileName, architecture, status, prompt, { prompt = it },
                        negativePrompt, { negativePrompt = it }, aspect, { aspect = it; custom = false },
                        custom, { custom = it }, customWidth, { customWidth = it.filter(Char::isDigit) },
                        customHeight, { customHeight = it.filter(Char::isDigit) }, steps, { steps = it },
                        cfg, { cfg = it }, seed, { seed = it }, scheduler, { scheduler = it },
                        onAdvanced = { advanced = true },
                        onGenerate = {
                            if (handle == 0L) generateMessage = "请先导入一个真实模型"
                            else generateMessage = "当前工程的生成图链路尚未接通；Runtime 不会伪造图片输出。模型 Inspector、ResolutionPolicy 和 CPU/Vulkan/NPU Runtime 已可执行。"
                        },
                        onImport = { picker.launch(arrayOf("application/octet-stream", "application/x-safetensors", "*/*")) },
                        loading = loading,
                        resolutionInfo = resolutionInfo
                    )
                    1 -> ModelPage(
                        fileName, fileSize, tensorCount, status, tensorInfo, modelHash, cacheKey, architectureInfo,
                        resolutionInfo, handle != 0L,
                        onImport = { picker.launch(arrayOf("application/octet-stream", "application/x-safetensors", "*/*")) },
                        onResolveResolution = { w, h ->
                            resolutionInfo = resolveResolution(context, architecture, w, h, fileSize)
                        },
                        onValidate = {
                            if (handle != 0L) try {
                                NativeRuntime.nativeValidateModel(handle)
                                status = "✓ 模型验证通过"
                            } catch (t: Throwable) { status = "✕ ${t.message}" }
                        },
                        onClose = { closeModel() }
                    )
                    2 -> NpuPage(
                        info = npuInfo,
                        testing = npuTesting,
                        onRefresh = {
                            npuTesting = true
                            npuInfo = "正在检测 NPU 能力…"
                            scope.launch {
                                val result = withContext(Dispatchers.Default) {
                                    try { NativeRuntime.nativeGetNpuInfo() }
                                    catch (t: Throwable) { "NPU 检测失败\n${t.message ?: "Native error"}" }
                                }
                                npuInfo = result
                                npuTesting = false
                            }
                        }
                    )
                    else -> RuntimePage(
                        runtimeInfo, gpuTest, gpuTesting,
                        onRefresh = { runtimeInfo = try { NativeRuntime.nativeGetDeviceInfo() } catch (t: Throwable) { "✕ ${t.message}" } },
                        onGpuTest = {
                            gpuTesting = true
                            gpuTest = "正在运行 Vulkan Compute…"
                            scope.launch {
                                val result = withContext(Dispatchers.Default) {
                                    try { NativeRuntime.nativeRunVulkanComputeTest() }
                                    catch (t: Throwable) { "Vulkan Compute 失败\n${t.message ?: "Native error"}" }
                                }
                                gpuTest = result
                                gpuTesting = false
                            }
                        },
                        tensorTest = tensorTest,
                        onTensorTest = { tensorTest = NativeRuntime.nativeRunTensorTest(context.cacheDir.absolutePath) }
                    )
                }
            }
        }

        if (advanced) {
            AdvancedParametersDialog(
                architecture = architecture,
                aspect = aspect,
                custom = custom,
                width = customWidth,
                height = customHeight,
                steps = steps,
                cfg = cfg,
                seed = seed,
                scheduler = scheduler,
                onAspect = { aspect = it; custom = false },
                onCustom = { custom = it },
                onWidth = { customWidth = it.filter(Char::isDigit) },
                onHeight = { customHeight = it.filter(Char::isDigit) },
                onSteps = { steps = it },
                onCfg = { cfg = it },
                onSeed = { seed = it },
                onScheduler = { scheduler = it },
                onReset = { applyRecommendedResolution(); steps = if (architecture == ModelArchitecture.SD15) 20 else 20; cfg = if (architecture == ModelArchitecture.SD15) 7f else 7f; scheduler = if (architecture == ModelArchitecture.SD3 || architecture == ModelArchitecture.SD35 || architecture == ModelArchitecture.FLUX || architecture == ModelArchitecture.ANIMA) "FlowMatch" else "Euler"; seed = null },
                onDismiss = { advanced = false },
                onResolve = {
                    val (w, h) = if (custom) customWidth.toIntOrNull() to customHeight.toIntOrNull()
                    else aspect.w * architecture.recommendedWidth / aspect.w to aspect.h * architecture.recommendedHeight / aspect.w
                    if (w != null && h != null && w > 0 && h > 0) {
                        resolutionInfo = resolveResolution(context, architecture, w, h, fileSize)
                    }
                    advanced = false
                }
            )
        }

        generateMessage?.let { message ->
            AlertDialog(
                onDismissRequest = { generateMessage = null },
                confirmButton = { TextButton(onClick = { generateMessage = null }) { Text("知道了") } },
                title = { Text("Runtime 状态") },
                text = { Text(message) },
                shape = MaterialTheme.shapes.large,
            )
        }
    }
}

// ============================================================
// Generation Page
// ============================================================

@Composable
private fun GenerationPage(
    tab: Int, onTab: (Int) -> Unit,
    fileName: String?, architecture: ModelArchitecture, status: String,
    prompt: String, onPrompt: (String) -> Unit,
    negativePrompt: String, onNegative: (String) -> Unit,
    aspect: AspectRatio, onAspect: (AspectRatio) -> Unit,
    custom: Boolean, onCustom: (Boolean) -> Unit,
    width: String, onWidth: (String) -> Unit,
    height: String, onHeight: (String) -> Unit,
    steps: Int, onSteps: (Int) -> Unit,
    cfg: Float, onCfg: (Float) -> Unit,
    seed: Long?, onSeed: (Long?) -> Unit,
    scheduler: String, onScheduler: (String) -> Unit,
    onAdvanced: () -> Unit, onGenerate: () -> Unit,
    onImport: () -> Unit, loading: Boolean, resolutionInfo: String
) {
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Spacer(Modifier.height(4.dp))

        TabRow(
            selectedTabIndex = tab,
            containerColor = Color.Transparent,
            indicator = { tabPositions ->
                TabRowDefaults.PrimaryIndicator(
                    modifier = Modifier
                        .width(48.dp)
                        .align(Alignment.CenterHorizontally),
                    width = 48.dp,
                    height = 3.dp,
                    shape = RoundedCornerShape(topStart = 3.dp, topEnd = 3.dp)
                )
            }
        ) {
            listOf("✨ 提示词", "🖼 生成结果", "📜 历史").forEachIndexed { index, title ->
                Tab(
                    selected = tab == index,
                    onClick = { onTab(index) },
                    text = { Text(title, fontSize = 13.sp, fontWeight = FontWeight.Medium) },
                    selectedContentColor = MaterialTheme.colorScheme.primary,
                    unselectedContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        when (tab) {
            0 -> {
                // Model Card
                PixelCard(glow = fileName != null) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Default.Image,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(28.dp)
                        )
                        Spacer(Modifier.width(12.dp))
                        Column(Modifier.weight(1f)) {
                            Text("当前模型", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text(
                                fileName ?: "未加载模型",
                                style = MaterialTheme.typography.titleLarge,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                    if (fileName != null) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            AssistChip(
                                onClick = {},
                                label = { Text(architecture.displayName, fontSize = 12.sp) },
                                shape = MaterialTheme.shapes.small
                            )
                            AssistChip(
                                onClick = {},
                                label = { Text("${architecture.recommendedWidth} × ${architecture.recommendedHeight}", fontSize = 12.sp) },
                                shape = MaterialTheme.shapes.small
                            )
                            StatusBadge("就绪", success = true)
                        }
                    } else {
                        PixelButton(onClick = onImport, Modifier.fillMaxWidth(), !loading) {
                            if (loading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                                Spacer(Modifier.width(8.dp))
                                Text("加载中…")
                            } else {
                                Text("导入模型")
                            }
                        }
                    }
                }

                // Prompt Input
                PixelCard {
                    Text("正面提示词", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(
                        value = prompt, onValueChange = onPrompt,
                        modifier = Modifier.fillMaxWidth(),
                        minLines = 5,
                        placeholder = { Text("例如：1girl, blue hair, detailed illustration, masterpiece") },
                        shape = MaterialTheme.shapes.medium,
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = MaterialTheme.colorScheme.primary,
                            unfocusedBorderColor = MaterialTheme.colorScheme.outlineVariant,
                        )
                    )
                    Text(
                        "${prompt.length} 字符",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                // Negative Prompt
                PixelCard {
                    Text("负面提示词", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    OutlinedTextField(
                        value = negativePrompt, onValueChange = onNegative,
                        modifier = Modifier.fillMaxWidth(),
                        minLines = 2,
                        placeholder = { Text("例如：blurry, low quality, watermark, bad anatomy") },
                        shape = MaterialTheme.shapes.medium,
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = MaterialTheme.colorScheme.secondary,
                            unfocusedBorderColor = MaterialTheme.colorScheme.outlineVariant,
                        )
                    )
                }

                // Quick actions
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(
                        onClick = {},
                        Modifier.weight(1f),
                        shape = MaterialTheme.shapes.medium,
                    ) {
                        Icon(Icons.Default.AddPhotoAlternate, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("图生图", fontSize = 13.sp)
                    }
                    OutlinedButton(
                        onClick = onAdvanced,
                        Modifier.weight(1f),
                        shape = MaterialTheme.shapes.medium,
                    ) {
                        Icon(Icons.Default.Tune, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("高级设置", fontSize = 13.sp)
                    }
                }

                // Resolution info
                if (resolutionInfo.isNotBlank()) {
                    PixelCard {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                Icons.Default.Smartphone,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.secondary,
                                modifier = Modifier.size(20.dp)
                            )
                            Spacer(Modifier.width(8.dp))
                            Text("分辨率规划", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        }
                        Text(
                            resolutionInfo,
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            lineHeight = 18.sp
                        )
                    }
                }

                // Generate Button
                PixelButton(
                    onClick = onGenerate,
                    enabled = fileName != null && prompt.isNotBlank() && !loading,
                    modifier = Modifier.fillMaxWidth().height(52.dp)
                ) {
                    Text("✨ 生成图像", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                }

                Text(
                    status,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(16.dp))
            }
            1 -> EmptyState("生成结果", "真实 Runtime 产生 PNG 后会显示在这里。", "🖼")
            2 -> EmptyState("历史记录", "生成记录会在真实生成链路接通后保存。", "📜")
        }
    }
}

// ============================================================
// NPU Page
// ============================================================

@Composable
private fun NpuPage(
    info: String,
    testing: Boolean,
    onRefresh: () -> Unit
) {
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Spacer(Modifier.height(4.dp))
        Text("NPU 加速", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        PixelCard(glow = true) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier
                        .size(44.dp)
                        .clip(MaterialTheme.shapes.medium)
                        .background(MaterialTheme.colorScheme.primaryContainer),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Default.Psychology,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.size(26.dp)
                    )
                }
                Spacer(Modifier.width(14.dp))
                Column(Modifier.weight(1f)) {
                    Text("Hexagon NPU", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text(
                        "Qualcomm QNN HTP Backend",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            Spacer(Modifier.height(6.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                StatusBadge("8 Gen3+", success = true)
                StatusBadge("v75 / v79 / v81", success = true)
                AssistChip(
                    onClick = {},
                    label = { Text("转接头架构", fontSize = 11.sp) },
                    shape = MaterialTheme.shapes.small
                )
            }
        }

        PixelCard {
            Text("NPU 能力检测", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(
                if (info.isBlank()) "点击下方按钮检测设备 NPU 能力" else info,
                style = MaterialTheme.typography.bodyMedium,
                fontFamily = FontFamily.Monospace,
                color = if (info.isBlank()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.onSurface,
                lineHeight = 18.sp
            )
            Spacer(Modifier.height(4.dp))
            PixelButton(
                onClick = onRefresh,
                enabled = !testing,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (testing) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                    Spacer(Modifier.width(8.dp))
                    Text("检测中…")
                } else {
                    Text("检测 NPU 能力")
                }
            }
        }

        PixelCard {
            Text("技术说明", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                InfoRow("架构", "运行时转接头 (Runtime Adapter)")
                InfoRow("底层 SDK", "Qualcomm QNN HTP")
                InfoRow("最低要求", "Snapdragon 8 Gen3 (Hexagon v75)")
                InfoRow("算子支持", "25+ 种核心算子")
                InfoRow("Fallback", "NPU → Vulkan → CPU")
                InfoRow("模型格式", "limodel (mmap 零拷贝)")
            }
        }

        PixelCard {
            Text("性能预期", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                PerformanceRow("MatMul", "~2-4x", PixelSuccess)
                PerformanceRow("Conv2D", "~1.5-3x", PixelSuccess)
                PerformanceRow("功耗", "~30-50% ↓", PixelSecondary)
                PerformanceRow("小算子延迟", "一般 (RPC开销)", PixelWarning)
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "* 相对于 Vulkan GPU 的预期提升，实际因模型和设备而异",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
    }
}

@Composable
private fun PerformanceRow(op: String, gain: String, color: Color) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(op, style = MaterialTheme.typography.bodySmall)
        Text(gain, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold, color = color, fontFamily = FontFamily.Monospace)
    }
}

// ============================================================
// Empty State
// ============================================================

@Composable
private fun EmptyState(title: String, message: String, emoji: String = "📭") {
    PixelCard {
        Column(
            Modifier.fillMaxWidth().padding(vertical = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(emoji, fontSize = 48.sp)
            Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(
                message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

// ============================================================
// Advanced Parameters Dialog
// ============================================================

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AdvancedParametersDialog(
    architecture: ModelArchitecture,
    aspect: AspectRatio, custom: Boolean,
    width: String, height: String,
    steps: Int, cfg: Float, seed: Long?, scheduler: String,
    onAspect: (AspectRatio) -> Unit, onCustom: (Boolean) -> Unit,
    onWidth: (String) -> Unit, onHeight: (String) -> Unit,
    onSteps: (Int) -> Unit, onCfg: (Float) -> Unit,
    onSeed: (Long?) -> Unit, onScheduler: (String) -> Unit,
    onReset: () -> Unit, onDismiss: () -> Unit, onResolve: () -> Unit
) {
    val schedulers = when (architecture) {
        ModelArchitecture.SD3, ModelArchitecture.SD35, ModelArchitecture.FLUX, ModelArchitecture.ANIMA -> listOf("FlowMatch")
        else -> listOf("DDIM", "Euler")
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("⚙ 高级参数设置", fontWeight = FontWeight.Bold) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                Text("图像比例", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(AspectRatio.SQUARE, AspectRatio.PORTRAIT_34, AspectRatio.LANDSCAPE_43).forEach {
                        FilterChip(
                            selected = !custom && aspect == it,
                            onClick = { onAspect(it) },
                            label = { Text(it.label, fontSize = 12.sp) },
                            shape = MaterialTheme.shapes.small
                        )
                    }
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(AspectRatio.LANDSCAPE_169, AspectRatio.PORTRAIT_916).forEach {
                        FilterChip(
                            selected = !custom && aspect == it,
                            onClick = { onAspect(it) },
                            label = { Text(it.label, fontSize = 12.sp) },
                            shape = MaterialTheme.shapes.small
                        )
                    }
                    FilterChip(
                        selected = custom,
                        onClick = { onCustom(true) },
                        label = { Text("自定义", fontSize = 12.sp) },
                        shape = MaterialTheme.shapes.small
                    )
                }
                if (custom) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(width, onWidth, Modifier.weight(1f), label = { Text("宽") }, singleLine = true, shape = MaterialTheme.shapes.medium)
                        OutlinedTextField(height, onHeight, Modifier.weight(1f), label = { Text("高") }, singleLine = true, shape = MaterialTheme.shapes.medium)
                    }
                } else {
                    Text("推荐基准：${architecture.recommendedWidth} × ${architecture.recommendedHeight}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                Text("调度器", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    schedulers.forEach { value ->
                        FilterChip(
                            selected = scheduler == value,
                            onClick = { onScheduler(value) },
                            label = { Text(value, fontSize = 12.sp) },
                            shape = MaterialTheme.shapes.small
                        )
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("生成步数：$steps", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                    Slider(
                        value = steps.toFloat(),
                        onValueChange = { onSteps(it.toInt().coerceIn(1, 100)) },
                        valueRange = 1f..100f,
                        colors = SliderDefaults.colors(
                            thumbColor = MaterialTheme.colorScheme.primary,
                            activeTrackColor = MaterialTheme.colorScheme.primary,
                        )
                    )
                }

                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("CFG Scale：${"%.1f".format(Locale.US, cfg)}", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                    Slider(
                        value = cfg,
                        onValueChange = { onCfg(it) },
                        valueRange = 0f..20f,
                        colors = SliderDefaults.colors(
                            thumbColor = MaterialTheme.colorScheme.secondary,
                            activeTrackColor = MaterialTheme.colorScheme.secondary,
                        )
                    )
                }

                OutlinedTextField(
                    value = seed?.toString() ?: "",
                    onValueChange = { onSeed(it.toLongOrNull()) },
                    label = { Text("随机种子") },
                    placeholder = { Text("留空 = 随机") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.medium
                )
            }
        },
        confirmButton = { TextButton(onClick = onResolve) { Text("确定") } },
        dismissButton = { TextButton(onClick = onReset) { Text("恢复默认") } },
        shape = MaterialTheme.shapes.large,
    )
}

// ============================================================
// Model Page
// ============================================================

@Composable
private fun ModelPage(
    fileName: String?, size: Long, count: Long, status: String, tensorInfo: String,
    hash: String, cacheKey: String, architectureInfo: String, resolutionInfo: String,
    mapped: Boolean, onImport: () -> Unit, onResolveResolution: (Int, Int) -> Unit,
    onValidate: () -> Unit, onClose: () -> Unit
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Spacer(Modifier.height(4.dp))
        Text("模型管理", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        PixelButton(onClick = onImport, Modifier.fillMaxWidth().height(48.dp)) {
            Text("+ 导入模型", style = MaterialTheme.typography.titleMedium)
        }

        PixelCard(glow = mapped) {
            Text(fileName ?: "未加载模型", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(4.dp))
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                InfoRow("格式", "SafeTensors")
                InfoRow("大小", formatBytes(size))
                InfoRow("张量数", count.toString())
                InfoRow("存储", if (mapped) "mmap / zero-copy" else "—")
            }
            Spacer(Modifier.height(4.dp))
            StatusBadge(status, success = mapped)
        }
        if (architectureInfo.isNotBlank()) {
            val arch = ModelArchitecture.fromId(architectureInfo.substringBefore('|').toIntOrNull() ?: 7)
            PixelCard {
                Text("模型架构", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(arch.displayName, style = MaterialTheme.typography.titleSmall)
                Text("推荐分辨率：${arch.recommendedWidth} × ${arch.recommendedHeight}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (resolutionInfo.isNotBlank()) Text(resolutionInfo, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            ResolutionEditor(arch, onResolveResolution)
        }
        if (tensorInfo.isNotBlank()) PixelCard {
            Text("张量示例", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(tensorInfo, fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 18.sp)
        }
        if (hash.isNotBlank()) PixelCard {
            Text("模型指纹", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("SHA-256", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(hash, fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 16.sp)
                Spacer(Modifier.height(2.dp))
                Text("Runtime Cache Key", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(cacheKey, fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 16.sp)
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton(onClick = onValidate, Modifier.weight(1f), enabled = mapped, shape = MaterialTheme.shapes.medium) { Text("验证") }
            OutlinedButton(onClick = onClose, Modifier.weight(1f), enabled = mapped, shape = MaterialTheme.shapes.medium) { Text("关闭") }
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun ResolutionEditor(architecture: ModelArchitecture, onResolve: (Int, Int) -> Unit) {
    var width by remember(architecture) { mutableStateOf(architecture.recommendedWidth.toString()) }
    var height by remember(architecture) { mutableStateOf(architecture.recommendedHeight.toString()) }
    var ratio by remember(architecture) { mutableStateOf(AspectRatio.SQUARE) }
    PixelCard {
        Text("Resolution Planner", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(AspectRatio.SQUARE, AspectRatio.PORTRAIT_34, AspectRatio.LANDSCAPE_43, AspectRatio.LANDSCAPE_169).forEach {
                FilterChip(
                    selected = ratio == it,
                    onClick = {
                        ratio = it
                        val base = architecture.recommendedWidth
                        width = base.toString()
                        height = (base.toLong() * it.h / it.w).toInt().toString()
                    },
                    label = { Text(it.label, fontSize = 11.sp) },
                    shape = MaterialTheme.shapes.small
                )
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(width, { width = it.filter(Char::isDigit) }, Modifier.weight(1f), label = { Text("宽") }, singleLine = true, shape = MaterialTheme.shapes.medium)
            OutlinedTextField(height, { height = it.filter(Char::isDigit) }, Modifier.weight(1f), label = { Text("高") }, singleLine = true, shape = MaterialTheme.shapes.medium)
        }
        PixelButton(onClick = {
            val w = width.toIntOrNull()
            val h = height.toIntOrNull()
            if (w != null && h != null && w > 0 && h > 0) onResolve(w, h)
        }, Modifier.fillMaxWidth()) { Text("检查并解析分辨率") }
    }
}

// ============================================================
// Runtime Page
// ============================================================

@Composable
private fun RuntimePage(
    info: String, gpuTest: String, testing: Boolean,
    onRefresh: () -> Unit, onGpuTest: () -> Unit,
    tensorTest: String, onTensorTest: () -> Unit
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Spacer(Modifier.height(4.dp))
        Text("Runtime 调试", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        PixelCard {
            Text("GPU 设备信息", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(
                if (info.isBlank()) "未查询" else info,
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                color = if (info.isBlank()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.onSurface,
                lineHeight = 18.sp
            )
        }
        PixelButton(onClick = onRefresh, Modifier.fillMaxWidth()) { Text("刷新设备信息") }

        PixelCard {
            Text("Vulkan Compute", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text("真实 GPU dispatch 测试", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(6.dp))
            PixelButton(
                onClick = onGpuTest,
                enabled = !testing,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (testing) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onPrimary)
                    Spacer(Modifier.width(8.dp))
                    Text("运行中…")
                } else {
                    Text("运行 Vulkan 测试")
                }
            }
            if (gpuTest.isNotBlank()) {
                Spacer(Modifier.height(8.dp))
                Text(gpuTest, fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 18.sp)
            }
        }

        PixelCard {
            Text("Tensor Runtime", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            PixelButton(onClick = onTensorTest, Modifier.fillMaxWidth()) { Text("运行 Tensor 测试") }
            if (tensorTest.isNotBlank()) {
                Spacer(Modifier.height(8.dp))
                Text(tensorTest, fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 18.sp)
            }
        }
        Spacer(Modifier.height(16.dp))
    }
}

// ============================================================
// Helpers
// ============================================================

private fun resolveResolution(context: Context, architecture: ModelArchitecture, width: Int, height: Int, weightBytes: Long): String {
    val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val mem = ActivityManager.MemoryInfo()
    am.getMemoryInfo(mem)
    return try {
        val result = ResolutionRuntime.resolve(
            architecture = architecture,
            width = width,
            height = height,
            weightBytes = weightBytes.coerceAtLeast(0),
            availableCpuBytes = mem.availMem,
            availableGpuBytes = try { NativeRuntime.nativeGetVulkanMemoryBytes() } catch (_: Throwable) { 0L },
            availableNpuBytes = 0
        )
        buildString {
            append("请求：${result.requestedWidth} × ${result.requestedHeight}")
            append("\n执行：${result.width} × ${result.height}  | latent ${result.latentWidth} × ${result.latentHeight}")
            append("\n最终输出：${result.outputWidth} × ${result.outputHeight}")
            append("\n估算峰值：${formatBytes(result.estimatedPeakBytes)}")
            if (result.memoryBudgetBytes > 0) append("\n可用预算：${formatBytes(result.memoryBudgetBytes)}")
            if (result.warning != null) append("\n⚠ ${result.warning}")
        }
    } catch (t: Throwable) {
        "✕ 分辨率规划失败：${t.message ?: "unknown error"}"
    }
}

private fun queryDisplayName(context: Context, uri: Uri): String? =
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { c: Cursor -> if (c.moveToFirst()) c.getString(c.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME)) else null }

private fun formatBytes(value: Long): String {
    if (value < 0) return "Unknown"
    if (value == 0L) return "0 B"
    val units = arrayOf("B", "KB", "MB", "GB", "TB")
    var n = value.toDouble()
    var i = 0
    while (n >= 1024 && i < units.lastIndex) { n /= 1024; i++ }
    return String.format(Locale.US, "%.2f %s", n, units[i])
}

object NativeRuntime {
    init { System.loadLibrary("localimage_runtime") }
    external fun initializeVulkan(): String
    external fun nativeOpenSafeTensor(fd: Int): Long
    external fun nativeGetFileSize(handle: Long): Long
    external fun nativeGetTensorCount(handle: Long): Long
    external fun nativeGetTensorInfo(handle: Long, index: Int): String
    external fun nativeGetFirstSupportedTensorBytes(handle: Long): String
    external fun nativeValidateModel(handle: Long): Boolean
    external fun nativeGetModelHash(handle: Long): String
    external fun nativeGetModelArchitecture(handle: Long): String
    external fun nativeGetDeviceInfo(): String
    external fun nativeGetVulkanMemoryBytes(): Long
    external fun nativeGetCacheKey(handle: Long): String
    external fun nativeCloseSafeTensor(handle: Long)
    external fun nativeRunTensorTest(tempDir: String): String
    external fun nativeRunVulkanComputeTest(): String
    external fun nativeGetNpuInfo(): String
    external fun nativeResolveResolution(
        architecture: Int, width: Int, height: Int, dtype: Int, backend: Int,
        weightBytes: Long, availableCpuBytes: Long, availableGpuBytes: Long, availableNpuBytes: Long
    ): String
}
