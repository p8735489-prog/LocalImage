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
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.haobai.localimage.runtime.AspectRatio
import com.haobai.localimage.runtime.ModelArchitecture
import com.haobai.localimage.runtime.ResolutionRuntime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

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
    val colors = if (Build.VERSION.SDK_INT >= 31) {
        if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
    } else {
        if (dark) darkColorScheme() else lightColorScheme()
    }
    MaterialTheme(colorScheme = colors, content = content)
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

        Scaffold(
            contentWindowInsets = WindowInsets.safeDrawing,
            topBar = {
                CenterAlignedTopAppBar(
                    title = { Text("LocalImage", fontWeight = FontWeight.SemiBold) },
                    actions = {
                        IconButton(onClick = { page = 2 }) {
                            Icon(Icons.Default.Settings, contentDescription = "Runtime 设置")
                        }
                    }
                )
            },
            bottomBar = {
                NavigationBar(windowInsets = WindowInsets.navigationBars) {
                    NavigationBarItem(page == 0, { page = 0 }, icon = { Icon(Icons.Default.Image, null) }, label = { Text("生成") })
                    NavigationBarItem(page == 1, { page = 1 }, icon = { Icon(Icons.Default.Memory, null) }, label = { Text("模型") })
                    NavigationBarItem(page == 2, { page = 2 }, icon = { Icon(Icons.Default.Settings, null) }, label = { Text("Runtime") })
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
                        else generateMessage = "当前工程的生成图链路尚未接通；Runtime 不会伪造图片输出。模型 Inspector、ResolutionPolicy 和 CPU/Vulkan Runtime 已可执行。"
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
                text = { Text(message) }
            )
        }
    }
}

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
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        TabRow(selectedTabIndex = tab) {
            listOf("提示词", "生成结果", "历史").forEachIndexed { index, title ->
                Tab(selected = tab == index, onClick = { onTab(index) }, text = { Text(title) })
            }
        }
        when (tab) {
            0 -> {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("当前模型", style = MaterialTheme.typography.titleMedium)
                        Text(fileName ?: "未加载模型", style = MaterialTheme.typography.titleLarge)
                        if (fileName != null) {
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                AssistChip(onClick = {}, label = { Text(architecture.displayName) })
                                AssistChip(onClick = {}, label = { Text("${architecture.recommendedWidth} × ${architecture.recommendedHeight}") })
                            }
                        } else {
                            Button(onClick = onImport, enabled = !loading, Modifier.fillMaxWidth()) { Text("导入模型") }
                        }
                    }
                }

                OutlinedTextField(
                    value = prompt, onValueChange = onPrompt,
                    modifier = Modifier.fillMaxWidth().semantics { contentDescription = "正面提示词" },
                    minLines = 5, label = { Text("图像生成提示词") },
                    placeholder = { Text("例如：1girl, blue hair, detailed illustration") },
                    supportingText = { Text("字符 ${prompt.length}") }
                )
                OutlinedTextField(
                    value = negativePrompt, onValueChange = onNegative,
                    modifier = Modifier.fillMaxWidth().semantics { contentDescription = "负面提示词" },
                    minLines = 3, label = { Text("负面提示词") },
                    placeholder = { Text("例如：blurry, low quality, watermark") }
                )

                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(onClick = {}, Modifier.weight(1f)) {
                        Icon(Icons.Default.AddPhotoAlternate, null)
                        Spacer(Modifier.padding(2.dp))
                        Text("img2img")
                    }
                    OutlinedButton(onClick = onAdvanced, Modifier.weight(1f)) {
                        Icon(Icons.Default.Tune, null)
                        Spacer(Modifier.padding(2.dp))
                        Text("高级设置")
                    }
                }

                if (resolutionInfo.isNotBlank()) {
                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                            Text("分辨率", style = MaterialTheme.typography.titleMedium)
                            Text(resolutionInfo)
                        }
                    }
                }

                Button(
                    onClick = onGenerate,
                    enabled = fileName != null && prompt.isNotBlank() && !loading,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("生成图像")
                }
                Text(status, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            1 -> EmptyState("生成结果", "真实 Runtime 产生 PNG 后会显示在这里。")
            2 -> EmptyState("历史", "生成记录会在真实生成链路接通后保存。")
        }
    }
}

@Composable
private fun EmptyState(title: String, message: String) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(28.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(Icons.Default.History, contentDescription = null)
            Text(title, style = MaterialTheme.typography.titleLarge)
            Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

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
        title = { Text("高级参数设置") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("图像宽高比", style = MaterialTheme.typography.titleSmall)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(AspectRatio.SQUARE, AspectRatio.PORTRAIT_34, AspectRatio.LANDSCAPE_43).forEach {
                        FilterChip(selected = !custom && aspect == it, onClick = { onAspect(it) }, label = { Text(it.label) })
                    }
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(AspectRatio.LANDSCAPE_169, AspectRatio.PORTRAIT_916).forEach {
                        FilterChip(selected = !custom && aspect == it, onClick = { onAspect(it) }, label = { Text(it.label) })
                    }
                    FilterChip(selected = custom, onClick = { onCustom(true) }, label = { Text("自定义") })
                }
                if (custom) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(width, onWidth, Modifier.weight(1f), label = { Text("宽") }, singleLine = true)
                        OutlinedTextField(height, onHeight, Modifier.weight(1f), label = { Text("高") }, singleLine = true)
                    }
                } else {
                    Text("推荐基准：${architecture.recommendedWidth} × ${architecture.recommendedHeight}")
                }

                Text("调度器", style = MaterialTheme.typography.titleSmall)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    schedulers.forEach { value ->
                        FilterChip(selected = scheduler == value, onClick = { onScheduler(value) }, label = { Text(value) })
                    }
                }

                Text("生成步数：$steps")
                Slider(value = steps.toFloat(), onValueChange = { onSteps(it.toInt().coerceIn(1, 100)) }, valueRange = 1f..100f)

                Text("CFG Scale：${"%.1f".format(Locale.US, cfg)}")
                Slider(value = cfg, onValueChange = { onCfg(it) }, valueRange = 0f..20f)

                OutlinedTextField(
                    value = seed?.toString() ?: "",
                    onValueChange = { onSeed(it.toLongOrNull()) },
                    label = { Text("随机种子（可选）") },
                    placeholder = { Text("留空 = 随机") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = { TextButton(onClick = onResolve) { Text("确定") } },
        dismissButton = { TextButton(onClick = onReset) { Text("恢复默认") } }
    )
}

@Composable
private fun ModelPage(
    fileName: String?, size: Long, count: Long, status: String, tensorInfo: String,
    hash: String, cacheKey: String, architectureInfo: String, resolutionInfo: String,
    mapped: Boolean, onImport: () -> Unit, onResolveResolution: (Int, Int) -> Unit,
    onValidate: () -> Unit, onClose: () -> Unit
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("模型", style = MaterialTheme.typography.headlineSmall)
        Button(onClick = onImport, Modifier.fillMaxWidth()) { Text("+ 导入模型") }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text(fileName ?: "未加载模型", style = MaterialTheme.typography.titleLarge)
                Text("格式：SafeTensors")
                Text("大小：${formatBytes(size)}")
                Text("张量：$count")
                Text("存储：${if (mapped) "mmap / zero-copy view" else "—"}")
                AssistChip(onClick = {}, label = { Text(status) })
            }
        }
        if (architectureInfo.isNotBlank()) {
            val arch = ModelArchitecture.fromId(architectureInfo.substringBefore('|').toIntOrNull() ?: 7)
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text("模型架构", style = MaterialTheme.typography.titleMedium)
                    Text(arch.displayName)
                    Text("推荐分辨率：${arch.recommendedWidth} × ${arch.recommendedHeight}")
                    if (resolutionInfo.isNotBlank()) Text(resolutionInfo, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            ResolutionEditor(arch, onResolveResolution)
        }
        if (tensorInfo.isNotBlank()) Card(Modifier.fillMaxWidth()) { Column(Modifier.padding(18.dp)) { Text("张量示例", style = MaterialTheme.typography.titleMedium); Text(tensorInfo) } }
        if (hash.isNotBlank()) Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text("模型 SHA-256", style = MaterialTheme.typography.titleMedium)
                Text(hash)
                Text("Runtime Cache Key", style = MaterialTheme.typography.titleMedium)
                Text(cacheKey)
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton(onClick = onValidate, Modifier.weight(1f), enabled = mapped) { Text("验证") }
            OutlinedButton(onClick = onClose, Modifier.weight(1f), enabled = mapped) { Text("关闭") }
        }
    }
}

@Composable
private fun ResolutionEditor(architecture: ModelArchitecture, onResolve: (Int, Int) -> Unit) {
    var width by remember(architecture) { mutableStateOf(architecture.recommendedWidth.toString()) }
    var height by remember(architecture) { mutableStateOf(architecture.recommendedHeight.toString()) }
    var ratio by remember(architecture) { mutableStateOf(AspectRatio.SQUARE) }
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Resolution Planner", style = MaterialTheme.typography.titleMedium)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(AspectRatio.SQUARE, AspectRatio.PORTRAIT_34, AspectRatio.LANDSCAPE_43, AspectRatio.LANDSCAPE_169).forEach {
                FilterChip(selected = ratio == it, onClick = {
                    ratio = it
                    val base = architecture.recommendedWidth
                    width = base.toString()
                    height = (base.toLong() * it.h / it.w).toInt().toString()
                }, label = { Text(it.label) })
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(width, { width = it.filter(Char::isDigit) }, Modifier.weight(1f), label = { Text("宽") }, singleLine = true)
            OutlinedTextField(height, { height = it.filter(Char::isDigit) }, Modifier.weight(1f), label = { Text("高") }, singleLine = true)
        }
        Button(onClick = {
            val w = width.toIntOrNull()
            val h = height.toIntOrNull()
            if (w != null && h != null && w > 0 && h > 0) onResolve(w, h)
        }, Modifier.fillMaxWidth()) { Text("检查并解析分辨率") }
    }
}

@Composable
private fun RuntimePage(
    info: String, gpuTest: String, testing: Boolean,
    onRefresh: () -> Unit, onGpuTest: () -> Unit,
    tensorTest: String, onTensorTest: () -> Unit
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("Runtime", style = MaterialTheme.typography.headlineSmall)
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Backend / Device", style = MaterialTheme.typography.titleLarge)
                Text(if (info.isBlank()) "未查询" else info, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Button(onClick = onRefresh, Modifier.fillMaxWidth()) { Text("刷新设备信息") }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Vulkan Compute", style = MaterialTheme.typography.titleMedium)
                AssistChip(onClick = {}, label = { Text("真实 GPU dispatch test") })
                Button(onClick = onGpuTest, enabled = !testing, Modifier.fillMaxWidth()) {
                    if (testing) CircularProgressIndicator() else Text("运行 Vulkan 测试")
                }
                if (gpuTest.isNotBlank()) Text(gpuTest)
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Tensor / SafeTensors", style = MaterialTheme.typography.titleMedium)
                Button(onClick = onTensorTest, Modifier.fillMaxWidth()) { Text("运行 Tensor Runtime 测试") }
                if (tensorTest.isNotBlank()) Text(tensorTest)
            }
        }
    }
}

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
    external fun nativeResolveResolution(
        architecture: Int, width: Int, height: Int, dtype: Int, backend: Int,
        weightBytes: Long, availableCpuBytes: Long, availableGpuBytes: Long, availableNpuBytes: Long
    ): String
}
