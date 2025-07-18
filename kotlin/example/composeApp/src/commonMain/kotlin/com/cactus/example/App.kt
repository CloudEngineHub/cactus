package com.cactus.example

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import com.cactus.Cactus
import com.cactus.CactusInitParams
import com.cactus.CactusCompletionParams
import com.cactus.SpeechRecognitionParams
import io.ktor.client.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.utils.io.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding

expect fun getModelCacheDir(): String
expect suspend fun downloadModelStreaming(url: String, fileName: String, onProgress: (Float) -> Unit): String?

@Composable
fun App() {
    val scope = rememberCoroutineScope()
    var cactus by remember { mutableStateOf<Cactus?>(null) }
    var isInitialized by remember { mutableStateOf(false) }
    var modelPath by remember { mutableStateOf<String?>(null) }
    var logs by remember { mutableStateOf(listOf<String>()) }
    var isDownloading by remember { mutableStateOf(false) }
    var downloadProgress by remember { mutableStateOf(0f) }
    
    // Initialize logging
    LaunchedEffect(Unit) {
        logs = listOf("📱 App started successfully", "👋 Welcome to Cactus Demo!")
    }

    fun addLog(message: String) {
        logs = logs + "${logs.size + 1}. $message"
    }
    
    suspend fun downloadModel(): String? {
        return withContext(Dispatchers.Default) {
            try {
                val modelUrl = "https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
                val fileName = "Qwen3-0.6B-Q8_0.gguf"
                
                addLog("Starting model download...")
                
                downloadModelStreaming(modelUrl, fileName) { progress ->
                    downloadProgress = progress
                }
            } catch (e: Exception) {
                addLog("Download failed: ${e.message}")
                null
            }
        }
    }
    
    MaterialTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
                    .windowInsetsPadding(WindowInsets.safeDrawing),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Cactus Demo",
                    style = MaterialTheme.typography.headlineMedium,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                
                if (isDownloading) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(progress = { downloadProgress })
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("Downloading model: ${(downloadProgress * 100).toInt()}%")
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                }
                
                Button(
                    onClick = {
                        addLog("Download button clicked")
                        scope.launch {
                            isDownloading = true
                            downloadProgress = 0f
                            val downloadedModelPath = downloadModel()
                            isDownloading = false
                            
                            if (downloadedModelPath != null) {
                                modelPath = downloadedModelPath
                                addLog("✓ Model downloaded successfully!")
                                addLog("Model saved to: $downloadedModelPath")
                            } else {
                                addLog("✗ Failed to download model")
                            }
                        }
                    },
                    enabled = !isDownloading && modelPath == null
                ) {
                    Text("Download Model")
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = {
                        addLog("Initialize button clicked")
                        val currentModelPath = modelPath
                        if (currentModelPath != null) {
                            addLog("Initializing Cactus...")
                            scope.launch {
                            try {
                                addLog("Creating Cactus instance...")
                                val instance = Cactus()
                                addLog("Setting up initialization parameters...")
                                val params = CactusInitParams(
                                    modelPath = currentModelPath,
                                    nCtx = 2048,
                                    nThreads = 4
                                )
                                addLog("Calling initialize...")
                                val success = instance.initialize(params)
                                if (success) {
                                    cactus = instance
                                    isInitialized = true
                                    addLog("✓ Cactus initialized successfully!")
                                } else {
                                    addLog("✗ Failed to initialize Cactus")
                                }
                            } catch (e: Exception) {
                                addLog("✗ Error: ${e.message}")
                                addLog("Exception type: ${e::class.simpleName}")
                                }
                            }
                        } else {
                            addLog("Please download the model first")
                        }
                    },
                    enabled = !isInitialized && modelPath != null && !isDownloading
                ) {
                    Text("Initialize Cactus")
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = {
                        cactus?.let { c ->
                            addLog("Getting model info...")
                            try {
                                val desc = c.getModelDesc()
                                val size = c.getModelSize()
                                val params = c.getModelParams()
                                addLog("Model: $desc")
                                addLog("Size: ${size / 1024 / 1024} MB")
                                addLog("Parameters: ${params / 1_000_000}M")
                            } catch (e: Exception) {
                                addLog("✗ Error getting model info: ${e.message}")
                            }
                        } ?: run { addLog("Please initialize Cactus first") }
                    },
                    enabled = isInitialized
                ) {
                    Text("Get Model Info")
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = {
                        cactus?.let { c ->
                            addLog("Tokenizing text...")
                            scope.launch {
                            try {
                                val result = c.tokenize("Hello, world!")
                                result?.let { r ->
                                    addLog("Tokens: ${r.tokens.contentToString()}")
                                    addLog("Token count: ${r.count}")
                                }
                            } catch (e: Exception) {
                                addLog("✗ Error tokenizing: ${e.message}")
                                }
                            }
                        } ?: run { addLog("Please initialize Cactus first") }
                    },
                    enabled = isInitialized
                ) {
                    Text("Tokenize Text")
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = {
                        cactus?.let { c ->
                            addLog("Running benchmark...")
                            scope.launch {
                            try {
                                val result = c.bench(pp = 512, tg = 128, pl = 1, nr = 1)
                                result?.let { r ->
                                    addLog("Benchmark Results:")
                                    addLog("PP Avg: ${r.ppAvg} tokens/s")
                                    addLog("TG Avg: ${r.tgAvg} tokens/s")
                                }
                            } catch (e: Exception) {
                                addLog("✗ Error running benchmark: ${e.message}")
                                }
                            }
                        } ?: run { addLog("Please initialize Cactus first") }
                    },
                    enabled = isInitialized
                ) {
                    Text("Run Benchmark")
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = {
                        cactus?.let { c ->
                            addLog("Generating completion...")
                            scope.launch {
                            try {
                                val params = CactusCompletionParams(
                                    prompt = "What is the capital of France?",
                                    nPredict = 50,
                                    temperature = 0.7
                                )
                                val result = c.completion(params)
                                result?.let { r ->
                                    addLog("Completion: ${r.text}")
                                    addLog("Tokens: ${r.tokensPredicted}")
                                }
                            } catch (e: Exception) {
                                addLog("✗ Error generating completion: ${e.message}")
                                }
                            }
                        } ?: run { addLog("Please initialize Cactus first") }
                    },
                    enabled = isInitialized
                ) {
                    Text("Generate Text")
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                HorizontalDivider()
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    text = "Speech Recognition:",
                    style = MaterialTheme.typography.titleMedium
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                SpeechRecordButton(
                    cactus = cactus,
                    modifier = Modifier.fillMaxWidth(),
                    onTextRecognized = { text ->
                        addLog("🎤 Speech: \"$text\"")
                    },
                    onError = { error ->
                        addLog("✗ Speech error: $error")
                    }
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                HorizontalDivider()
                
                Spacer(modifier = Modifier.height(16.dp))
                
                Text(
                    text = "Logs:",
                    style = MaterialTheme.typography.titleMedium
                )
                
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(logs) { log ->
                        Text(
                            text = log,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(4.dp)
                        )
                    }
                }
            }
        }
    }
    
    DisposableEffect(Unit) {
        onDispose {
            cactus?.close()
        }
    }
} 