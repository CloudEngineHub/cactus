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
import com.cactus.CactusLM
import com.cactus.CactusVLM
import com.cactus.CactusSTT
import com.cactus.CactusTTS
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding

@Composable
fun App() {
    val scope = rememberCoroutineScope()
    val lm = remember { CactusLM() }
    val vlm = remember { CactusVLM() }
    val stt = remember { CactusSTT() }
    val tts = remember { CactusTTS() }
    
    var logs by remember { mutableStateOf(listOf<String>()) }
    
    fun addLog(message: String) {
        logs = logs + "${logs.size + 1}. $message"
    }
    
    LaunchedEffect(Unit) {
        addLog("App started - Cactus Modular Demo")
        addLog("Available: Language Model, Vision, Speech-to-Text, Text-to-Speech")
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
                    text = "Cactus Modular Demo",
                    style = MaterialTheme.typography.headlineMedium,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                
                // Language Model Section
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Language Model", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Downloading LM model...")
                                        val success = lm.download()
                                        addLog(if (success) "LM model downloaded" else "LM download failed")
                                    }
                                }
                            ) { Text("Download") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Loading LM model...")
                                        val success = lm.load("Qwen3-0.6B-Q8_0.gguf")
                                        addLog(if (success) "LM model loaded" else "LM load failed")
                                    }
                                }
                            ) { Text("Load") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Generating text...")
                                        val result = lm.completion("What is AI?", maxTokens = 50)
                                        addLog("LM: ${result ?: "No response"}")
                                    }
                                }
                            ) { Text("Generate") }
                        }
                    }
                }
                
                // Vision Language Model Section
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Vision Language Model", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Downloading VLM models...")
                                        val success = vlm.download()
                                        addLog(if (success) "VLM models downloaded" else "VLM download failed")
                                    }
                                }
                            ) { Text("Download") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Loading VLM model...")
                                        val success = vlm.load("SmolVLM2-500M-Video-Instruct-Q8_0.gguf")
                                        addLog(if (success) "VLM model loaded" else "VLM load failed")
                                    }
                                }
                            ) { Text("Load") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Analyzing image...")
                                        val result = vlm.completion("Describe this", "image.jpg", maxTokens = 50)
                                        addLog("VLM: ${result ?: "No response"}")
                                    }
                                }
                            ) { Text("Analyze") }
                        }
                    }
                }
                
                // Speech to Text Section
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Speech to Text", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Downloading STT model...")
                                        val downloadSuccess = stt.download()
                                        if (downloadSuccess) {
                                            addLog("Initializing STT...")
                                            val initSuccess = stt.initialize()
                                            addLog(if (initSuccess) "STT ready" else "STT init failed")
                                        } else {
                                            addLog("STT download failed")
                                        }
                                    }
                                }
                            ) { Text("Setup") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        if (!stt.isReady()) {
                                            addLog("Setting up STT first...")
                                            val downloadSuccess = stt.download()
                                            if (downloadSuccess) {
                                                val initSuccess = stt.initialize()
                                                if (!initSuccess) {
                                                    addLog("STT initialization failed")
                                                    return@launch
                                                }
                                            } else {
                                                addLog("STT download failed")
                                                return@launch
                                            }
                                        }
                                        
                                        addLog("Listening...")
                                        val result = stt.transcribe()
                                        addLog("STT: ${result?.text ?: "No speech detected"}")
                                    }
                                }
                            ) { Text("Listen") }
                        }
                    }
                }
                
                // Text to Speech Section
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Text to Speech", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Setting up TTS...")
                                        val downloadSuccess = tts.download("https://example.com/tts.gguf")
                                        if (downloadSuccess) {
                                            val initSuccess = tts.initialize()
                                            addLog(if (initSuccess) "TTS ready" else "TTS init failed")
                                        } else {
                                            addLog("TTS download failed")
                                        }
                                    }
                                }
                            ) { Text("Setup") }
                            
                            Button(
                                onClick = {
                                    scope.launch {
                                        addLog("Speaking...")
                                        val success = tts.speak("Hello from Cactus TTS")
                                        addLog(if (success) "TTS completed" else "TTS failed")
                                    }
                                }
                            ) { Text("Speak") }
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                HorizontalDivider()
                Spacer(modifier = Modifier.height(16.dp))
                
                Text("Logs:", style = MaterialTheme.typography.titleMedium)
                
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
} 