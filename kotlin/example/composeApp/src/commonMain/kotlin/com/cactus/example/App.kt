package com.cactus.example

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cactus.Cactus
import com.cactus.CactusInitParams
import com.cactus.CactusCompletionParams

@Composable
fun App() {
    var logs by remember { mutableStateOf(listOf<String>()) }
    var cactus by remember { mutableStateOf<Cactus?>(null) }
    var isInitialized by remember { mutableStateOf(false) }
    
    fun addLog(message: String) {
        logs = logs + message
    }
    
    MaterialTheme {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Cactus Demo",
                style = MaterialTheme.typography.headlineMedium,
                modifier = Modifier.padding(bottom = 16.dp)
            )
            
            Button(
                onClick = {
                    addLog("Initializing Cactus...")
                    try {
                        val instance = Cactus()
                        val params = CactusInitParams(
                            modelPath = "/path/to/model.gguf",
                            nCtx = 2048,
                            nThreads = 4
                        )
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
                    }
                },
                enabled = !isInitialized
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
                    } ?: addLog("Please initialize Cactus first")
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
                        try {
                            val result = c.tokenize("Hello, world!")
                            addLog("Tokens: ${result?.tokens?.contentToString()}")
                            addLog("Token count: ${result?.count}")
                        } catch (e: Exception) {
                            addLog("✗ Error tokenizing: ${e.message}")
                        }
                    } ?: addLog("Please initialize Cactus first")
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
                    } ?: addLog("Please initialize Cactus first")
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
                    } ?: addLog("Please initialize Cactus first")
                },
                enabled = isInitialized
            ) {
                Text("Generate Text")
            }
            
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
    
    DisposableEffect(Unit) {
        onDispose {
            cactus?.close()
        }
    }
} 