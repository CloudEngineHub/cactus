package com.cactus.example

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import com.cactus.Cactus
import com.cactus.SpeechRecognitionParams

@Composable
fun SpeechRecordButton(
    cactus: Cactus?,
    modifier: Modifier = Modifier,
    onTextRecognized: (String) -> Unit = {},
    onError: (String) -> Unit = {}
) {
    var isRecording by remember { mutableStateOf(false) }
    var isInitialized by remember { mutableStateOf(false) }
    var hasPermission by remember { mutableStateOf(false) }
    var lastRecognizedText by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()

    // Auto-initialize speech when cactus is available
    LaunchedEffect(cactus) {
        if (cactus != null) {
            try {
                // Initialize speech
                val speechInit = cactus.initializeSpeechRecognition()
                if (speechInit) {
                    isInitialized = true
                    
                    // Check/request permissions
                    val permission = cactus.requestSpeechPermissions()
                    hasPermission = permission
                } else {
                    onError("Failed to initialize speech recognition")
                }
            } catch (e: Exception) {
                onError("Failed to initialize: ${e.message}")
            }
        }
    }

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        SpeechPermissionHandler(
            onPermissionResult = { granted ->
                hasPermission = granted
                if (!granted) {
                    onError("Microphone permission required")
                }
            }
        ) { requestPermission ->
            
            Button(
                onClick = {
                    if (!hasPermission) {
                        requestPermission()
                        return@Button
                    }
                    
                    if (!isRecording) {
                        isRecording = true
                        scope.launch {
                            try {
                                val result = cactus?.speechToText(
                                    SpeechRecognitionParams(
                                        language = "en-US",
                                        enablePartialResults = false,
                                        maxDuration = 10
                                    )
                                )
                                
                                isRecording = false
                                
                                if (result != null && result.text.isNotBlank()) {
                                    lastRecognizedText = result.text
                                    onTextRecognized(result.text)
                                } else {
                                    onError("No speech detected")
                                }
                            } catch (e: Exception) {
                                isRecording = false
                                onError("Recording failed: ${e.message}")
                            }
                        }
                    } else {
                        cactus?.stopSpeechRecognition()
                        isRecording = false
                    }
                },
                enabled = cactus != null && isInitialized,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isRecording) MaterialTheme.colorScheme.error 
                                   else MaterialTheme.colorScheme.primary
                )
            ) {
                Text(
                    when {
                        cactus == null -> "🤖 Initialize Cactus First"
                        !isInitialized -> "Initializing..."
                        !hasPermission -> "Grant Permission"
                        isRecording -> "🎤 Recording... (Tap to Stop)"
                        else -> "🎤 Press to Record"
                    }
                )
            }
        }
        
        if (lastRecognizedText.isNotBlank()) {
            Card(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)
            ) {
                Text(
                    text = "\"$lastRecognizedText\"",
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        }
        
        if (isRecording) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text("Listening...", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}