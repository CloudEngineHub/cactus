package com.cactus.example

import androidx.compose.runtime.*

@Composable
actual fun SpeechPermissionHandler(
    onPermissionResult: (Boolean) -> Unit,
    content: @Composable (requestPermission: () -> Unit) -> Unit
) {
    // iOS speech recognition permissions are handled differently
    // The speech recognition API will request permissions automatically
    val requestPermission = {
        // On iOS, permission is requested through the speech recognition API itself
        onPermissionResult(true) // Let the speech API handle it
    }

    content(requestPermission)
} 