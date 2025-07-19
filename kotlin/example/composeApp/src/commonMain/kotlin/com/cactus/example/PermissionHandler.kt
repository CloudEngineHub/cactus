package com.cactus.example

import androidx.compose.runtime.Composable
 
@Composable
expect fun SpeechPermissionHandler(
    onPermissionResult: (Boolean) -> Unit,
    content: @Composable (requestPermission: () -> Unit) -> Unit
) 