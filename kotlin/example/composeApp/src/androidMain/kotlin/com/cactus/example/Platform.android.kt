package com.cactus.example

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import com.cactus.CactusContextInitializer

actual suspend fun saveImageToTempFile(imageBytes: ByteArray): String? {
    return withContext(Dispatchers.IO) {
        try {
            val context = CactusContextInitializer.getApplicationContext()
            val tempFile = File(context.cacheDir, "temp_image.jpg")
            
            tempFile.writeBytes(imageBytes)
            
            tempFile.absolutePath
        } catch (e: Exception) {
            null
        }
    }
} 