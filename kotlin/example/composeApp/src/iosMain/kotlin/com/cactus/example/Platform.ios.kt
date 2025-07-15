@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
package com.cactus.example

import kotlinx.cinterop.*
import kotlinx.coroutines.suspendCancellableCoroutine
import platform.Foundation.*
import platform.darwin.NSObject
import kotlin.coroutines.resume

actual fun getModelCacheDir(): String {
    val cacheDirs = NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory,
        NSUserDomainMask,
        true
    )
    val cacheDir = (cacheDirs[0] as NSString)
    return cacheDir.toString()
}

actual suspend fun downloadModelStreaming(url: String, fileName: String, onProgress: (Float) -> Unit): String? {
    return suspendCancellableCoroutine { continuation ->
        try {
            println("=== Starting iOS model download ===")
            println("URL: $url")
            println("File: $fileName")
            
            val cacheDir = getModelCacheDir()
            val modelsDir = "$cacheDir/models"
            
            // Create models directory if it doesn't exist
            val fileManager = NSFileManager.defaultManager
            fileManager.createDirectoryAtPath(
                modelsDir,
                withIntermediateDirectories = true,
                attributes = null,
                error = null
            )
            
            val filePath = "$modelsDir/$fileName"
            println("Target file path: $filePath")
            
            // If file already exists, return it
            if (fileManager.fileExistsAtPath(filePath)) {
                println("File already exists")
                continuation.resume(filePath)
                return@suspendCancellableCoroutine
            }
            
            // Create URL and request
            val nsUrl = NSURL.URLWithString(url)
            if (nsUrl == null) {
                println("Invalid URL: $url")
                continuation.resume(null)
                return@suspendCancellableCoroutine
            }
            
            val request = NSMutableURLRequest.requestWithURL(nsUrl)
            request.setTimeoutInterval(300.0) // 5 minutes timeout
            
            // Create download task
            val session = NSURLSession.sharedSession
            val downloadTask = session.downloadTaskWithRequest(request) { tempUrl, response, error ->
                if (error != null) {
                    println("Download error: ${error.localizedDescription}")
                    continuation.resume(null)
                    return@downloadTaskWithRequest
                }
                
                val httpResponse = response as? NSHTTPURLResponse
                if (httpResponse?.statusCode != 200L) {
                    println("HTTP error: ${httpResponse?.statusCode}")
                    continuation.resume(null)
                    return@downloadTaskWithRequest
                }
                
                if (tempUrl == null) {
                    println("No temporary URL received")
                    continuation.resume(null)
                    return@downloadTaskWithRequest
                }
                
                // Move downloaded file to final location
                val finalUrl = NSURL.fileURLWithPath(filePath)
                val moveError = memScoped {
                    val errorPtr = alloc<ObjCObjectVar<NSError?>>()
                    val success = fileManager.moveItemAtURL(tempUrl, finalUrl, errorPtr.ptr)
                    if (!success) {
                        errorPtr.value
                    } else {
                        null
                    }
                }
                
                if (moveError != null) {
                    println("Error moving file: ${moveError.localizedDescription}")
                    continuation.resume(null)
                } else {
                    println("✓ Download completed successfully")
                    continuation.resume(filePath)
                }
            }
            
            // Start download
            println("Starting download...")
            downloadTask.resume()
            
            // Handle cancellation
            continuation.invokeOnCancellation {
                downloadTask.cancel()
            }
            
        } catch (e: Exception) {
            println("Exception during download: ${e.message}")
            e.printStackTrace()
            continuation.resume(null)
        }
    }
} 