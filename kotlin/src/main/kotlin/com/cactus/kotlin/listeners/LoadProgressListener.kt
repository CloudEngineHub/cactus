package com.cactus.kotlin.listeners

/**
 * Listener for model loading progress
 */
interface LoadProgressListener {
    /**
     * Called when model loading progress updates
     * @param progress The loading progress as a percentage (0-100)
     */
    fun onProgress(progress: Int)
    
    /**
     * Called when the model loading is complete
     */
    fun onComplete()
    
    /**
     * Called when an error occurs during model loading
     * @param error The error message
     */
    fun onError(error: String)
} 