package com.cactus.kotlin.listeners

/**
 * Listener for streaming completion tokens
 */
interface CompletionListener {
    /**
     * Called when a new token is generated
     * @param tokenText The text of the new token
     * @param isPartial Whether this is a partial token (when using byte-pair encoding)
     * @param tokenId The ID of the token in the vocabulary
     */
    fun onToken(tokenText: String, isPartial: Boolean, tokenId: Int)
    
    /**
     * Called when completion is finished
     */
    fun onComplete()
    
    /**
     * Called when an error occurs during completion
     * @param error The error message
     */
    fun onError(error: String)
} 