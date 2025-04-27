package com.cactus.kotlin.models

/**
 * Information about a loaded model
 */
data class ModelInfo(
    /** Model name */
    val name: String,
    
    /** Model architecture */
    val architecture: String,
    
    /** Model parameters */
    val params: Long,
    
    /** Context size */
    val contextSize: Int,
    
    /** Vocabulary size */
    val vocabSize: Int,
    
    /** Embedding dimensions */
    val embeddingSize: Int,
    
    /** Whether the model supports embeddings */
    val supportsEmbeddings: Boolean,
    
    /** Metadata from the model */
    val metadata: Map<String, String>
) 