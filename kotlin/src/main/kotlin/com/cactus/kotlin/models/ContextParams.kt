package com.cactus.kotlin.models

/**
 * Parameters for initializing a Llama context
 */
data class ContextParams(
    /** Path to the model file */
    val model: String,
    
    /** Custom chat template */
    val chatTemplate: String? = null,
    
    /** Reasoning format (default: "none") */
    val reasoningFormat: String = "none",
    
    /** Enable embedding */
    val embedding: Boolean = false,
    
    /** Embedding normalization (-1 for default) */
    val embdNormalize: Int = -1,
    
    /** Context size */
    val nCtx: Int = 512,
    
    /** Batch size */
    val nBatch: Int = 512,
    
    /** Micro batch size */
    val nUbatch: Int = 512,
    
    /** Number of threads (0 for auto) */
    val nThreads: Int = 0,
    
    /** Number of GPU layers (0 for CPU only) */
    val nGpuLayers: Int = 0,
    
    /** Use flash attention */
    val flashAttn: Boolean = false,
    
    /** KV cache type for keys */
    val cacheTypeK: String = "f16",
    
    /** KV cache type for values */
    val cacheTypeV: String = "f16",
    
    /** Use mlock */
    val useMlock: Boolean = true,
    
    /** Use mmap */
    val useMmap: Boolean = true,
    
    /** Load only vocabulary */
    val vocabOnly: Boolean = false,
    
    /** Path to LoRA adapter */
    val lora: String? = null,
    
    /** LoRA scaling factor */
    val loraScaled: Float = 1.0f,
    
    /** List of LoRA adapters */
    val loraList: List<String>? = null,
    
    /** RoPE frequency base */
    val ropeFreqBase: Float = 0.0f,
    
    /** RoPE frequency scale */
    val ropeFreqScale: Float = 0.0f,
    
    /** Pooling type (-1 for default) */
    val poolingType: Int = -1
) 