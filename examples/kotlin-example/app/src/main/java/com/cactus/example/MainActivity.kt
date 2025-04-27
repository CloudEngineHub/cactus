package com.cactus.example

import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.cactus.kotlin.Cactus
import com.cactus.kotlin.LlamaContext
import com.cactus.kotlin.listeners.CompletionListener
import com.cactus.kotlin.listeners.LoadProgressListener
import com.cactus.kotlin.models.CompletionParams
import com.cactus.kotlin.models.ContextParams
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private val TAG = "CactusExample"
    
    // UI Elements
    private lateinit var modelPathInput: TextInputEditText
    private lateinit var promptInput: TextInputEditText
    private lateinit var loadButton: Button
    private lateinit var generateButton: Button
    private lateinit var clearButton: Button
    private lateinit var outputText: TextView
    private lateinit var loadProgress: ProgressBar
    private lateinit var statsText: TextView
    private lateinit var outputScrollView: ScrollView
    
    // Llama Context
    private var llamaContext: LlamaContext? = null
    private var isGenerating = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize UI elements
        modelPathInput = findViewById(R.id.modelPathInput)
        promptInput = findViewById(R.id.promptInput)
        loadButton = findViewById(R.id.loadButton)
        generateButton = findViewById(R.id.generateButton)
        clearButton = findViewById(R.id.clearButton)
        outputText = findViewById(R.id.outputText)
        loadProgress = findViewById(R.id.loadProgress)
        statsText = findViewById(R.id.statsText)
        outputScrollView = findViewById(R.id.outputScrollView)
        
        // Set up UI behavior
        setupListeners()
        
        // Check architecture compatibility
        if (!Cactus.isArchitectureSupported()) {
            Snackbar.make(
                findViewById(android.R.id.content),
                "This device architecture is not supported. Only 64-bit is supported.",
                Snackbar.LENGTH_LONG
            ).show()
            loadButton.isEnabled = false
        }
    }
    
    private fun setupListeners() {
        loadButton.setOnClickListener {
            val modelPath = modelPathInput.text.toString()
            if (modelPath.isNotEmpty()) {
                loadModel(modelPath)
            } else {
                Snackbar.make(
                    findViewById(android.R.id.content),
                    "Please specify a model path",
                    Snackbar.LENGTH_SHORT
                ).show()
            }
        }
        
        generateButton.setOnClickListener {
            val prompt = promptInput.text.toString()
            if (prompt.isNotEmpty() && llamaContext != null) {
                if (!isGenerating) {
                    generateCompletion(prompt)
                } else {
                    stopGeneration()
                }
            }
        }
        
        clearButton.setOnClickListener {
            outputText.text = ""
            statsText.text = ""
        }
    }
    
    private fun loadModel(modelPath: String) {
        // Disable UI during loading
        loadButton.isEnabled = false
        loadProgress.visibility = View.VISIBLE
        loadProgress.progress = 0
        outputText.text = "Loading model: $modelPath\n"
        
        // Create a progress listener
        val progressListener = object : LoadProgressListener {
            override fun onProgress(progress: Int) {
                runOnUiThread {
                    loadProgress.progress = progress
                    if (progress % 10 == 0) {
                        outputText.append("Loading: $progress%\n")
                        scrollToBottom()
                    }
                }
            }
            
            override fun onComplete() {
                runOnUiThread {
                    outputText.append("Model loaded successfully!\n")
                    scrollToBottom()
                }
            }
            
            override fun onError(error: String) {
                runOnUiThread {
                    outputText.append("Error loading model: $error\n")
                    scrollToBottom()
                }
            }
        }
        
        // Set up model parameters
        val contextParams = ContextParams(
            model = modelPath,
            nCtx = 2048,
            nThreads = 4
        )
        
        // Load the model in a background thread
        CoroutineScope(Dispatchers.Main).launch {
            try {
                // Create the context
                llamaContext = Cactus.createContext(
                    params = contextParams,
                    androidContext = this@MainActivity,
                    progressListener = progressListener
                )
                
                // Update UI when model is loaded
                withContext(Dispatchers.Main) {
                    loadButton.isEnabled = true
                    loadProgress.visibility = View.GONE
                    generateButton.isEnabled = true
                    
                    val modelInfo = llamaContext!!.getModelInfo()
                    outputText.append("""
                        |Model Info:
                        |Name: ${modelInfo.name}
                        |Architecture: ${modelInfo.architecture}
                        |Parameters: ${modelInfo.params / 1_000_000}M
                        |Context size: ${modelInfo.contextSize}
                        |
                    """.trimMargin())
                    scrollToBottom()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                withContext(Dispatchers.Main) {
                    loadButton.isEnabled = true
                    loadProgress.visibility = View.GONE
                    outputText.append("Failed to load model: ${e.message}\n")
                    scrollToBottom()
                    
                    Snackbar.make(
                        findViewById(android.R.id.content),
                        "Failed to load model: ${e.message}",
                        Snackbar.LENGTH_LONG
                    ).show()
                }
            }
        }
    }
    
    private fun generateCompletion(prompt: String) {
        isGenerating = true
        generateButton.text = "Stop"
        
        // Set up completion parameters
        val completionParams = CompletionParams(
            prompt = prompt,
            maxTokens = 512,
            temperature = 0.7f,
            topP = 0.9f,
            repetitionPenalty = 1.1f
        )
        
        // Token streaming listener
        val completionListener = object : CompletionListener {
            override fun onToken(tokenText: String, isPartial: Boolean, tokenId: Int) {
                runOnUiThread {
                    // Append token text to the output
                    outputText.append(tokenText)
                    scrollToBottom()
                }
            }
            
            override fun onComplete() {
                runOnUiThread {
                    isGenerating = false
                    generateButton.text = "Generate"
                }
            }
            
            override fun onError(error: String) {
                runOnUiThread {
                    outputText.append("\nError: $error\n")
                    scrollToBottom()
                    isGenerating = false
                    generateButton.text = "Generate"
                }
            }
        }
        
        // Generate text
        outputText.append("\n\n$prompt")
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    llamaContext?.completion(completionParams, completionListener)
                }
                
                result?.let {
                    statsText.text = """
                        Time: ${it.timeTaken / 1000.0}s
                        Tokens: ${it.completionTokens}
                        Speed: ${it.tokensPerSecond.toInt()} t/s
                    """.trimIndent()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Error during generation", e)
                outputText.append("\nError during generation: ${e.message}\n")
                scrollToBottom()
                
                isGenerating = false
                generateButton.text = "Generate"
            }
        }
    }
    
    private fun stopGeneration() {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                withContext(Dispatchers.IO) {
                    llamaContext?.stopCompletion()
                }
                isGenerating = false
                generateButton.text = "Generate"
                outputText.append("\n[Generation stopped]\n")
                scrollToBottom()
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping generation", e)
            }
        }
    }
    
    private fun scrollToBottom() {
        outputScrollView.post {
            outputScrollView.fullScroll(View.FOCUS_DOWN)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Release model resources
        CoroutineScope(Dispatchers.IO).launch {
            llamaContext?.release()
        }
    }
} 