package com.cactus.example;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.cactus.Cactus;
import com.cactus.LlamaContext;
import com.cactus.listeners.CompletionListener;
import com.cactus.listeners.LoadProgressListener;
import com.cactus.models.CompletionParams;
import com.cactus.models.ContextParams;
import com.cactus.models.ModelInfo;
import com.google.android.material.snackbar.Snackbar;
import com.google.android.material.textfield.TextInputEditText;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "CactusExample";
    
    // UI Elements
    private TextInputEditText modelPathInput;
    private TextInputEditText promptInput;
    private Button loadButton;
    private Button generateButton;
    private Button clearButton;
    private TextView outputText;
    private ProgressBar loadProgress;
    private TextView statsText;
    private ScrollView outputScrollView;
    
    // Llama Context
    private LlamaContext llamaContext;
    private boolean isGenerating = false;
    
    // Executors
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize UI elements
        modelPathInput = findViewById(R.id.modelPathInput);
        promptInput = findViewById(R.id.promptInput);
        loadButton = findViewById(R.id.loadButton);
        generateButton = findViewById(R.id.generateButton);
        clearButton = findViewById(R.id.clearButton);
        outputText = findViewById(R.id.outputText);
        loadProgress = findViewById(R.id.loadProgress);
        statsText = findViewById(R.id.statsText);
        outputScrollView = findViewById(R.id.outputScrollView);
        
        // Set up UI behavior
        setupListeners();
        
        // Check architecture compatibility
        if (!Cactus.isArchitectureSupported()) {
            Snackbar.make(
                findViewById(android.R.id.content),
                "This device architecture is not supported. Only 64-bit is supported.",
                Snackbar.LENGTH_LONG
            ).show();
            loadButton.setEnabled(false);
        }
    }
    
    private void setupListeners() {
        loadButton.setOnClickListener(v -> {
            String modelPath = modelPathInput.getText().toString();
            if (!modelPath.isEmpty()) {
                loadModel(modelPath);
            } else {
                Snackbar.make(
                    findViewById(android.R.id.content),
                    "Please specify a model path",
                    Snackbar.LENGTH_SHORT
                ).show();
            }
        });
        
        generateButton.setOnClickListener(v -> {
            String prompt = promptInput.getText().toString();
            if (!prompt.isEmpty() && llamaContext != null) {
                if (!isGenerating) {
                    generateCompletion(prompt);
                } else {
                    stopGeneration();
                }
            }
        });
        
        clearButton.setOnClickListener(v -> {
            outputText.setText("");
            statsText.setText("");
        });
    }
    
    private void loadModel(String modelPath) {
        // Disable UI during loading
        loadButton.setEnabled(false);
        loadProgress.setVisibility(View.VISIBLE);
        loadProgress.setProgress(0);
        outputText.setText("Loading model: " + modelPath + "\n");
        
        // Create a progress listener
        LoadProgressListener progressListener = new LoadProgressListener() {
            @Override
            public void onProgress(int progress) {
                mainHandler.post(() -> {
                    loadProgress.setProgress(progress);
                    if (progress % 10 == 0) {
                        outputText.append("Loading: " + progress + "%\n");
                        scrollToBottom();
                    }
                });
            }
            
            @Override
            public void onComplete() {
                mainHandler.post(() -> {
                    outputText.append("Model loaded successfully!\n");
                    scrollToBottom();
                });
            }
            
            @Override
            public void onError(String error) {
                mainHandler.post(() -> {
                    outputText.append("Error loading model: " + error + "\n");
                    scrollToBottom();
                });
            }
        };
        
        // Set up model parameters
        ContextParams contextParams = new ContextParams(
            modelPath,  // model path
            2048,       // context size
            4           // thread count
        );
        
        // Load the model in a background thread
        executor.execute(() -> {
            try {
                // Create the context
                llamaContext = Cactus.createContext(
                    contextParams,
                    MainActivity.this,  // Android context
                    progressListener
                );
                
                // Update UI when model is loaded
                mainHandler.post(() -> {
                    loadButton.setEnabled(true);
                    loadProgress.setVisibility(View.GONE);
                    generateButton.setEnabled(true);
                    
                    try {
                        ModelInfo modelInfo = llamaContext.getModelInfo();
                        StringBuilder infoBuilder = new StringBuilder();
                        infoBuilder.append("Model Info:\n")
                                  .append("Name: ").append(modelInfo.getName()).append("\n")
                                  .append("Architecture: ").append(modelInfo.getArchitecture()).append("\n")
                                  .append("Parameters: ").append(modelInfo.getParams() / 1_000_000).append("M\n")
                                  .append("Context size: ").append(modelInfo.getContextSize()).append("\n");
                        outputText.append(infoBuilder.toString());
                        scrollToBottom();
                    } catch (Exception e) {
                        Log.e(TAG, "Error getting model info", e);
                    }
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Error loading model", e);
                mainHandler.post(() -> {
                    loadButton.setEnabled(true);
                    loadProgress.setVisibility(View.GONE);
                    outputText.append("Failed to load model: " + e.getMessage() + "\n");
                    scrollToBottom();
                    
                    Snackbar.make(
                        findViewById(android.R.id.content),
                        "Failed to load model: " + e.getMessage(),
                        Snackbar.LENGTH_LONG
                    ).show();
                });
            }
        });
    }
    
    private void generateCompletion(String prompt) {
        isGenerating = true;
        generateButton.setText("Stop");
        
        // Set up completion parameters
        CompletionParams completionParams = new CompletionParams(
            prompt,     // prompt
            512,        // max tokens
            0.7f,       // temperature
            0.9f,       // top p
            1.1f        // repetition penalty
        );
        
        // Token streaming listener
        CompletionListener completionListener = new CompletionListener() {
            @Override
            public void onToken(String tokenText, boolean isPartial, int tokenId) {
                mainHandler.post(() -> {
                    // Append token text to the output
                    outputText.append(tokenText);
                    scrollToBottom();
                });
            }
            
            @Override
            public void onComplete() {
                mainHandler.post(() -> {
                    isGenerating = false;
                    generateButton.setText("Generate");
                });
            }
            
            @Override
            public void onError(String error) {
                mainHandler.post(() -> {
                    outputText.append("\nError: " + error + "\n");
                    scrollToBottom();
                    isGenerating = false;
                    generateButton.setText("Generate");
                });
            }
        };
        
        // Generate text
        outputText.append("\n\n" + prompt);
        executor.execute(() -> {
            try {
                final long startTime = System.currentTimeMillis();
                final int[] tokenCount = {0};
                
                // Start completion with token callback
                llamaContext.completion(completionParams, completionListener, result -> {
                    tokenCount[0] = result.getCompletionTokens();
                    long timeTaken = System.currentTimeMillis() - startTime;
                    float tokensPerSecond = tokenCount[0] / (timeTaken / 1000.0f);
                    
                    mainHandler.post(() -> {
                        statsText.setText(
                            "Time: " + (timeTaken / 1000.0) + "s\n" +
                            "Tokens: " + tokenCount[0] + "\n" +
                            "Speed: " + (int)tokensPerSecond + " t/s"
                        );
                    });
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Error during generation", e);
                mainHandler.post(() -> {
                    outputText.append("\nError during generation: " + e.getMessage() + "\n");
                    scrollToBottom();
                    
                    isGenerating = false;
                    generateButton.setText("Generate");
                });
            }
        });
    }
    
    private void stopGeneration() {
        executor.execute(() -> {
            try {
                llamaContext.stopCompletion();
                mainHandler.post(() -> {
                    isGenerating = false;
                    generateButton.setText("Generate");
                    outputText.append("\n[Generation stopped]\n");
                    scrollToBottom();
                });
            } catch (Exception e) {
                Log.e(TAG, "Error stopping generation", e);
            }
        });
    }
    
    private void scrollToBottom() {
        outputScrollView.post(() -> outputScrollView.fullScroll(View.FOCUS_DOWN));
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Release model resources
        if (llamaContext != null) {
            executor.execute(() -> {
                try {
                    llamaContext.release();
                } catch (Exception e) {
                    Log.e(TAG, "Error releasing context", e);
                }
            });
        }
        executor.shutdown();
    }
} 