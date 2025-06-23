import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:cactus/cactus.dart';

class CactusService {
  CactusContext? _cactusContext;

  // ValueNotifiers for UI updates
  final ValueNotifier<List<ChatMessage>> chatMessages = ValueNotifier([]);
  final ValueNotifier<bool> isLoading = ValueNotifier(true);
  final ValueNotifier<bool> isBenchmarking = ValueNotifier(false);
  final ValueNotifier<String> statusMessage = ValueNotifier('Initializing...');
  final ValueNotifier<String?> initError = ValueNotifier(null);
  final ValueNotifier<double?> downloadProgress = ValueNotifier(null);
  final ValueNotifier<BenchResult?> benchResult = ValueNotifier(null);
  final ValueNotifier<String?> imagePathForNextMessage = ValueNotifier(null);
  final ValueNotifier<String?> stagedAssetPath = ValueNotifier(null); // For image picker display
  
  // Performance tracking
  final ValueNotifier<ConversationResult?> lastConversationResult = ValueNotifier(null);
  final ValueNotifier<bool> isConversationActive = ValueNotifier(false);

  bool _lastInteractionWasMultimodal = false;

  Future<void> initialize() async {
    isLoading.value = true;
    initError.value = null;
    statusMessage.value = 'Initializing plugin...';
    downloadProgress.value = null; 

    // Cactus usage 
    try {
      final params = CactusInitParams(
        modelUrl: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf', 
        mmprojUrl: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf',

        onInitProgress: (progress, status, isError) {
          statusMessage.value = status; 
          downloadProgress.value = progress; 

          if (isError) {
            initError.value = status;
            isLoading.value = false;
          }
        },
      );

      _cactusContext = await CactusContext.init(params);
      
      if (initError.value == null) { 
        isLoading.value = false;
        statusMessage.value = 'Cactus initialized successfully!';
        await runBenchmark();
      } else {
        statusMessage.value = 'Initialization failed: ${initError.value}';
      }

    } on CactusInitializationException catch (e) {
      initError.value = "Initialization Error: ${e.message}";
      statusMessage.value = 'Failed to initialize context: ${e.message}';
      isLoading.value = false;
      debugPrint("Cactus Initialization Exception: ${e.toString()}");
    } catch (e) {
      initError.value ??= "An unexpected error occurred: ${e.toString()}";
      statusMessage.value = 'Initialization failed: ${initError.value}';
      isLoading.value = false;
      debugPrint("Generic Exception during Cactus Init: ${e.toString()}");
    }
  }

  Future<void> sendMessage(String userInput) async {
    if (_cactusContext == null) {
      chatMessages.value = [...chatMessages.value, ChatMessage(role: 'system', content: 'Error: CactusContext not initialized.')];
      return;
    }

    final userMessage = ChatMessage(role: 'user', content: userInput);
    final String? imagePathToSend = imagePathForNextMessage.value;
    bool isCurrentlyMultimodal = imagePathToSend != null;

    imagePathForNextMessage.value = null;
    stagedAssetPath.value = null;

    // Handle mode switching
    if (_lastInteractionWasMultimodal != isCurrentlyMultimodal) {
      debugPrint("Mode switch: ${_lastInteractionWasMultimodal ? 'multimodal' : 'text'} -> ${isCurrentlyMultimodal ? 'multimodal' : 'text'}, clearing conversation state");
      _cactusContext!.clearConversation();
      chatMessages.value = [];
      lastConversationResult.value = null;
      isConversationActive.value = false;
    }
    _lastInteractionWasMultimodal = isCurrentlyMultimodal;

    // Add user message to chat UI
    List<ChatMessage> workingHistory = List.from(chatMessages.value);
    workingHistory.add(userMessage);
    workingHistory.add(ChatMessage(role: 'assistant', content: ''));
    chatMessages.value = workingHistory;
    isLoading.value = true;

    final stopwatch = Stopwatch()..start();
    String currentAssistantResponse = "";

    try {
      if (isCurrentlyMultimodal) {
        debugPrint("Using multimodal completion");
        
        // For multimodal, still use the traditional completion API with manual history
        List<ChatMessage> completionHistory = List.from(workingHistory);
        if (completionHistory.isNotEmpty && completionHistory.last.role == 'assistant' && completionHistory.last.content.isEmpty) {
          completionHistory.removeLast();
        }

        final completionParams = CactusCompletionParams(
          messages: completionHistory,
          stopSequences: ['<|im_end|>', '<end_of_utterance>'],
          temperature: 0.3,
          topK: 10,
          topP: 0.9,
          penaltyRepeat: 1.1,
          onNewToken: (String token) {
            if (!isLoading.value) return false;
            if (token == '<|im_end|>') return false;

            if (token.isNotEmpty) {
              currentAssistantResponse += token;
              final List<ChatMessage> streamingMessages = List.from(chatMessages.value);
              if (streamingMessages.isNotEmpty && streamingMessages.last.role == 'assistant') {
                streamingMessages[streamingMessages.length - 1] = ChatMessage(
                  role: 'assistant',
                  content: currentAssistantResponse,
                );
                chatMessages.value = streamingMessages;
              }
            }
            return true;
          },
        );

        final result = await _cactusContext!.multimodalCompletion(completionParams, [imagePathToSend]);
        
        // Clear conversation performance data for multimodal (doesn't use conversation API)
        lastConversationResult.value = null;
        isConversationActive.value = false;
        
        String finalCleanText = result.text.trim();
        if (finalCleanText.isEmpty && currentAssistantResponse.trim().isNotEmpty) {
          finalCleanText = currentAssistantResponse.trim();
        }

        final List<ChatMessage> finalMessages = List.from(chatMessages.value);
        if (finalMessages.isNotEmpty && finalMessages.last.role == 'assistant') {
          finalMessages[finalMessages.length - 1] = ChatMessage(
            role: 'assistant',
            content: finalCleanText.isNotEmpty ? finalCleanText : "(No response generated)",
          );
          chatMessages.value = finalMessages;
        }

      } else {
        debugPrint("Using optimized conversation API");
        
        // For text mode, use the new conversation management API
        final result = await _cactusContext!.continueConversation(userInput, maxTokens: 256);
        
        stopwatch.stop();
        
        // Store performance results for UI display
        lastConversationResult.value = result;
        isConversationActive.value = _cactusContext!.isConversationActive();
        
        debugPrint("[PERFORMANCE] TTFT: ${result.timeToFirstToken}ms, "
                   "Total: ${result.totalTime}ms, "
                   "Tokens: ${result.tokensGenerated}, "
                   "Speed: ${result.tokensPerSecond.toStringAsFixed(1)} tok/s");
        debugPrint("[TIMING] Flutter overhead: ${stopwatch.elapsedMilliseconds - result.totalTime}ms");
        debugPrint("[STATUS] Conversation active: ${isConversationActive.value}");
        
        // Update chat UI with final result
        final List<ChatMessage> finalMessages = List.from(chatMessages.value);
        if (finalMessages.isNotEmpty && finalMessages.last.role == 'assistant') {
          finalMessages[finalMessages.length - 1] = ChatMessage(
            role: 'assistant',
            content: result.text.isNotEmpty ? result.text : "(No response generated)",
          );
          chatMessages.value = finalMessages;
        }
      }

    } on CactusCompletionException catch (e) {
      _addErrorMessageToChat("Completion Error: ${e.message}");
      debugPrint("Cactus Completion Exception: ${e.toString()}");
    } catch (e) {
      _addErrorMessageToChat("An unexpected error occurred during completion: ${e.toString()}");
      debugPrint("Generic Exception during completion: ${e.toString()}");
    } finally {
      isLoading.value = false;
    }
  }

  void _addErrorMessageToChat(String errorMessage) {
      final List<ChatMessage> errorMessages = List.from(chatMessages.value);
      if (errorMessages.isNotEmpty && errorMessages.last.role == 'assistant') {
        errorMessages[errorMessages.length - 1] = ChatMessage(
          role: 'assistant',
          content: errorMessage,
        );
      } else {
        errorMessages.add(ChatMessage(role: 'system', content: errorMessage));
      }
      chatMessages.value = errorMessages;
  }

  Future<void> runBenchmark() async {
    if (_cactusContext == null) return;
    isBenchmarking.value = true;
    statusMessage.value = 'Running benchmark...';
    try {
      final result = _cactusContext!.bench();
      benchResult.value = result;
      statusMessage.value = 'Benchmark complete.';
    } catch (e) {
      debugPrint("Benchmark Error: ${e.toString()}");
      statusMessage.value = 'Benchmark failed: ${e.toString()}';
    } finally {
      isBenchmarking.value = false;
    }
  }

  void stageImageFromAsset(String assetPath, String tempFilename) async {
      try {
        final ByteData assetData = await rootBundle.load(assetPath); // Requires services.dart
        final Directory tempDir = await getTemporaryDirectory();
        final String tempFilePath = '${tempDir.path}/$tempFilename'; 
        final File tempFile = File(tempFilePath);
        await tempFile.writeAsBytes(assetData.buffer.asUint8List(), flush: true);
        imagePathForNextMessage.value = tempFilePath;
        stagedAssetPath.value = assetPath;
      } catch (e) {
        debugPrint("Error staging image from asset: $e");
        statusMessage.value = "Error staging image: $e";
      }
  }

  void clearStagedImage() {
    imagePathForNextMessage.value = null;
    stagedAssetPath.value = null;
  }

  void clearConversation() {
    chatMessages.value = [];
    _lastInteractionWasMultimodal = false;
    _cactusContext?.clearConversation();
    lastConversationResult.value = null;
    isConversationActive.value = false;
    debugPrint("Conversation history and native conversation state cleared, interaction mode reset");
  }

  Future<String> generateSimpleResponse(String userInput, {int maxTokens = 200}) async {
    if (_cactusContext == null) {
      throw Exception('CactusContext not initialized');
    }
    
    try {
      return await _cactusContext!.generateResponse(userInput, maxTokens: maxTokens);
    } catch (e) {
      debugPrint("Error in generateSimpleResponse: $e");
      rethrow;
    }
  }

  void dispose() {
    _cactusContext?.free();
    chatMessages.dispose();
    isLoading.dispose();
    isBenchmarking.dispose();
    statusMessage.dispose();
    initError.dispose();
    downloadProgress.dispose();
    benchResult.dispose();
    imagePathForNextMessage.dispose();
    stagedAssetPath.dispose();
    lastConversationResult.dispose();
    isConversationActive.dispose();
  }
} 