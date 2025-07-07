import 'dart:async';

import './types.dart';
import './context.dart';
import './telemetary.dart';

class CactusLM {
  CactusContext? _context;
  CactusInitParams? _initParams;
  
  CactusLM._();

  static Future<CactusLM> init({
    required String modelUrl,
    String? modelFilename,
    String? chatTemplate,
    int contextSize = 2048,
    int gpuLayers = 0,
    int threads = 4,
    bool generateEmbeddings = false,
    CactusProgressCallback? onProgress,
  }) async {
    final lm = CactusLM._();
    
    final initParams = CactusInitParams(
      modelUrl: modelUrl,
      modelFilename: modelFilename,
      chatTemplate: chatTemplate,
      contextSize: contextSize,
      gpuLayers: gpuLayers,
      threads: threads,
      generateEmbeddings: generateEmbeddings,
      onInitProgress: onProgress,
    );
    
    try {
      lm._context = await CactusContext.init(initParams);
      lm._initParams = initParams;
    } catch (e) {
      CactusTelemetry.error(e, initParams);
      rethrow;
    }
    
    return lm;
  }

  Future<CactusCompletionResult> completion(
    List<ChatMessage> messages, {
    int maxTokens = 256,
    double? temperature,
    int? topK,
    double? topP,
    List<String>? stopSequences,
    CactusTokenCallback? onToken,
  }) async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    
    final startTime = DateTime.now();
    bool firstTokenReceived = false;
    DateTime? firstTokenTime;
    
    // Wrap the callback to capture first token timing
    CactusTokenCallback? wrappedCallback;
    if (onToken != null) {
      wrappedCallback = (String token) {
        if (!firstTokenReceived) {
          firstTokenTime = DateTime.now();
          firstTokenReceived = true;
        }
        return onToken(token);
      };
    }
    
    final result = await _context!.completion(
      CactusCompletionParams(
        messages: messages,
        maxPredictedTokens: maxTokens,
        temperature: temperature,
        topK: topK,
        topP: topP,
        stopSequences: stopSequences,
        onNewToken: wrappedCallback,
      ),
    );
    
    // Track telemetry after completion
    if (_initParams != null) {
      final endTime = DateTime.now();
      final totalTime = endTime.difference(startTime).inMilliseconds;
      final tokPerSec = totalTime > 0 ? (result.tokensPredicted * 1000.0) / totalTime : null;
      final ttft = firstTokenTime != null ? firstTokenTime!.difference(startTime).inMilliseconds : null;
      
      CactusTelemetry.track({
        'event': 'completion',
        'tok_per_sec': tokPerSec,
        'toks_generated': result.tokensPredicted,
        'ttft': ttft,
      }, _initParams!);
    }
    
    return result;
  }

  Future<List<double>> embedding(String text) async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    return await _context!.embedding(text);
  }

  Future<List<int>> tokenize(String text) async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    return await _context!.tokenize(text);
  }

  Future<String> detokenize(List<int> tokens) async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    return await _context!.detokenize(tokens);
  }

  Future<void> applyLoraAdapters(List<LoraAdapterInfo> adapters) async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    await _context!.applyLoraAdapters(adapters);
  }

  Future<void> removeLoraAdapters() async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    await _context!.removeLoraAdapters();
  }

  Future<List<LoraAdapterInfo>> getLoadedLoraAdapters() async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    return await _context!.getLoadedLoraAdapters();
  }

  Future<void> rewind() async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    await _context!.rewind();
  }

  Future<void> stopCompletion() async {
    if (_context == null) throw CactusException('CactusLM not initialized');
    await _context!.stopCompletion();
  }

  void dispose() {
    _context?.release();
    _context = null;
  }
} 