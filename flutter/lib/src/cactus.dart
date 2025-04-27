import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:io';

import 'package:flutter/services.dart';

import 'types.dart';
import 'model_downloader.dart';
import 'llama_context.dart';

/// Events for the Cactus plugin
class CactusEvents {
  static const String EVENT_ON_INIT_CONTEXT_PROGRESS = '@Cactus_onInitContextProgress';
  static const String EVENT_ON_TOKEN = '@Cactus_onToken';
  static const String EVENT_ON_NATIVE_LOG = '@Cactus_onNativeLog';
  
  static const EventChannel _eventChannel = EventChannel('com.cactus.flutter/events');
  
  static Stream<Map<String, dynamic>>? _eventStream;
  
  static Stream<Map<String, dynamic>> get _events {
    _eventStream ??= _eventChannel.receiveBroadcastStream().map((event) => 
        Map<String, dynamic>.from(event));
    return _eventStream!;
  }
  
  static Stream<Map<String, dynamic>> get onInitContextProgress =>
      _events.where((event) => event['type'] == EVENT_ON_INIT_CONTEXT_PROGRESS);
      
  static Stream<Map<String, dynamic>> get onToken =>
      _events.where((event) => event['type'] == EVENT_ON_TOKEN);
      
  static Stream<Map<String, dynamic>> get onNativeLog =>
      _events.where((event) => event['type'] == EVENT_ON_NATIVE_LOG);
}

/// Native methods for the Cactus plugin
class CactusNative {
  static const MethodChannel _channel = MethodChannel('com.cactus.flutter/methods');
  
  static List<Function(String level, String text)> _logListeners = [];

  /// Toggle native logging
  static Future<void> toggleNativeLog(bool enabled) async {
    try {
      await _channel.invokeMethod('toggleNativeLog', {'enabled': enabled});
    } catch (e) {
      print('Failed to toggle native log: $e');
    }
  }
  
  /// Add a native log listener
  static void addNativeLogListener(Function(String level, String text) listener) {
    _logListeners.add(listener);
    
    // Make sure at least one listener is registered
    if (_logListeners.length == 1) {
      // Set up the stream listener if this is the first one
      CactusEvents.onNativeLog.listen((event) {
        final level = event['level'] as String;
        final text = event['text'] as String;
        for (final listener in _logListeners) {
          listener(level, text);
        }
      });
      
      // Trigger the native side
      toggleNativeLog(false);
    }
  }
  
  /// Remove a native log listener
  static void removeNativeLogListener(Function(String level, String text) listener) {
    _logListeners.remove(listener);
  }
  
  /// Set the maximum number of contexts that can be loaded at once
  static Future<void> setContextLimit(int limit) async {
    await _channel.invokeMethod('setContextLimit', {'limit': limit});
  }
  
  /// Load model info
  static Future<Map<String, dynamic>> modelInfo(String path, {List<String>? skip}) async {
    final result = await _channel.invokeMethod('modelInfo', {
      'path': path,
      'skip': skip,
    });
    return Map<String, dynamic>.from(result);
  }
  
  /// Initialize a context
  static Future<Map<String, dynamic>> initContext(int contextId, Map<String, dynamic> params) async {
    final result = await _channel.invokeMethod('initContext', {
      'contextId': contextId,
      'params': params,
    });
    return Map<String, dynamic>.from(result);
  }
  
  /// Get formatted chat
  static Future<dynamic> getFormattedChat(
    int contextId,
    String messages,
    String? chatTemplate,
    Map<String, dynamic> params,
  ) async {
    final result = await _channel.invokeMethod('getFormattedChat', {
      'contextId': contextId,
      'messages': messages,
      'chatTemplate': chatTemplate,
      'params': params,
    });
    
    if (result is String) {
      return result;
    } else if (result is Map) {
      return JinjaFormattedChatResult.fromJson(Map<String, dynamic>.from(result));
    }
    
    throw Exception('Unexpected result type from getFormattedChat');
  }
  
  /// Load a session
  static Future<SessionLoadResult> loadSession(int contextId, String filepath) async {
    final result = await _channel.invokeMethod('loadSession', {
      'contextId': contextId,
      'filepath': filepath,
    });
    return SessionLoadResult.fromJson(Map<String, dynamic>.from(result));
  }
  
  /// Save a session
  static Future<int> saveSession(int contextId, String filepath, int size) async {
    final result = await _channel.invokeMethod('saveSession', {
      'contextId': contextId,
      'filepath': filepath,
      'size': size,
    });
    return result as int;
  }
  
  /// Perform completion
  static Future<CompletionResult> completion(int contextId, Map<String, dynamic> params) async {
    final result = await _channel.invokeMethod('completion', {
      'contextId': contextId,
      'params': params,
    });
    return CompletionResult.fromJson(Map<String, dynamic>.from(result));
  }
  
  /// Stop completion
  static Future<void> stopCompletion(int contextId) async {
    await _channel.invokeMethod('stopCompletion', {
      'contextId': contextId,
    });
  }
  
  /// Tokenize text
  static Future<TokenizeResult> tokenize(int contextId, String text) async {
    final result = await _channel.invokeMethod('tokenize', {
      'contextId': contextId,
      'text': text,
    });
    return TokenizeResult.fromJson(Map<String, dynamic>.from(result));
  }
  
  /// Detokenize tokens
  static Future<String> detokenize(int contextId, List<int> tokens) async {
    final result = await _channel.invokeMethod('detokenize', {
      'contextId': contextId,
      'tokens': tokens,
    });
    return result as String;
  }
  
  /// Get embeddings
  static Future<EmbeddingResult> embedding(
    int contextId,
    String text,
    Map<String, dynamic> params,
  ) async {
    final result = await _channel.invokeMethod('embedding', {
      'contextId': contextId,
      'text': text,
      'params': params,
    });
    return EmbeddingResult.fromJson(Map<String, dynamic>.from(result));
  }
  
  /// Run benchmarks
  static Future<BenchResult> bench(
    int contextId,
    int pp,
    int tg,
    int pl,
    int nr,
  ) async {
    final result = await _channel.invokeMethod('bench', {
      'contextId': contextId,
      'pp': pp,
      'tg': tg,
      'pl': pl,
      'nr': nr,
    });
    
    if (result is String) {
      final map = jsonDecode(result) as Map<String, dynamic>;
      return BenchResult.fromJson(map);
    }
    
    return BenchResult.fromJson(Map<String, dynamic>.from(result));
  }
  
  /// Apply LoRA adapters
  static Future<void> applyLoraAdapters(
    int contextId,
    List<LoraAdapter> loraAdapters,
  ) async {
    await _channel.invokeMethod('applyLoraAdapters', {
      'contextId': contextId,
      'loraAdapters': loraAdapters.map((e) => e.toJson()).toList(),
    });
  }
  
  /// Remove LoRA adapters
  static Future<void> removeLoraAdapters(int contextId) async {
    await _channel.invokeMethod('removeLoraAdapters', {
      'contextId': contextId,
    });
  }
  
  /// Get loaded LoRA adapters
  static Future<List<LoraAdapter>> getLoadedLoraAdapters(int contextId) async {
    final result = await _channel.invokeMethod('getLoadedLoraAdapters', {
      'contextId': contextId,
    });
    
    final list = List<Map<String, dynamic>>.from(result);
    return list.map((e) => LoraAdapter.fromJson(e)).toList();
  }
  
  /// Release a context
  static Future<void> releaseContext(int contextId) async {
    await _channel.invokeMethod('releaseContext', {
      'contextId': contextId,
    });
  }
  
  /// Release all contexts
  static Future<void> releaseAllContexts() async {
    await _channel.invokeMethod('releaseAllContexts');
  }
}

/// Generate a random context ID for initialization
int _contextIdRandom() {
  if (Platform.environment['FLUTTER_TEST'] == 'true') {
    return 0;
  }
  return Random().nextInt(100000);
}

/// Load model info
Future<Map<String, dynamic>> loadLlamaModelInfo(String model) async {
  // Try to load the model info
  try {
    return await CactusNative.modelInfo(model);
  } catch (e) {
    // Download the model if it doesn't exist
    if (e is PlatformException && e.code == 'file_not_found') {
      if (model.startsWith('http')) {
        final downloader = ModelDownloader(modelUrl: model);
        final path = await downloader.downloadModelIfNotExists();
        return await CactusNative.modelInfo(path);
      }
    }
    
    // Re-throw any other errors
    rethrow;
  }
}

/// Initialize a Llama context
Future<LlamaContext> initLlama(
  ContextParams params, {
  Function(double progress)? onProgress,
}) async {
  // Set up parameter conversion
  final nativeParams = params.toNative();
  
  // Set up progress tracking if needed
  StreamSubscription? progressSubscription;
  if (onProgress != null) {
    nativeParams['use_progress_callback'] = true;
    
    progressSubscription = CactusEvents.onInitContextProgress.listen((event) {
      final progress = event['progress'] as double;
      onProgress(progress);
    });
  }
  
  try {
    // Generate a random context ID
    final contextId = _contextIdRandom();
    
    // Initialize the context
    final result = await CactusNative.initContext(contextId, nativeParams);
    
    // Map the result to a LlamaContext
    return LlamaContext(
      id: result['contextId'] as int,
      gpu: result['gpu'] as bool,
      reasonNoGPU: result['reasonNoGPU'] as String,
      model: ModelInfo.fromJson(result['model'] as Map<String, dynamic>),
    );
  } finally {
    // Clean up the progress subscription
    await progressSubscription?.cancel();
  }
}

/// Release all loaded Llama contexts
Future<void> releaseAllLlama() async {
  return CactusNative.releaseAllContexts();
}

/// Toggle native logging
Future<void> toggleNativeLog(bool enabled) async {
  return CactusNative.toggleNativeLog(enabled);
}

/// Add a native log listener
/// Returns a function that can be called to remove the listener
Function addNativeLogListener(Function(String level, String text) listener) {
  CactusNative.addNativeLogListener(listener);
  return () => CactusNative.removeNativeLogListener(listener);
}

/// Set the context limit
Future<void> setContextLimit(int limit) async {
  return CactusNative.setContextLimit(limit);
} 