import 'dart:convert';
import 'dart:async';
import 'dart:io';

import 'package:path/path.dart' as p;

import 'chat.dart';
import 'cactus.dart';
import 'types.dart';

/// A class representing a Llama context
class LlamaContext {
  final int id;
  final bool gpu;
  final String reasonNoGPU;
  final ModelInfo model;
  
  /// Create a new LlamaContext
  /// 
  /// This constructor is typically not called directly. Use [initLlama] instead.
  LlamaContext({
    required this.id,
    required this.gpu,
    required this.reasonNoGPU,
    required this.model,
  });
  
  /// Load a saved session from a file
  Future<SessionLoadResult> loadSession(String filepath) async {
    String path = filepath;
    if (path.startsWith('file://')) {
      path = path.substring(7);
    }
    return CactusNative.loadSession(id, path);
  }
  
  /// Save the current session to a file
  Future<int> saveSession(
    String filepath, {
    int? tokenSize,
  }) async {
    return CactusNative.saveSession(
      id,
      filepath,
      tokenSize ?? -1,
    );
  }
  
  /// Check if LlamaChat format is supported
  bool isLlamaChatSupported() {
    return model.chatTemplates.llamaChat;
  }
  
  /// Check if Jinja templates are supported
  bool isJinjaSupported() {
    final minja = model.chatTemplates.minja;
    return (minja?.toolUse ?? false) || (minja?.default_ ?? false);
  }
  
  /// Get formatted chat prompt
  Future<dynamic> getFormattedChat(
    List<ChatMessage> messages, {
    String? template,
    bool? jinja,
    CompletionResponseFormat? responseFormat,
    Map<String, dynamic>? tools,
    Map<String, dynamic>? parallelToolCalls,
    String? toolChoice,
  }) async {
    final chat = formatChatToJson(messages);
    final useJinja = isJinjaSupported() && (jinja ?? false);
    String? tmpl;
    
    if (template != null) {
      tmpl = template;
    } else if (!isLlamaChatSupported() && !useJinja) {
      tmpl = 'chatml';
    }
    
    final jsonSchema = _getJsonSchema(responseFormat);
    
    return CactusNative.getFormattedChat(
      id,
      chat,
      tmpl,
      {
        'jinja': useJinja,
        'json_schema': jsonSchema != null ? jsonEncode(jsonSchema) : null,
        'tools': tools != null ? jsonEncode(tools) : null,
        'parallel_tool_calls': parallelToolCalls != null ? jsonEncode(parallelToolCalls) : null,
        'tool_choice': toolChoice,
      },
    );
  }
  
  /// Complete a prompt
  Future<CompletionResult> completion(
    CompletionParams params, {
    Function(TokenData)? callback,
  }) async {
    final nativeParams = {
      ...params.toNative(emitPartialCompletion: callback != null),
    };
    
    if (params.messages != null) {
      // Messages take precedence over prompt
      final formattedResult = await getFormattedChat(
        params.messages!,
        template: params.chatTemplate,
        jinja: params.jinja,
        responseFormat: params.responseFormat,
        tools: params.tools,
        parallelToolCalls: params.parallelToolCalls,
        toolChoice: params.toolChoice,
      );
      
      if (formattedResult is JinjaFormattedChatResult) {
        nativeParams['prompt'] = formattedResult.prompt;
        if (formattedResult.chatFormat != null) {
          nativeParams['chat_format'] = formattedResult.chatFormat;
        }
        if (formattedResult.grammar != null) {
          nativeParams['grammar'] = formattedResult.grammar;
        }
        if (formattedResult.grammarLazy != null) {
          nativeParams['grammar_lazy'] = formattedResult.grammarLazy;
        }
        if (formattedResult.grammarTriggers != null) {
          nativeParams['grammar_triggers'] = formattedResult.grammarTriggers.map((t) => t.toJson()).toList();
        }
        if (formattedResult.preservedTokens != null) {
          nativeParams['preserved_tokens'] = formattedResult.preservedTokens;
        }
        if (formattedResult.additionalStops != null && formattedResult.additionalStops!.isNotEmpty) {
          final stops = params.stop ?? [];
          nativeParams['stop'] = [...stops, ...formattedResult.additionalStops!];
        }
      } else if (formattedResult is String) {
        nativeParams['prompt'] = formattedResult;
      }
    } else if (params.responseFormat?.type == 'json_object' || params.responseFormat?.type == 'json_schema') {
      final jsonSchema = _getJsonSchema(params.responseFormat);
      if (jsonSchema != null) {
        nativeParams['json_schema'] = jsonEncode(jsonSchema);
      }
    }
    
    // Set up token callback if provided
    StreamSubscription? subscription;
    if (callback != null) {
      subscription = CactusEvents.onToken.listen((event) {
        if (event['contextId'] == id) {
          final tokenData = TokenData(
            token: event['tokenResult']['token'],
            completionProbabilities: event['tokenResult']['completion_probabilities'] != null
                ? List<TokenProb>.from(
                    (event['tokenResult']['completion_probabilities'] as List)
                        .map((prob) => TokenProb.fromJson(prob)))
                : null,
          );
          callback(tokenData);
        }
      });
    }
    
    try {
      final result = await CactusNative.completion(id, nativeParams);
      return result;
    } finally {
      await subscription?.cancel();
    }
  }
  
  /// Stop the current completion
  Future<void> stopCompletion() async {
    return CactusNative.stopCompletion(id);
  }
  
  /// Tokenize text into tokens
  Future<TokenizeResult> tokenize(String text) async {
    return CactusNative.tokenize(id, text);
  }
  
  /// Detokenize tokens into text
  Future<String> detokenize(List<int> tokens) async {
    return CactusNative.detokenize(id, tokens);
  }
  
  /// Get embeddings for text
  Future<EmbeddingResult> embedding(
    String text, {
    EmbeddingParams? params,
  }) async {
    return CactusNative.embedding(
      id,
      text,
      params?.toNative() ?? {},
    );
  }
  
  /// Benchmark the model
  Future<BenchResult> bench({
    required int pp,
    required int tg,
    required int pl,
    required int nr,
  }) async {
    final result = await CactusNative.bench(id, pp, tg, pl, nr);
    return result;
  }
  
  /// Apply LoRA adapters
  Future<void> applyLoraAdapters(List<LoraAdapter> loraList) async {
    return CactusNative.applyLoraAdapters(id, loraList);
  }
  
  /// Remove all LoRA adapters
  Future<void> removeLoraAdapters() async {
    return CactusNative.removeLoraAdapters(id);
  }
  
  /// Get loaded LoRA adapters
  Future<List<LoraAdapter>> getLoadedLoraAdapters() async {
    return CactusNative.getLoadedLoraAdapters(id);
  }
  
  /// Release the context
  Future<void> release() async {
    return CactusNative.releaseContext(id);
  }
  
  /// Get JSON schema from response format
  Map<String, dynamic>? _getJsonSchema(CompletionResponseFormat? responseFormat) {
    if (responseFormat?.type == 'json_schema') {
      return responseFormat?.jsonSchema?.schema;
    }
    if (responseFormat?.type == 'json_object') {
      return responseFormat?.schema ?? {};
    }
    return null;
  }
} 