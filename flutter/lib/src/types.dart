import 'package:json_annotation/json_annotation.dart';

part 'types.g.dart';

/// Context parameters for initializing the model
@JsonSerializable()
class ContextParams {
  /// Path to the model file
  final String model;
  
  /// Chat template to override the default one from the model
  final String? chatTemplate;
  
  /// Whether the model is an asset in the app bundle
  final bool? isModelAsset;
  
  /// Use progress callback during initialization
  final bool? useProgressCallback;
  
  /// Context size for the model
  final int? nCtx;
  
  /// Batch size for processing
  final int? nBatch;
  
  /// Micro batch size
  final int? nUbatch;
  
  /// Number of threads to use for computation
  final int? nThreads;
  
  /// Number of layers to store in VRAM (Currently only for iOS)
  final int? nGpuLayers;
  
  /// Skip GPU devices (iOS only)
  final bool? noGpuDevices;
  
  /// Enable flash attention, only recommended in GPU device
  final bool? flashAttn;
  
  /// KV cache data type for K
  final String? cacheTypeK;
  
  /// KV cache data type for V
  final String? cacheTypeV;
  
  /// Use mlock to keep model in memory
  final bool? useMlock;
  
  /// Use memory mapping for model loading
  final bool? useMmap;
  
  /// Only load vocabulary
  final bool? vocabOnly;
  
  /// Single LoRA adapter path
  final String? lora;
  
  /// Single LoRA adapter scale
  final double? loraScaled;
  
  /// List of LoRA adapters
  final List<LoraAdapter>? loraList;
  
  /// RoPE base frequency
  final double? ropeFreqBase;
  
  /// RoPE frequency scaling
  final double? ropeFreqScale;
  
  /// Pooling type
  final String? poolingType;
  
  /// Enable embedding mode
  final bool? embedding;
  
  /// Normalize embeddings
  final double? embdNormalize;

  ContextParams({
    required this.model,
    this.chatTemplate,
    this.isModelAsset,
    this.useProgressCallback,
    this.nCtx,
    this.nBatch,
    this.nUbatch,
    this.nThreads,
    this.nGpuLayers,
    this.noGpuDevices,
    this.flashAttn,
    this.cacheTypeK,
    this.cacheTypeV,
    this.useMlock,
    this.useMmap,
    this.vocabOnly,
    this.lora,
    this.loraScaled,
    this.loraList,
    this.ropeFreqBase,
    this.ropeFreqScale,
    this.poolingType,
    this.embedding,
    this.embdNormalize,
  });

  factory ContextParams.fromJson(Map<String, dynamic> json) =>
      _$ContextParamsFromJson(json);
  
  Map<String, dynamic> toJson() => _$ContextParamsToJson(this);
  
  /// Convert to native format
  Map<String, dynamic> toNative() {
    final Map<String, dynamic> native = {
      'model': model,
    };
    
    if (chatTemplate != null) native['chat_template'] = chatTemplate;
    if (isModelAsset != null) native['is_model_asset'] = isModelAsset;
    if (useProgressCallback != null) native['use_progress_callback'] = useProgressCallback;
    if (nCtx != null) native['n_ctx'] = nCtx;
    if (nBatch != null) native['n_batch'] = nBatch;
    if (nUbatch != null) native['n_ubatch'] = nUbatch;
    if (nThreads != null) native['n_threads'] = nThreads;
    if (nGpuLayers != null) native['n_gpu_layers'] = nGpuLayers;
    if (noGpuDevices != null) native['no_gpu_devices'] = noGpuDevices;
    if (flashAttn != null) native['flash_attn'] = flashAttn;
    if (cacheTypeK != null) native['cache_type_k'] = cacheTypeK;
    if (cacheTypeV != null) native['cache_type_v'] = cacheTypeV;
    if (useMlock != null) native['use_mlock'] = useMlock;
    if (useMmap != null) native['use_mmap'] = useMmap;
    if (vocabOnly != null) native['vocab_only'] = vocabOnly;
    if (lora != null) native['lora'] = lora;
    if (loraScaled != null) native['lora_scaled'] = loraScaled;
    if (loraList != null) native['lora_list'] = loraList!.map((e) => e.toJson()).toList();
    if (ropeFreqBase != null) native['rope_freq_base'] = ropeFreqBase;
    if (ropeFreqScale != null) native['rope_freq_scale'] = ropeFreqScale;
    if (poolingType != null) {
      native['pooling_type'] = _poolingTypeToInt(poolingType!);
    }
    if (embedding != null) native['embedding'] = embedding;
    if (embdNormalize != null) native['embd_normalize'] = embdNormalize;
    
    return native;
  }
  
  int _poolingTypeToInt(String poolingType) {
    switch (poolingType) {
      case 'none': return 0;
      case 'mean': return 1;
      case 'cls': return 2;
      case 'last': return 3;
      case 'rank': return 4;
      default: return 0;
    }
  }
}

/// LoRA adapter configuration
@JsonSerializable()
class LoraAdapter {
  /// Path to the LoRA adapter file
  final String path;
  
  /// Scaling factor for the adapter
  final double? scaled;

  LoraAdapter({required this.path, this.scaled});

  factory LoraAdapter.fromJson(Map<String, dynamic> json) =>
      _$LoraAdapterFromJson(json);
  
  Map<String, dynamic> toJson() => _$LoraAdapterToJson(this);
}

/// Message part for chat messages
@JsonSerializable()
class MessagePart {
  final String? text;
  
  MessagePart({this.text});
  
  factory MessagePart.fromJson(Map<String, dynamic> json) =>
      _$MessagePartFromJson(json);
  
  Map<String, dynamic> toJson() => _$MessagePartToJson(this);
}

/// OpenAI-compatible chat message
@JsonSerializable()
class ChatMessage {
  final String role;
  final dynamic content;
  
  ChatMessage({required this.role, required this.content});
  
  factory ChatMessage.fromJson(Map<String, dynamic> json) =>
      _$ChatMessageFromJson(json);
  
  Map<String, dynamic> toJson() => _$ChatMessageToJson(this);
}

/// Completion response format
@JsonSerializable()
class CompletionResponseFormat {
  final String type; // 'text' | 'json_object' | 'json_schema'
  final JsonSchema? jsonSchema;
  final Map<String, dynamic>? schema; // for json_object type
  
  CompletionResponseFormat({
    required this.type,
    this.jsonSchema,
    this.schema,
  });
  
  factory CompletionResponseFormat.fromJson(Map<String, dynamic> json) =>
      _$CompletionResponseFormatFromJson(json);
  
  Map<String, dynamic> toJson() => _$CompletionResponseFormatToJson(this);
}

/// JSON schema configuration
@JsonSerializable()
class JsonSchema {
  final bool? strict;
  final Map<String, dynamic> schema;
  
  JsonSchema({required this.schema, this.strict});
  
  factory JsonSchema.fromJson(Map<String, dynamic> json) =>
      _$JsonSchemaFromJson(json);
  
  Map<String, dynamic> toJson() => _$JsonSchemaToJson(this);
}

/// Base parameters for completion requests
@JsonSerializable()
class CompletionBaseParams {
  final String? prompt;
  final List<ChatMessage>? messages;
  final String? chatTemplate;
  final bool? jinja;
  final Map<String, dynamic>? tools;
  final Map<String, dynamic>? parallelToolCalls;
  final String? toolChoice;
  final CompletionResponseFormat? responseFormat;
  
  CompletionBaseParams({
    this.prompt,
    this.messages,
    this.chatTemplate,
    this.jinja,
    this.tools,
    this.parallelToolCalls,
    this.toolChoice,
    this.responseFormat,
  });
  
  factory CompletionBaseParams.fromJson(Map<String, dynamic> json) =>
      _$CompletionBaseParamsFromJson(json);
  
  Map<String, dynamic> toJson() => _$CompletionBaseParamsToJson(this);
}

/// Full completion parameters
@JsonSerializable()
class CompletionParams extends CompletionBaseParams {
  final List<String>? stop;
  final int? nPredict;
  final int? nProbs;
  final int? topK;
  final double? topP;
  final double? minP;
  final double? xtcProbability;
  final double? xtcThreshold;
  final double? typicalP;
  final double? temperature;
  final int? penaltyLastN;
  final double? penaltyRepeat;
  final double? penaltyFreq;
  final double? penaltyPresent;
  final int? mirostat;
  final double? mirostatTau;
  final double? mirostatEta;
  final double? dryMultiplier;
  final double? dryBase;
  final int? dryAllowedLength;
  final int? dryPenaltyLastN;
  final List<String>? drySequenceBreakers;
  final double? topNSigma;
  final bool? ignoreEos;
  final List<List<dynamic>>? logitBias;
  final int? seed;
  
  CompletionParams({
    super.prompt,
    super.messages,
    super.chatTemplate,
    super.jinja,
    super.tools,
    super.parallelToolCalls,
    super.toolChoice,
    super.responseFormat,
    this.stop,
    this.nPredict,
    this.nProbs,
    this.topK,
    this.topP,
    this.minP,
    this.xtcProbability,
    this.xtcThreshold,
    this.typicalP,
    this.temperature,
    this.penaltyLastN,
    this.penaltyRepeat,
    this.penaltyFreq,
    this.penaltyPresent,
    this.mirostat,
    this.mirostatTau,
    this.mirostatEta,
    this.dryMultiplier,
    this.dryBase,
    this.dryAllowedLength,
    this.dryPenaltyLastN,
    this.drySequenceBreakers,
    this.topNSigma,
    this.ignoreEos,
    this.logitBias,
    this.seed,
  });
  
  factory CompletionParams.fromJson(Map<String, dynamic> json) =>
      _$CompletionParamsFromJson(json);
  
  @override
  Map<String, dynamic> toJson() => _$CompletionParamsToJson(this);
  
  /// Convert to native format
  Map<String, dynamic> toNative({bool emitPartialCompletion = false}) {
    final Map<String, dynamic> native = {
      'prompt': prompt ?? '',
      'emit_partial_completion': emitPartialCompletion,
    };
    
    if (stop != null) native['stop'] = stop;
    if (nPredict != null) native['n_predict'] = nPredict;
    if (nProbs != null) native['n_probs'] = nProbs;
    if (topK != null) native['top_k'] = topK;
    if (topP != null) native['top_p'] = topP;
    if (minP != null) native['min_p'] = minP;
    if (xtcProbability != null) native['xtc_probability'] = xtcProbability;
    if (xtcThreshold != null) native['xtc_threshold'] = xtcThreshold;
    if (typicalP != null) native['typical_p'] = typicalP;
    if (temperature != null) native['temperature'] = temperature;
    if (penaltyLastN != null) native['penalty_last_n'] = penaltyLastN;
    if (penaltyRepeat != null) native['penalty_repeat'] = penaltyRepeat;
    if (penaltyFreq != null) native['penalty_freq'] = penaltyFreq;
    if (penaltyPresent != null) native['penalty_present'] = penaltyPresent;
    if (mirostat != null) native['mirostat'] = mirostat;
    if (mirostatTau != null) native['mirostat_tau'] = mirostatTau;
    if (mirostatEta != null) native['mirostat_eta'] = mirostatEta;
    if (dryMultiplier != null) native['dry_multiplier'] = dryMultiplier;
    if (dryBase != null) native['dry_base'] = dryBase;
    if (dryAllowedLength != null) native['dry_allowed_length'] = dryAllowedLength;
    if (dryPenaltyLastN != null) native['dry_penalty_last_n'] = dryPenaltyLastN;
    if (drySequenceBreakers != null) native['dry_sequence_breakers'] = drySequenceBreakers;
    if (topNSigma != null) native['top_n_sigma'] = topNSigma;
    if (ignoreEos != null) native['ignore_eos'] = ignoreEos;
    if (logitBias != null) native['logit_bias'] = logitBias;
    if (seed != null) native['seed'] = seed;
    
    return native;
  }
}

/// Token probability item
@JsonSerializable()
class TokenProbItem {
  final String tokStr;
  final double prob;
  
  TokenProbItem({required this.tokStr, required this.prob});
  
  factory TokenProbItem.fromJson(Map<String, dynamic> json) =>
      _$TokenProbItemFromJson(json);
  
  Map<String, dynamic> toJson() => _$TokenProbItemToJson(this);
}

/// Token probability
@JsonSerializable()
class TokenProb {
  final String content;
  final List<TokenProbItem> probs;
  
  TokenProb({required this.content, required this.probs});
  
  factory TokenProb.fromJson(Map<String, dynamic> json) =>
      _$TokenProbFromJson(json);
  
  Map<String, dynamic> toJson() => _$TokenProbToJson(this);
}

/// Token data
@JsonSerializable()
class TokenData {
  final String token;
  final List<TokenProb>? completionProbabilities;
  
  TokenData({
    required this.token,
    this.completionProbabilities,
  });
  
  factory TokenData.fromJson(Map<String, dynamic> json) =>
      _$TokenDataFromJson(json);
  
  Map<String, dynamic> toJson() => _$TokenDataToJson(this);
}

/// Completion result timings
@JsonSerializable()
class CompletionResultTimings {
  final int promptN;
  final int promptMs;
  final double promptPerTokenMs;
  final double promptPerSecond;
  final int predictedN;
  final int predictedMs;
  final double predictedPerTokenMs;
  final double predictedPerSecond;
  
  CompletionResultTimings({
    required this.promptN,
    required this.promptMs,
    required this.promptPerTokenMs,
    required this.promptPerSecond,
    required this.predictedN,
    required this.predictedMs,
    required this.predictedPerTokenMs,
    required this.predictedPerSecond,
  });
  
  factory CompletionResultTimings.fromJson(Map<String, dynamic> json) =>
      _$CompletionResultTimingsFromJson(json);
  
  Map<String, dynamic> toJson() => _$CompletionResultTimingsToJson(this);
}

/// Function call
@JsonSerializable()
class FunctionCall {
  final String type; // Usually 'function'
  final FunctionCallDetails function;
  final String? id;
  
  FunctionCall({
    required this.type,
    required this.function,
    this.id,
  });
  
  factory FunctionCall.fromJson(Map<String, dynamic> json) =>
      _$FunctionCallFromJson(json);
  
  Map<String, dynamic> toJson() => _$FunctionCallToJson(this);
}

/// Function call details
@JsonSerializable()
class FunctionCallDetails {
  final String name;
  final String arguments;
  
  FunctionCallDetails({
    required this.name,
    required this.arguments,
  });
  
  factory FunctionCallDetails.fromJson(Map<String, dynamic> json) =>
      _$FunctionCallDetailsFromJson(json);
  
  Map<String, dynamic> toJson() => _$FunctionCallDetailsToJson(this);
}

/// Completion result
@JsonSerializable()
class CompletionResult {
  final String text;
  final String reasoningContent;
  final List<FunctionCall> toolCalls;
  final String content;
  final int tokensPredicted;
  final int tokensEvaluated;
  final bool truncated;
  final bool stoppedEos;
  final String stoppedWord;
  final int stoppedLimit;
  final String stoppingWord;
  final int tokensCached;
  final CompletionResultTimings timings;
  final List<TokenProb>? completionProbabilities;
  
  CompletionResult({
    required this.text,
    required this.reasoningContent,
    required this.toolCalls,
    required this.content,
    required this.tokensPredicted,
    required this.tokensEvaluated,
    required this.truncated,
    required this.stoppedEos,
    required this.stoppedWord,
    required this.stoppedLimit,
    required this.stoppingWord,
    required this.tokensCached,
    required this.timings,
    this.completionProbabilities,
  });
  
  factory CompletionResult.fromJson(Map<String, dynamic> json) =>
      _$CompletionResultFromJson(json);
  
  Map<String, dynamic> toJson() => _$CompletionResultToJson(this);
}

/// Tokenize result
@JsonSerializable()
class TokenizeResult {
  final List<int> tokens;
  
  TokenizeResult({required this.tokens});
  
  factory TokenizeResult.fromJson(Map<String, dynamic> json) =>
      _$TokenizeResultFromJson(json);
  
  Map<String, dynamic> toJson() => _$TokenizeResultToJson(this);
}

/// Embedding result
@JsonSerializable()
class EmbeddingResult {
  final List<double> embedding;
  
  EmbeddingResult({required this.embedding});
  
  factory EmbeddingResult.fromJson(Map<String, dynamic> json) =>
      _$EmbeddingResultFromJson(json);
  
  Map<String, dynamic> toJson() => _$EmbeddingResultToJson(this);
}

/// Embedding parameters
@JsonSerializable()
class EmbeddingParams {
  final double? embdNormalize;
  
  EmbeddingParams({this.embdNormalize});
  
  factory EmbeddingParams.fromJson(Map<String, dynamic> json) =>
      _$EmbeddingParamsFromJson(json);
  
  Map<String, dynamic> toJson() => _$EmbeddingParamsToJson(this);
  
  /// Convert to native format
  Map<String, dynamic> toNative() {
    final Map<String, dynamic> native = {};
    if (embdNormalize != null) native['embd_normalize'] = embdNormalize;
    return native;
  }
}

/// Chat template capabilities
@JsonSerializable()
class ChatTemplateCaps {
  final bool tools;
  final bool toolCalls;
  final bool toolResponses;
  final bool systemRole;
  final bool parallelToolCalls;
  final bool toolCallId;
  
  ChatTemplateCaps({
    required this.tools,
    required this.toolCalls,
    required this.toolResponses,
    required this.systemRole,
    required this.parallelToolCalls,
    required this.toolCallId,
  });
  
  factory ChatTemplateCaps.fromJson(Map<String, dynamic> json) =>
      _$ChatTemplateCapsFromJson(json);
  
  Map<String, dynamic> toJson() => _$ChatTemplateCapsToJson(this);
}

/// Chat templates
@JsonSerializable()
class ChatTemplates {
  final bool llamaChat;
  final MinjaTemplates? minja;
  
  ChatTemplates({
    required this.llamaChat,
    this.minja,
  });
  
  factory ChatTemplates.fromJson(Map<String, dynamic> json) =>
      _$ChatTemplatesFromJson(json);
  
  Map<String, dynamic> toJson() => _$ChatTemplatesToJson(this);
}

/// Minja templates
@JsonSerializable()
class MinjaTemplates {
  final bool? default_;
  final ChatTemplateCaps? defaultCaps;
  final bool? toolUse;
  final ChatTemplateCaps? toolUseCaps;
  
  MinjaTemplates({
    this.default_,
    this.defaultCaps,
    this.toolUse,
    this.toolUseCaps,
  });
  
  factory MinjaTemplates.fromJson(Map<String, dynamic> json) =>
      _$MinjaTemplatesFromJson(json);
  
  Map<String, dynamic> toJson() => _$MinjaTemplatesToJson(this);
}

/// Model information
@JsonSerializable()
class ModelInfo {
  final String desc;
  final int size;
  final int nEmbd;
  final int nParams;
  final ChatTemplates chatTemplates;
  final Map<String, dynamic> metadata;
  final bool isChatTemplateSupported;
  
  ModelInfo({
    required this.desc,
    required this.size,
    required this.nEmbd,
    required this.nParams,
    required this.chatTemplates,
    required this.metadata,
    required this.isChatTemplateSupported,
  });
  
  factory ModelInfo.fromJson(Map<String, dynamic> json) =>
      _$ModelInfoFromJson(json);
  
  Map<String, dynamic> toJson() => _$ModelInfoToJson(this);
}

/// Grammar trigger
@JsonSerializable()
class GrammarTrigger {
  final int type;
  final String value;
  final int token;
  
  GrammarTrigger({
    required this.type,
    required this.value,
    required this.token,
  });
  
  factory GrammarTrigger.fromJson(Map<String, dynamic> json) =>
      _$GrammarTriggerFromJson(json);
  
  Map<String, dynamic> toJson() => _$GrammarTriggerToJson(this);
}

/// Jinja formatted chat result
@JsonSerializable()
class JinjaFormattedChatResult {
  final String prompt;
  final int? chatFormat;
  final String? grammar;
  final bool? grammarLazy;
  final List<GrammarTrigger>? grammarTriggers;
  final List<String>? preservedTokens;
  final List<String>? additionalStops;
  
  JinjaFormattedChatResult({
    required this.prompt,
    this.chatFormat,
    this.grammar,
    this.grammarLazy,
    this.grammarTriggers,
    this.preservedTokens,
    this.additionalStops,
  });
  
  factory JinjaFormattedChatResult.fromJson(Map<String, dynamic> json) =>
      _$JinjaFormattedChatResultFromJson(json);
  
  Map<String, dynamic> toJson() => _$JinjaFormattedChatResultToJson(this);
}

/// Session load result
@JsonSerializable()
class SessionLoadResult {
  final int tokensLoaded;
  final String prompt;
  
  SessionLoadResult({
    required this.tokensLoaded,
    required this.prompt,
  });
  
  factory SessionLoadResult.fromJson(Map<String, dynamic> json) =>
      _$SessionLoadResultFromJson(json);
  
  Map<String, dynamic> toJson() => _$SessionLoadResultToJson(this);
}

/// Benchmark result
@JsonSerializable()
class BenchResult {
  final String modelDesc;
  final int modelSize;
  final int modelNParams;
  final double ppAvg;
  final double ppStd;
  final double tgAvg;
  final double tgStd;
  
  BenchResult({
    required this.modelDesc,
    required this.modelSize,
    required this.modelNParams,
    required this.ppAvg,
    required this.ppStd,
    required this.tgAvg,
    required this.tgStd,
  });
  
  factory BenchResult.fromJson(Map<String, dynamic> json) =>
      _$BenchResultFromJson(json);
  
  Map<String, dynamic> toJson() => _$BenchResultToJson(this);
} 