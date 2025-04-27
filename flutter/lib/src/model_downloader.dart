import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:dio/dio.dart';
import 'package:path/path.dart' as p;

/// Model downloader utility for downloading and managing LLM models
class ModelDownloader {
  final String modelUrl;
  final String modelFolderName;
  late String modelName;
  late String fullModelPath;
  late String fullModelFolderPath;
  
  final List<String> supportedProviders = ['huggingface.co'];
  final List<String> supportedFormats = ['gguf'];
  
  static const String defaultModelFolderName = "models";
  static const String defaultModelUrl = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf";
  
  /// Create a new ModelDownloader instance
  /// 
  /// [modelUrl] is the URL to download the model from
  /// [modelFolderName] is the name of the folder to store models in
  ModelDownloader({
    String? modelUrl,
    String? modelFolderName,
  }) : 
    modelUrl = modelUrl ?? defaultModelUrl,
    modelFolderName = modelFolderName ?? defaultModelFolderName {
    validateUrlFormat();
    parseModelName();
  }
  
  /// Validate the model URL format
  void validateUrlFormat() {
    bool isValid = true;
    if (!modelUrl.startsWith('http://') && !modelUrl.startsWith('https://')) {
      print('Invalid model URL: $modelUrl');
      isValid = false;
    }
    
    final Uri uri = Uri.parse(modelUrl);
    final String? host = uri.host;
    if (host == null || !supportedProviders.contains(host)) {
      print('Invalid model provider: $host');
      isValid = false;
    }
    
    final String extension = p.extension(modelUrl).toLowerCase().replaceAll('.', '');
    if (!supportedFormats.contains(extension)) {
      print('Invalid model format: $extension');
      isValid = false;
    }
    
    if (!isValid) {
      throw Exception("Invalid model URL");
    }
    
    print('Valid URL format: $modelUrl');
  }
  
  /// Parse the model name from the URL
  Future<void> parseModelName() async {
    final Uri uri = Uri.parse(modelUrl);
    final String path = uri.path;
    modelName = p.basename(path);
    
    final appDocDir = await getApplicationDocumentsDirectory();
    fullModelFolderPath = '${appDocDir.path}/$modelFolderName';
    fullModelPath = '$fullModelFolderPath/$modelName';
    
    print('Model name: $modelName');
    print('Model folder path: $fullModelFolderPath');
    print('Model path: $fullModelPath');
  }
  
  /// Create the model folder if it doesn't exist
  Future<void> createModelFolder() async {
    final dir = Directory(fullModelFolderPath);
    if (!await dir.exists()) {
      await dir.create(recursive: true);
      print('Model folder created: $fullModelFolderPath');
    } else {
      print('Model folder already exists: $fullModelFolderPath');
    }
  }
  
  /// Download the model if it doesn't exist
  /// 
  /// [onProgress] is called with the download progress (0-100)
  /// [onSuccess] is called when the download completes successfully
  Future<String> downloadModelIfNotExists({
    Function(int progress)? onProgress,
    Function(String modelPath)? onSuccess,
  }) async {
    final file = File(fullModelPath);
    
    if (!await file.exists()) {
      await createModelFolder();
      print('Downloading model from $modelUrl');
      
      final dio = Dio();
      
      try {
        await dio.download(
          modelUrl,
          fullModelPath,
          onReceiveProgress: (received, total) {
            if (total != -1) {
              final percentage = ((received / total) * 100).floor();
              onProgress?.call(percentage);
            }
          },
        );
        
        print('Model downloaded successfully to $fullModelPath');
        onSuccess?.call(fullModelPath);
      } catch (e) {
        print('Error downloading model: $e');
        throw Exception('Error downloading model: $e');
      }
    } else {
      print('Model already exists at $fullModelPath');
      onSuccess?.call(fullModelPath);
    }
    
    return fullModelPath;
  }
}

/// Download a model if it doesn't exist
/// 
/// [modelUrl] is the URL to download the model from
/// [modelFolderName] is the name of the folder to store models in
/// [onProgress] is called with the download progress (0-100)
/// [onSuccess] is called when the download completes successfully
Future<String> downloadModelIfNotExists({
  String? modelUrl,
  String? modelFolderName,
  Function(int progress)? onProgress,
  Function(String modelPath)? onSuccess,
}) async {
  final downloader = ModelDownloader(
    modelUrl: modelUrl,
    modelFolderName: modelFolderName,
  );
  
  return downloader.downloadModelIfNotExists(
    onProgress: onProgress,
    onSuccess: onSuccess,
  );
} 