import 'package:flutter/material.dart';
import 'package:cactus_flutter/cactus_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool _isModelLoaded = false;
  bool _isLoading = false;
  String _loadingStatus = '';
  LlamaContext? _context;
  final TextEditingController _promptController = TextEditingController();
  final List<ChatMessage> _messages = [];
  bool _isGenerating = false;
  String _currentGeneration = '';

  @override
  void dispose() {
    _releaseModel();
    _promptController.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    if (_isLoading) return;
    
    setState(() {
      _isLoading = true;
      _loadingStatus = 'Downloading model...';
    });
    
    try {
      final modelPath = await downloadModelIfNotExists(
        onProgress: (progress) {
          setState(() {
            _loadingStatus = 'Downloading model: $progress%';
          });
        },
      );
      
      setState(() {
        _loadingStatus = 'Loading model...';
      });
      
      final context = await initLlama(
        ContextParams(
          model: modelPath,
          nGpuLayers: 16, // Use GPU layers on supported devices
        ),
        onProgress: (progress) {
          setState(() {
            _loadingStatus = 'Loading model: ${(progress * 100).toInt()}%';
          });
        },
      );
      
      setState(() {
        _context = context;
        _isModelLoaded = true;
        _isLoading = false;
        _loadingStatus = '';
        
        // Initialize with a system message
        _messages.add(
          ChatMessage(
            role: 'system',
            content: 'You are a helpful assistant. Be concise and clear in your responses.',
          ),
        );
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _loadingStatus = 'Error: $e';
      });
    }
  }
  
  Future<void> _releaseModel() async {
    if (_context != null) {
      await _context!.release();
      setState(() {
        _isModelLoaded = false;
        _context = null;
      });
    }
  }
  
  Future<void> _generateResponse() async {
    if (_context == null || _isGenerating || _promptController.text.isEmpty) {
      return;
    }
    
    final userMessage = ChatMessage(
      role: 'user',
      content: _promptController.text,
    );
    
    setState(() {
      _messages.add(userMessage);
      _isGenerating = true;
      _currentGeneration = '';
      _promptController.clear();
    });
    
    try {
      await _context!.completion(
        CompletionParams(
          messages: _messages,
          temperature: 0.7,
          topP: 0.95,
        ),
        callback: (token) {
          setState(() {
            _currentGeneration += token.token;
          });
        },
      );
      
      final assistantMessage = ChatMessage(
        role: 'assistant',
        content: _currentGeneration,
      );
      
      setState(() {
        _messages.add(assistantMessage);
        _isGenerating = false;
        _currentGeneration = '';
      });
    } catch (e) {
      setState(() {
        _isGenerating = false;
        _currentGeneration = 'Error: $e';
      });
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cactus Flutter Demo'),
        actions: [
          if (!_isModelLoaded && !_isLoading)
            IconButton(
              icon: const Icon(Icons.download),
              onPressed: _loadModel,
              tooltip: 'Load Model',
            ),
          if (_isModelLoaded)
            IconButton(
              icon: const Icon(Icons.delete),
              onPressed: _releaseModel,
              tooltip: 'Release Model',
            ),
        ],
      ),
      body: _isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(_loadingStatus),
                ],
              ),
            )
          : _isModelLoaded
              ? Column(
                  children: [
                    Expanded(
                      child: ListView.builder(
                        padding: const EdgeInsets.all(16),
                        itemCount: _messages.length + (_isGenerating ? 1 : 0),
                        itemBuilder: (context, index) {
                          if (index == _messages.length && _isGenerating) {
                            // Show current generation
                            return _buildMessageBubble(
                              'assistant', 
                              _currentGeneration,
                              isTyping: true,
                            );
                          }
                          
                          final message = _messages[index];
                          if (message.role == 'system') return const SizedBox.shrink();
                          
                          return _buildMessageBubble(
                            message.role, 
                            message.content as String,
                          );
                        },
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: _promptController,
                              decoration: const InputDecoration(
                                hintText: 'Enter your message',
                                border: OutlineInputBorder(),
                              ),
                              onSubmitted: (_) => _generateResponse(),
                              enabled: !_isGenerating,
                            ),
                          ),
                          IconButton(
                            icon: const Icon(Icons.send),
                            onPressed: _isGenerating ? null : _generateResponse,
                          ),
                        ],
                      ),
                    ),
                  ],
                )
              : Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text('No model loaded'),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: _loadModel,
                        child: const Text('Load Model'),
                      ),
                    ],
                  ),
                ),
    );
  }
  
  Widget _buildMessageBubble(String role, String content, {bool isTyping = false}) {
    final isUser = role == 'user';
    
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser 
              ? Theme.of(context).colorScheme.primary 
              : Theme.of(context).colorScheme.secondary,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              isUser ? 'You' : 'Assistant',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: isUser 
                    ? Theme.of(context).colorScheme.onPrimary 
                    : Theme.of(context).colorScheme.onSecondary,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              content,
              style: TextStyle(
                color: isUser 
                    ? Theme.of(context).colorScheme.onPrimary 
                    : Theme.of(context).colorScheme.onSecondary,
              ),
            ),
            if (isTyping)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SizedBox(
                      width: 12,
                      height: 12,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Theme.of(context).colorScheme.onSecondary,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Typing...',
                      style: TextStyle(
                        fontSize: 12,
                        color: Theme.of(context).colorScheme.onSecondary,
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
} 