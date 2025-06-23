# Conversation Management API - Flutter Example

This example demonstrates the new conversation management API that provides optimized performance for multi-turn conversations.

## Key Features

### 🚀 **Performance Optimized**
- **Consistent TTFT**: First token latency stays ~50ms regardless of conversation length
- **KV Cache Optimization**: Automatic cache management for maximum efficiency
- **3-6x Performance Improvement**: For longer conversations compared to traditional approach

### 🎯 **Simple API**
- **Automatic Chat Formatting**: No manual prompt engineering needed
- **Built-in State Management**: Native conversation context handling
- **Dual Mode Support**: Text optimization + multimodal compatibility

### 📊 **Performance Tracking**
- **Real-time Metrics**: TTFT, total time, tokens/sec tracking
- **Conversation Status**: Active/inactive state monitoring
- **Debug Logging**: Detailed performance information

## Usage Examples

### Basic Conversation
```dart
// Initialize the service
final cactusService = CactusService();
await cactusService.initialize();

// Send messages with optimized performance
await cactusService.sendMessage("Hello! How are you?");
await cactusService.sendMessage("What can you help me with?");
await cactusService.sendMessage("Tell me about space");
```

### Simple Response (Non-UI)
```dart
// For programmatic use without UI updates
final response = await cactusService.generateSimpleResponse(
  "What is the meaning of life?",
  maxTokens: 100,
);
print(response);
```

### Performance Monitoring
```dart
// Listen to performance metrics
cactusService.lastConversationResult.addListener(() {
  final result = cactusService.lastConversationResult.value;
  if (result != null) {
    print('TTFT: ${result.timeToFirstToken}ms');
    print('Speed: ${result.tokensPerSecond.toStringAsFixed(1)} tok/s');
    print('Tokens: ${result.tokensGenerated}');
  }
});

// Check conversation status
cactusService.isConversationActive.addListener(() {
  final isActive = cactusService.isConversationActive.value;
  print('Conversation active: $isActive');
});
```

## CactusService Updates

### New ValueNotifiers
```dart
// Performance tracking
final ValueNotifier<ConversationResult?> lastConversationResult;
final ValueNotifier<bool> isConversationActive;
```

### Updated Methods
```dart
// Enhanced sendMessage with dual-mode support
Future<void> sendMessage(String userInput);

// New simple response method
Future<String> generateSimpleResponse(String userInput, {int maxTokens = 200});

// Enhanced clear with native state management
void clearConversation();
```

## Performance Widget

The included `PerformanceWidget` displays real-time metrics:

```dart
PerformanceWidget(
  conversationResult: cactusService.lastConversationResult,
  isConversationActive: cactusService.isConversationActive,
)
```

Shows:
- **TTFT** - Time to first token (green)
- **Total Time** - Complete generation time (blue) 
- **Tokens** - Number of tokens generated (orange)
- **Speed** - Tokens per second (purple)
- **Status** - Conversation active indicator

## Mode Switching

The service automatically handles switching between text and multimodal modes:

### Text Mode (Optimized)
- Uses `continueConversation()` API
- Automatic KV cache optimization
- Consistent performance regardless of conversation length
- Built-in performance tracking

### Multimodal Mode (Compatible)
- Uses traditional `multimodalCompletion()` API
- Manual conversation history management
- Supports images and vision tasks
- Falls back for complex multimodal scenarios

### Automatic Switching
```dart
// Text conversation - uses optimized API
await cactusService.sendMessage("Hello!");
await cactusService.sendMessage("How are you?");

// Add image - automatically switches to multimodal mode
cactusService.stageImageFromAsset('assets/image.jpg', 'temp.jpg');
await cactusService.sendMessage("What do you see?");

// Back to text - automatically switches back to optimized mode
await cactusService.sendMessage("Thank you!");
```

## Performance Comparison

### Traditional Approach
```
Turn 1: TTFT ~50ms   (50 tokens processed)
Turn 2: TTFT ~150ms  (150 tokens processed) 
Turn 3: TTFT ~300ms  (300 tokens processed)
Turn 4: TTFT ~500ms  (500 tokens processed)
```

### Conversation Management API
```
Turn 1: TTFT ~50ms   (cache building)
Turn 2: TTFT ~50ms   (cache optimized)
Turn 3: TTFT ~50ms   (cache optimized)
Turn 4: TTFT ~50ms   (cache optimized)
```

## Debug Output

Enable detailed logging to see performance metrics:

```
[PERFORMANCE] TTFT: 45ms, Total: 892ms, Tokens: 23, Speed: 25.8 tok/s
[TIMING] Flutter overhead: 12ms
[STATUS] Conversation active: true
```

## Best Practices

### 1. **Use Text Mode for Conversations**
For optimal performance, use text-only conversations when possible.

### 2. **Monitor Performance**
Use the performance widgets to track TTFT consistency.

### 3. **Clear When Needed**
Call `clearConversation()` when starting fresh topics.

### 4. **Handle Mode Switches**
The service automatically manages mode transitions.

## Integration Example

```dart
class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final CactusService _cactusService = CactusService();
  
  @override
  void initState() {
    super.initState();
    _cactusService.initialize();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Optimized Chat')),
      body: Column(
        children: [
          // Performance metrics
          PerformanceWidget(
            conversationResult: _cactusService.lastConversationResult,
            isConversationActive: _cactusService.isConversationActive,
          ),
          
          // Info about benefits
          PerformanceInfoWidget(),
          
          // Chat messages
          Expanded(
            child: ValueListenableBuilder<List<ChatMessage>>(
              valueListenable: _cactusService.chatMessages,
              builder: (context, messages, child) {
                return ListView.builder(
                  itemCount: messages.length,
                  itemBuilder: (context, index) {
                    final message = messages[index];
                    return ChatBubble(message: message);
                  },
                );
              },
            ),
          ),
          
          // Input field
          MessageInputField(
            onSend: (text) => _cactusService.sendMessage(text),
            onClear: () => _cactusService.clearConversation(),
          ),
        ],
      ),
    );
  }
  
  @override
  void dispose() {
    _cactusService.dispose();
    super.dispose();
  }
}
```

This integration provides a complete chat experience with optimal performance, real-time metrics, and automatic mode management. 