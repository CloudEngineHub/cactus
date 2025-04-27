import 'dart:convert';
import 'types.dart';

/// Native chat message format for the Cactus backend
class NativeChatMessage {
  final String role;
  final String content;
  
  NativeChatMessage({
    required this.role,
    required this.content,
  });
  
  Map<String, dynamic> toJson() => {
    'role': role,
    'content': content,
  };
}

/// Format chat messages to the native format
List<NativeChatMessage> formatChat(List<ChatMessage> messages) {
  final List<NativeChatMessage> chat = [];
  
  for (final currMsg in messages) {
    final String role = currMsg.role;
    String content = '';
    
    if (currMsg.content is String) {
      content = currMsg.content as String;
    } else if (currMsg.content is List) {
      final parts = currMsg.content as List;
      
      for (int i = 0; i < parts.length; i++) {
        final part = parts[i];
        
        if (part is Map<String, dynamic> && part.containsKey('text')) {
          if (content.isNotEmpty) {
            content += '\n';
          }
          content += part['text'] as String;
        } else if (part is MessagePart && part.text != null) {
          if (content.isNotEmpty) {
            content += '\n';
          }
          content += part.text!;
        }
      }
    } else {
      throw TypeError();
    }
    
    chat.add(NativeChatMessage(role: role, content: content));
  }
  
  return chat;
}

/// Format a list of chat messages to a JSON string
String formatChatToJson(List<ChatMessage> messages) {
  final List<NativeChatMessage> chat = formatChat(messages);
  return jsonEncode(chat.map((m) => m.toJson()).toList());
} 