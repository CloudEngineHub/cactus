import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart'; // For ChatMessage

class MessageBubble extends StatelessWidget {
  final ChatMessage message;

  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    bool isUser = message.role == 'user';
    bool isSystem = message.role == 'system';

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4.0),
        padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 8.0),
        decoration: BoxDecoration(
          color: isSystem
              ? Colors.red[100]
              : (isUser ? Colors.blue[100] : Colors.green[100]),
          borderRadius: BorderRadius.circular(12.0),
        ),
        child: Text(
          message.content,
          style: TextStyle(color: isSystem ? Colors.red[900] : Colors.black),
        ),
      ),
    );
  }
} 