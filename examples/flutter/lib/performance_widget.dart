import 'package:flutter/material.dart';
import 'package:cactus/cactus.dart';

class PerformanceWidget extends StatelessWidget {
  final ValueNotifier<ConversationResult?> conversationResult;
  final ValueNotifier<bool> isConversationActive;

  const PerformanceWidget({
    Key? key,
    required this.conversationResult,
    required this.isConversationActive,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(8.0),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.speed, size: 16, color: Colors.blue),
                const SizedBox(width: 4),
                Text(
                  'Performance Metrics',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            ValueListenableBuilder<ConversationResult?>(
              valueListenable: conversationResult,
              builder: (context, result, child) {
                if (result == null) {
                  return const Text(
                    'No conversation metrics available\n(Use text mode for optimized performance)',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey,
                      fontStyle: FontStyle.italic,
                    ),
                  );
                }

                return Column(
                  children: [
                    _buildMetricRow('TTFT', '${result.timeToFirstToken}ms', Colors.green),
                    _buildMetricRow('Total Time', '${result.totalTime}ms', Colors.blue),
                    _buildMetricRow('Tokens', '${result.tokensGenerated}', Colors.orange),
                    _buildMetricRow('Speed', '${result.tokensPerSecond.toStringAsFixed(1)} tok/s', Colors.purple),
                  ],
                );
              },
            ),
            const SizedBox(height: 8),
            ValueListenableBuilder<bool>(
              valueListenable: isConversationActive,
              builder: (context, isActive, child) {
                return Row(
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isActive ? Colors.green : Colors.grey,
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      'Conversation ${isActive ? 'Active' : 'Inactive'}',
                      style: const TextStyle(fontSize: 12),
                    ),
                  ],
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricRow(String label, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: Text(
              value,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class PerformanceInfoWidget extends StatelessWidget {
  const PerformanceInfoWidget({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(8.0),
      color: Colors.blue.shade50,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.info_outline, size: 16, color: Colors.blue),
                const SizedBox(width: 4),
                Text(
                  'Conversation Management Benefits',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Colors.blue.shade700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            const Text(
              '• Consistent TTFT regardless of conversation length\n'
              '• Automatic KV cache optimization\n'
              '• No manual conversation history management\n'
              '• Built-in performance tracking\n'
              '• Stateful conversation context',
              style: TextStyle(fontSize: 12, height: 1.4),
            ),
          ],
        ),
      ),
    );
  }
} 