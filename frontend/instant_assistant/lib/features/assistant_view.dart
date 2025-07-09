// lib/features/assistant_view.dart (修正版)
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:instant_assistant/features/assistant_view_model.dart';
import 'package:instant_assistant/core/models/chat_message.dart';

class AssistantView extends ConsumerWidget {
  const AssistantView({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final assistantState = ref.watch(assistantViewModelProvider);
    final messages = assistantState.messages;
    final TextEditingController textController =
        TextEditingController(); // 创建控制器

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // 对话历史展示区
            Expanded(
              child: ListView.builder(
                itemCount: messages.length + (assistantState.isLoading ? 1 : 0),
                itemBuilder: (context, index) {
                  if (assistantState.isLoading && index == messages.length) {
                    // 【修正点】: 移除了这里的 'const' 关键字
                    return _MessageBubble(
                      message: ChatMessage(text: '思考中...', isUser: false),
                    );
                  }
                  final message = messages[index];
                  return _MessageBubble(message: message);
                },
              ),
            ),
            const SizedBox(height: 16),
            // 输入框
            TextField(
              controller: textController, // 使用控制器
              autofocus: true,
              style: const TextStyle(fontSize: 16),
              decoration: InputDecoration(
                hintText: '输入您的问题或要记录的内容...',
                filled: true,
                fillColor: const Color(0xFF303134),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide.none,
                ),
                enabled: !assistantState.isLoading,
              ),
              onSubmitted: (value) {
                if (value.trim().isNotEmpty) {
                  ref
                      .read(assistantViewModelProvider.notifier)
                      .sendQuery(value);
                  textController.clear(); // 发送后清空输入框
                }
              },
            ),
          ],
        ),
      ),
    );
  }
}

// 消息气泡 Widget
class _MessageBubble extends StatelessWidget {
  final ChatMessage message;

  // 【最佳实践】: 为这个Widget也添加一个 const 构造函数
  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
        margin: const EdgeInsets.symmetric(vertical: 4),
        decoration: BoxDecoration(
          color: message.isUser
              ? Theme.of(context).primaryColor
              : const Color(0xFF3C4043),
          borderRadius: BorderRadius.circular(12),
        ),
        child: SelectableText(
          message.text,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 16,
            height: 1.5,
          ),
        ),
      ),
    );
  }
}
