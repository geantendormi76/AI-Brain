// lib/features/assistant_view.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart'; // 1. 导入 riverpod
import 'package:instant_assistant/features/assistant_view_model.dart'; // 2. 导入我们的ViewModel

// 3. 将 StatelessWidget 修改为 ConsumerWidget
class AssistantView extends ConsumerWidget {
  const AssistantView({super.key});

  @override
  // 4. 在 build 方法中添加 WidgetRef ref 参数
  Widget build(BuildContext context, WidgetRef ref) {
    // 5. 监听我们的ViewModel的状态
    final assistantState = ref.watch(assistantViewModelProvider);

    // 决定在响应区显示什么内容
    Widget responseWidget;
    if (assistantState.isLoading) {
      responseWidget = const Center(child: CircularProgressIndicator());
    } else if (assistantState.error != null) {
      responseWidget = SelectableText(
        assistantState.error!,
        style: const TextStyle(fontSize: 16, height: 1.5, color: Colors.red),
      );
    } else {
      responseWidget = SelectableText(
        assistantState.response ?? '',
        style: const TextStyle(fontSize: 16, height: 1.5),
      );
    }

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // 响应显示区
            Expanded(
              child: Container(
                alignment: Alignment.topLeft,
                child: responseWidget, // 使用我们动态创建的widget
              ),
            ),

            const SizedBox(height: 16),

            // 输入框
            TextField(
              autofocus: true,
              style: const TextStyle(fontSize: 16),
              decoration: InputDecoration(
                hintText: '输入您的问题...',
                filled: true,
                fillColor: const Color(0xFF303134),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide.none,
                ),
                // 当加载时，禁用输入框
                enabled: !assistantState.isLoading,
              ),
              onSubmitted: (value) {
                if (value.trim().isNotEmpty) {
                  // 6. 当用户提交时，调用ViewModel的方法
                  ref.read(assistantViewModelProvider.notifier).sendQuery(value);
                }
              },
            ),
          ],
        ),
      ),
    );
  }
}