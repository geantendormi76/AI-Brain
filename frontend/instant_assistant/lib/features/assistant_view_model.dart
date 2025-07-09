// lib/features/assistant_view_model.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:instant_assistant/core/services/backend_service.dart';
import 'package:instant_assistant/core/models/chat_message.dart';

class AssistantState {
  final bool isLoading;
  // 将 response 替换为 messages 列表
  final List<ChatMessage> messages;
  final String? error;

  // 构造函数也需要更新
  AssistantState({
    this.isLoading = false,
    this.messages = const [], // 默认为空列表
    this.error,
  });

  // copyWith 方法同样更新
  AssistantState copyWith({
    bool? isLoading,
    List<ChatMessage>? messages,
    String? error,
  }) {
    return AssistantState(
      isLoading: isLoading ?? this.isLoading,
      messages: messages ?? this.messages,
      error: error, // 允许将 error 清空
    );
  }
}

// 2. 创建一个BackendService的Provider
// 这样我们就可以在ViewModel中访问它
final backendServiceProvider = Provider<BackendService>(
  (ref) => BackendService(),
);

class AssistantViewModel extends StateNotifier<AssistantState> {
  final BackendService _backendService;

  // 初始状态包含一条欢迎消息
  AssistantViewModel(this._backendService)
    : super(
        AssistantState(
          messages: [ChatMessage(text: '你好，有什么可以帮您？', isUser: false)],
        ),
      );

  Future<void> sendQuery(String query) async {
    // 1. 立刻将用户消息添加到列表中并更新UI
    final userMessage = ChatMessage(text: query, isUser: true);
    state = state.copyWith(
      isLoading: true,
      messages: [...state.messages, userMessage],
      error: null, // 清除之前的错误
    );

    try {
      // 2. 调用后端服务
      final backendResponse = await _backendService.sendQuery(query);

      // 3. 将AI的响应添加到列表中
      final aiMessage = ChatMessage(
        text: backendResponse.response,
        isUser: false,
      );
      state = state.copyWith(
        isLoading: false,
        messages: [...state.messages, aiMessage],
      );
    } catch (e) {
      // 4. 如果出错，也将错误信息作为一条AI消息添加
      final errorMessage = ChatMessage(
        text: '错误: ${e.toString()}',
        isUser: false,
      );
      state = state.copyWith(
        isLoading: false,
        messages: [...state.messages, errorMessage],
      );
    }
  }
}

// ViewModel的Provider保持不变
final assistantViewModelProvider =
    StateNotifierProvider<AssistantViewModel, AssistantState>((ref) {
      final backendService = ref.watch(backendServiceProvider);
      return AssistantViewModel(backendService);
    });
