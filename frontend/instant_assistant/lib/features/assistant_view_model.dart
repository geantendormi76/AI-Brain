// lib/features/assistant_view_model.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:instant_assistant/core/services/backend_service.dart';

// 1. 定义状态类 (保持不变)
class AssistantState {
  final bool isLoading;
  final String? response;
  final String? error;

  AssistantState({this.isLoading = false, this.response, this.error});

  AssistantState copyWith({bool? isLoading, String? response, String? error}) {
    return AssistantState(
      isLoading: isLoading ?? this.isLoading,
      response: response ?? this.response,
      error: error ?? this.error,
    );
  }
}

// 2. 创建一个BackendService的Provider
// 这样我们就可以在ViewModel中访问它
final backendServiceProvider = Provider<BackendService>((ref) => BackendService());


// 3. 创建 ViewModel (Notifier)
class AssistantViewModel extends StateNotifier<AssistantState> {
  // 注入BackendService
  final BackendService _backendService;

  // 构造函数现在接收一个service实例
  AssistantViewModel(this._backendService) : super(AssistantState(response: '你好，有什么可以帮您？'));

  // 业务方法：处理用户提交的查询
  Future<void> sendQuery(String query) async {
    state = state.copyWith(isLoading: true, error: null);

    try {
      // 调用真实的网络服务
      final backendResponse = await _backendService.sendQuery(query);
      // 更新状态为成功
      state = state.copyWith(isLoading: false, response: backendResponse.response);
    } catch (e) {
      // 如果发生任何错误，更新状态为失败
      state = state.copyWith(isLoading: false, error: e.toString());
    }
  }
}

// 4. 更新ViewModel的Provider，将service注入进去
final assistantViewModelProvider =
    StateNotifierProvider<AssistantViewModel, AssistantState>((ref) {
  // 从ref中读取BackendService的实例
  final backendService = ref.watch(backendServiceProvider);
  // 将实例传递给ViewModel
  return AssistantViewModel(backendService);
});