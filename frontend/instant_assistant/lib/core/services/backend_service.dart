// lib/core/services/backend_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:instant_assistant/core/models/backend_response.dart';

class BackendService {
  // 您的Rust后端将来会运行在这个地址上
  // 我们先用一个占位符，因为现在后端还没运行
  final String _baseUrl = 'http://localhost:8282/api/v1/dispatch'; // 注意：这里的端口要和你的Rust后端匹配

  Future<BackendResponse> sendQuery(String query) async {
    try {
      final response = await http.post(
        Uri.parse(_baseUrl),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'ProcessText': query}), // 匹配您Rust后端的Command::ProcessText枚举
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = json.decode(utf8.decode(response.bodyBytes));
        // 假设您的Rust后端直接返回 { "Text": "这是回答" } 这样的格式
        // 我们需要从中提取出文本
        if (responseData.containsKey('Text')) {
           return BackendResponse.fromJson({'response': responseData['Text']});
        } else {
           throw Exception('Backend response did not contain "Text" field.');
        }
      } else {
        // 如果服务器返回了非200的状态码，我们也抛出异常
        throw Exception('Failed to query backend. Status code: ${response.statusCode}');
      }
    } catch (e) {
      // 捕获所有类型的错误（网络错误、解析错误等）并重新抛出
      throw Exception('Failed to connect to backend: $e');
    }
  }
}