// lib/core/models/backend_response.dart
import 'dart:convert';

class BackendResponse {
  final String response;

  BackendResponse({required this.response});

  // 一个工厂构造函数，用于从JSON Map创建实例
  factory BackendResponse.fromJson(Map<String, dynamic> json) {
    return BackendResponse(
      response: json['response'] ?? 'Error: Response field is missing.',
    );
  }
}