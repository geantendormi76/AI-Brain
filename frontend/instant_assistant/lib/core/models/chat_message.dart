// lib/core/models/chat_message.dart

class ChatMessage {
  final String text;
  final bool isUser; // true代表用户消息, false代表AI消息

  ChatMessage({required this.text, required this.isUser});
}
