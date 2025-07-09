// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart'; // 1. 导入 riverpod
import 'package:instant_assistant/features/assistant_view.dart';

void main() {
  // 2. 用 ProviderScope 包裹 runApp
  runApp(const ProviderScope(child: MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Instant Assistant',
      theme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.blue,
        scaffoldBackgroundColor: const Color(0xFF202124),
      ),
      home: const AssistantView(),
      debugShowCheckedModeBanner: false,
    );
  }
}