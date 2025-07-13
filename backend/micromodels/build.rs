// backend/micromodels/build.rs (V2 - 带DLL复制功能)
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Protobuf 代码生成部分 (保持不变) ---
    println!("cargo:rerun-if-changed=src/proto/preprocessor.proto");
    prost_build::compile_protos(&["src/proto/preprocessor.proto"], &["src/proto/"])?;

    // --- DLL 自动复制部分 (新增) ---
    // 1. 获取 ONNX Runtime 库的源目录
    let ort_lib_location = env::var("ORT_LIB_LOCATION")
        .expect("ERROR: ORT_LIB_LOCATION environment variable not set. Please configure it in .cargo/config.toml");
    let src_dir = PathBuf::from(ort_lib_location);

    // 2. 获取 Cargo 的输出目录 (例如 .../target/debug)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // 我们需要的是 target/debug 或 target/release 目录，它在 OUT_DIR 的几层父目录之上
    let profile = env::var("PROFILE").unwrap(); // "debug" or "release"
    let target_dir = out_dir.ancestors()
        .find(|p| p.ends_with(&profile))
        .expect("Failed to find target directory");

    // 3. 定义要复制的DLL文件名
    let dll_files = ["onnxruntime.dll", "DirectML.dll"];

    // 4. 执行复制
    for file_name in &dll_files {
        let src_path = src_dir.join(file_name);
        let dest_path = target_dir.join(file_name);

        if src_path.exists() {
            println!("cargo:rerun-if-changed={}", src_path.display());
            fs::copy(&src_path, &dest_path)?;
            println!("Copied {} to {}", src_path.display(), dest_path.display());
        }
    }

    Ok(())
}