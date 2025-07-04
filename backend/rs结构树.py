import os
import json

def get_project_info(root_dir):
    """
    获取项目信息，包括目录结构、核心代码文件内容和依赖信息。
    该报告会过滤掉虚拟环境、缓存目录和常见的二进制文件，
    只保留对AI分析项目核心有用的信息。

    Args:
        root_dir (str): 要分析的根目录路径。

    Returns:
        dict: 包含项目信息的字典。
    """
    project_info = {
        "directory_tree": {},
        "core_code_info": [],
        "dependencies": {}
    }

    # Define directories to ignore
    ignored_dirs = [
        'venv', '.venv', 'env', '__pycache__', '.git', '.idea',
        'node_modules', 'dist', 'build', '.vscode', '.pytest_cache',
        'site-packages', 'target' # Common Python, Rust, and general build/cache directories
    ]
    # Define file extensions to ignore (binary files, logs, config files, etc.)
    # This list is more comprehensive, inspired by the provided py结构树.py
    ignored_extensions = [
        '.log', '.bin', '.dll', '.exe', '.so', '.dylib', '.DS_Store',
        '.pyc', '.pyo', '.ipynb_checkpoints', '.tmp', '.bak', '.swp',
        '.sqlite3', '.db', '.dat', '.json', '.yaml', '.yml', '.toml', # .toml is handled separately for dependencies
        '.xml', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pdf',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico',
        '.zip', '.tar', '.gz', '.rar', '.7z', '.mp3', '.mp4', '.avi',
        '.lock', # Cargo.lock is handled separately for dependencies
    ]

    # Define core code file extensions to read content from
    core_code_extensions = ['.rs', '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp']

    # Traverse directories, build directory tree and collect core code info
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter out ignored directories in place
        dirnames[:] = [d for d in dirnames if d not in ignored_dirs]

        # Calculate relative path for structured output
        relative_path = os.path.relpath(dirpath, root_dir)
        
        # Navigate to the current level in the directory tree dictionary
        current_level = project_info["directory_tree"]
        if relative_path != ".":
            parts = relative_path.split(os.sep)
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

        # Add files to the directory tree and process their content
        for f in filenames:
            file_full_path = os.path.join(dirpath, f)
            
            # Check if the file should be ignored based on its extension
            if any(f.endswith(ext) for ext in ignored_extensions):
                current_level[f] = "ignored_file" # Mark as ignored in tree
                continue

            current_level[f] = "file" # Default mark as file

            # Identify and read core code files
            if any(f.endswith(ext) for ext in core_code_extensions):
                try:
                    with open(file_full_path, 'r', encoding='utf-8') as code_file:
                        content = code_file.read()
                        project_info["core_code_info"].append({
                            "file_path": os.path.join(relative_path, f) if relative_path != "." else f,
                            "content": content
                        })
                except Exception as e:
                    print(f"Error reading core code file {file_full_path}: {e}")

            # Identify and read dependency files (e.g., Cargo.toml, Cargo.lock)
            if f in ["Cargo.toml", "Cargo.lock"]:
                try:
                    with open(file_full_path, 'r', encoding='utf-8') as dep_file:
                        project_info["dependencies"][os.path.join(relative_path, f) if relative_path != "." else f] = dep_file.read()
                except Exception as e:
                    print(f"Error reading dependency file {file_full_path}: {e}")

    return project_info

def print_directory_tree_console(tree, indent=0, is_last=False, prefix=""):
    """
    递归打印目录树到控制台，更接近传统树形图。
    """
    items = list(tree.items())
    for i, (key, value) in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        current_prefix = prefix + ("    " if is_last else "│   ")

        if key == "file": # Skip the 'file' marker itself
            continue

        if isinstance(value, dict): # It's a directory
            print(f"{prefix}{connector}{key}/")
            print_directory_tree_console(value, indent + 1, i == len(items) - 1, current_prefix)
        else: # It's a file or ignored_file
            print(f"{prefix}{connector}{key} ({value})") # Show file type (file/ignored_file)

if __name__ == "__main__":
    # Please replace '.' with the actual path to your project root directory
    # For example: root_directory = "/path/to/your/project"
    root_directory = "."

    print(f"Analyzing project: {os.path.abspath(root_directory)}\n")

    # Ensure root directory exists
    if not os.path.isdir(root_directory):
        print(f"Error: The specified path does not exist or is not a directory: {root_directory}")
    else:
        info = get_project_info(root_directory)

        print("Project Directory Structure:")
        # Start printing from the root, showing its name
        print(f"{os.path.basename(os.path.abspath(root_directory))}/")
        print_directory_tree_console(info["directory_tree"])
        print("\n" + "="*50 + "\n")

        print("Core Code Information:")
        if info["core_code_info"]:
            for code_file in info["core_code_info"]:
                print(f"File Path: {code_file['file_path']}")
                print("--- Code Content Start ---")
                print(code_file['content'])
                print("--- Code Content End ---\n")
        else:
            print("No core code files found (e.g., .rs, .py, .js, etc.).")
        print("\n" + "="*50 + "\n")

        print("Dependency Information:")
        if info["dependencies"]:
            for dep_file_path, dep_content in info["dependencies"].items():
                print(f"Dependency File: {dep_file_path}")
                print("--- Dependency Content Start ---")
                print(dep_content)
                print("--- Dependency Content End ---\n")
        else:
            print("No dependency files found (e.g., Cargo.toml, Cargo.lock).")
        print("\n" + "="*50 + "\n")

        # Save all information to a JSON file for external AI analysis
        output_filename = "project_analysis.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
        print(f"All information has been saved to {output_filename} file.")
        print("Operation completed.")

