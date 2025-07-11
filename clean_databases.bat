@echo off
title AI Brain Database Cleaner

echo Cleaning up databases...
echo.

echo Deleting SQLite database...
set SQLITE_PATH=%USERPROFILE%\.memos_agent
if exist "%SQLITE_PATH%" (
    RMDIR /S /Q "%SQLITE_PATH%"
    echo SQLite database deleted successfully.
) else (
    echo SQLite database at "%SQLITE_PATH%" not found, skipping.
)

echo.

echo Deleting Qdrant vector database...
set QDRANT_PATH=%APPDATA%\com.tauri.dev\zhzAI_data
if exist "%QDRANT_PATH%" (
    RMDIR /S /Q "%QDRANT_PATH%"
    echo Qdrant database deleted successfully.
) else (
    echo Qdrant database at "%QDRANT_PATH%" not found, skipping.
)

echo.
echo Cleanup complete.
pause