@echo off
title AI Brain Database Cleaner V2.0 (Dev & Prod)

echo Cleaning up databases...
echo.

echo [1/4] Deleting Production Qdrant database...
set PROD_QDRANT_PATH=%APPDATA%\com.tauri.dev\zhzAI_data
if exist "%PROD_QDRANT_PATH%" (
    RMDIR /S /Q "%PROD_QDRANT_PATH%"
    echo    -> Production Qdrant database deleted successfully.
) else (
    echo    -> Production Qdrant database at "%PROD_QDRANT_PATH%" not found, skipping.
)
echo.

echo [2/4] Deleting Development Qdrant database...
rem 获取批处理文件所在的目录，并假定qdrant可执行文件位于上一级的services目录
set DEV_QDRANT_PATH=%~dp0..\frontend\instant_assistant\src-tauri\services\storage
if exist "%DEV_QDRANT_PATH%" (
    RMDIR /S /Q "%DEV_QDRANT_PATH%"
    echo    -> Development Qdrant database deleted successfully.
) else (
    echo    -> Development Qdrant database at "%DEV_QDRANT_PATH%" not found, skipping.
)
echo.

echo [3/4] Deleting Production SQLite database...
set PROD_SQLITE_PATH=%USERPROFILE%\.memos_agent
if exist "%PROD_SQLITE_PATH%" (
    RMDIR /S /Q "%PROD_SQLITE_PATH%"
    echo    -> Production SQLite database deleted successfully.
) else (
    echo    -> Production SQLite database at "%PROD_SQLITE_PATH%" not found, skipping.
)
echo.

echo [4/4] Deleting Development SQLite database...
rem 开发环境的SQLite数据库通常就在backend的.memos_agent目录
rem 但为了保险，我们还是用USERPROFILE
set DEV_SQLITE_PATH=%USERPROFILE%\.memos_agent
if exist "%DEV_SQLITE_PATH%" (
    if "%PROD_SQLITE_PATH%" neq "%DEV_SQLITE_PATH%" (
        RMDIR /S /Q "%DEV_SQLITE_PATH%"
        echo    -> Development SQLite database deleted successfully.
    ) else (
        echo    -> Same as production path, already handled.
    )
) else (
    echo    -> Development SQLite database at "%DEV_SQLITE_PATH%" not found, skipping.
)


echo.
echo Cleanup complete.
pause