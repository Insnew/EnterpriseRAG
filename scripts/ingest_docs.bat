@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 正在把 samples\ 目录的演示文档摄入知识库（需要已填好 .env）...
echo.
uv run python scripts/ingest.py default samples/
echo.
echo 摄入完成。下一步：双击 dev_start.bat 启动服务。
pause
