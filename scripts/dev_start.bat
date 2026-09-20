@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 正在打开两个窗口：后端 API + 网页界面，请稍候...
start "RAG-后端 API" cmd /k "uv run uvicorn app.main:app --reload"
start "RAG-网页界面" cmd /k "uv run streamlit run ui/streamlit_app.py"
echo.
echo 两个窗口已打开（黑色的后端窗口和网页界面窗口）。
echo 等网页界面窗口出现网址后，用浏览器打开 http://localhost:8501 提问。
echo 全部用完想关闭时，把这两个窗口关掉即可。
