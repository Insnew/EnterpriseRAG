@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if exist .env (
    echo .env 已存在，打开记事本供你检查或修改。
    notepad .env
) else (
    copy .env.example .env >nul
    echo.
    echo 已创建 .env 文件，即将用记事本打开。
    echo 请把 DEEPSEEK_API_KEY 和 SILICONFLOW_API_KEY 后面的 sk-xxx
    echo 替换成你的真实 key（DeepSeek 和硅基流动各一个），保存后关闭。
    echo.
    pause
    notepad .env
)
