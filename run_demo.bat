@echo off
echo ====================================================================
echo  Starting LangChain RAG & Prompt Engineering Research Assistant
echo ====================================================================
echo.

cd /d "%~dp0"

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found in .\venv!
    echo Please create it using: python -m venv venv
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Streamlit Web Application on Localhost...
streamlit run app.py

pause
