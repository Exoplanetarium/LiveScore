@echo off
echo Starting LiveScore Audio Analysis API...
echo.
echo The API will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.

cd /d "%~dp0"
call env\Scripts\activate.bat
python main.py
