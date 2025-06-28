Write-Host "Starting LiveScore Audio Analysis API..." -ForegroundColor Green
Write-Host ""
Write-Host "The API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API documentation at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot
& ".\env\Scripts\Activate.ps1"
python main.py
