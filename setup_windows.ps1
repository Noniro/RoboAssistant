# setup_windows.ps1
# Run this in PowerShell inside your new project folder on Windows

Write-Host "Setting up Robotic Lab Assistant on Windows..." -ForegroundColor Cyan

# 1. Create Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Upgrade Pip
python -m pip install --upgrade pip

# 3. Install Core Dependencies
# Note: We omit vllm (linux only) and use transformers/requests
pip install customtkinter opencv-python transformers torch torchvision torchaudio lerobot pyserial requests pillow qwen-vl-utils

Write-Host "------------------------------------------------"
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "To start the app, run: .\venv\Scripts\activate; python main_ui.py"
