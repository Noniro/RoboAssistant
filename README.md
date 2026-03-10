# RoboAssistant 

A modular Robotic Lab Assistant with Real-Time Vision, Face Recognition, and LLM-powered reasoning.

## Features
- **Real-Time Vision**: YOLOv4-tiny object detection and dlib-based face recognition.
- **Brain Module**: Integrates with LM Studio (Local LLM) for intelligent dialogue and intent determination.
- **Voice (TTS)**: Multi-threaded background Text-To-Speech using `pyttsx3`.
- **Hardware Integration**: Support for Generic Robotic Arm controllers.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `.\venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Ensure LM Studio is running on `localhost:1234`.

## Usage
Run the main supervisor UI:
```powershell
python main_ui.py
```
