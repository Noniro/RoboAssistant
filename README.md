# RoboAssistant: Multimodal Robot Orchestrator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/openvla/openvla-7b)
[![LeRobot](https://img.shields.io/badge/LeRobot-Integrated-green)](https://github.com/huggingface/lerobot)

A robotic control system for high-level reasoning and precision execution. RoboAssistant integrates Vision-Language-Action (VLA) models and Action Chunking Transformers (ACT) for intelligent laboratory automation.

---

## Performance Showcase

### Precision Manipulation (ACT)
*High-speed movements learned from demonstrations using transformer-based action chunking.*

![ACT Demonstration](docs/assets/act_demo.gif)
> *[Optional: Add your ACT recording here: `docs/assets/act_demo.gif`]*

### Real-Time Reasoning (OpenVLA)
*The robot interprets visual context and natural language to plan maneuvers.*

![OpenVLA Reasoning](docs/assets/vla_inference.gif)
> *[Optional: Add your OpenVLA recording here: `docs/assets/vla_inference.gif`]*

---

## Key Features

### Multimodal Intelligence
- **OpenVLA Integration**: Utilizes `openvla-7b` for zero-shot and LoRA fine-tuned task execution.
- **ACT Policies**: High-precision control for specialized manipulation tasks.
- **Local Brain**: Uses LM Studio for private LLM reasoning and intent parsing.

### Perception & Vision
- **Object Detection**: Real-time identification using YOLOv4-tiny.
- **Face Recognition**: Personalized interaction via `dlib` and `face_recognition`.
- **Multicam Processing**: Support for multiple camera streams (cam_high, cam_front).

### Hardware & UI
- **Unified Dashboard**: CustomTkinter UI for monitoring and manual control.
- **Voice Feedback**: Multithreaded TTS engine using `pyttsx3`.
- **Robot Agnostic**: Configurable profiles compatible with LeRobot standards.

---

## Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU (8GB+ VRAM for 4-bit OpenVLA)
- LM Studio (Running on `localhost:1234`)

### Installation

1. **Clone & Environment**
   ```bash
   git clone https://github.com/YourUsername/RoboAssistant.git
   cd RoboAssistant
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install transformers accelerate bitsandbytes peft
   ```

---

## Project Structure

- **`arm_control.py`**: Hardware communication and movement planning.
- **`vision_module.py`**: Vision pipeline (Detection, Face ID).
- **`brain_module.py`**: LLM orchestration and command logic.
- **`train_openvla_lora.py`**: OpenVLA adaptation for specific environments.
- **`unified_vla_worker.py`**: Integrated reasoning and action execution.

---

## Roadmap & Future Work

The primary objective is to achieve full autonomy through advanced model training and edge-deployment integrations.

- [ ] **VLA Mastery**: Successful fine-tuning of OpenVLA for specific laboratory pick-and-place tasks (Current Priority).
- [ ] **Edge Deployment**: Port the entire stack to **Raspberry Pi AI** or **NVIDIA AI Toolkits** for standalone operation.
- [ ] **Full Autonomy**: Transition from human-triggered tasks to self-governed routine execution.
- [ ] **Enhanced Locomotion**: Integration for mobile robot bases.

---
*Developed for the Advanced Robotics Community.*
