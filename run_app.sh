#!/bin/bash
# Apply permissions just in case
sudo chmod 666 /dev/video0 /dev/video1

# Load the virtual environment
source venv/bin/activate

# Launch with V4L2 compatibility wrapper (essential for many webcams in WSL)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libv4l/v4l2convert.so
python main_ui.py
