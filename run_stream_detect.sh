#!/bin/bash

/bin/sleep 5


# Create a new tmux session named helloworld...
tmux new-session -d -s object_detector

tmux send-keys -t object_detector "python3 /home/cichons/Smart-Motion-Detector/stream_detect.py" C-m
