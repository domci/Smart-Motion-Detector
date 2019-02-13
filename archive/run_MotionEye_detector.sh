until python3 /home/cichons/Smart-Motion-Detector/MotionEye_detector.py; do
    echo "Object Detector crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
