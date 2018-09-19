until python3 /home/cichons/Smart-Motion-Detector/detect.py; do
    echo "Pbject Detector crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
