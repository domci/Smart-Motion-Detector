# Deep Learning upgrade for Ubiquiti UVC-G3 (or any) Surveillance Camera in Python 3

I own a lot of Smart Home gadgets and alliances already so the next logical step was to upgrade my Smart Home with a Surveilliance System.

After some research I settled on Ubiquiti UVC-G3 as its price is reasonable, it has an API (which turned out I do not to need) and also I like its design more than most other cameras'. Only drawback is that it does not come with a smart motion detector, wich is particularly important to me, because I don't want to be alerted every time some tree branch moves in front of the camera. 

There are some Cameras with smart motion detection and even face recognition available but they tend to be quite expensive and have a closed ecosystem. A No-Go for me since I like playing around with my gadgets freely.

So after some research and thanks to [Arun Ponnusamy Blog Post](https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html) I set up a System that uses the G3 Cameras in-built Motion Detector and runs a pretrained YOLO Algorithm (You Only Look Once, see: https://pjreddie.com/darknet/) to check each motion capture for objects of interest.

## How does it work?

Well, as it turned out it's quite simple to use the G3 Camera (with the Ubuquiti NVR) for Object Detection. The plan was to write a Python Application that streams a live stream from the camera and performs Object detection on each frame. I realised that this might be a bit over the top as the camera has 3 Frames per Second and I might add more cameras later. So I already knew that this solution will need some computational power. 

While playing around with my new gadget I found out that the NVR stores a .JPG Snapshot of every Motion alarm and also logs everything nicely in logfiles. This brought me to the idea to scan and parse the motion-logfile instead of using the live stream. Ant that turned out to work like a charm!

So the general flow is:

1. NVR (camera) detects Motion
2. writes new line to motion-logfile
3. detector.py scans logfile
4. detector.py parses snapshot path
5. detector.py runs object detection
6. if interesting object was detected send push Notification via [Pushover](https://pushover.net)


