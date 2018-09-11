# Deep Learning upgrade for Ubiquiti UVC-G3 (or any) Surveillance Camera

I own a lot of Smart Home gadgets and alliances already so the next logical step was to upgrade my Smart Home with a Surveilliance System.

After some research I settled on Ubiquiti UVC-G3 as its price is reasonable, it has an API (which turned out I do not to need) and also I like its design more than most other cameras'. Only drawback is that it does not come with a smart motion detector, wich is particularly important to me, because I don't want to be alerted every time some tree branch moves in front of the camera. 

There are some Cameras with smart motion detection and even face recognition available but they tend to be quite expensive and have a closed ecosystem. A No-Go for me since I like playing around with my gadgets freely.

So after some research and thanks to Arun Ponnusamy Blog Post (https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html) I set up a System that uses the G3 Cameras in-built Motion Detector and runs a pretrained YOLO Algorithm (You Only Look Once, see: https://pjreddie.com/darknet/) to check each motion capture for objects of interest.

tbc
