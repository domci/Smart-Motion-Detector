#!/usr/bin/env python




################################################
# Import Libraries
################################################

import os
#import sys
#import subprocess
#import collections
import time
#import mmap
#import pandas as pd
import json
import fnmatch
import cv2
#import argparse
import numpy as np
import requests
#import json
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
#import time





args =  {'classes': '/home/cichons/Smart-Motion-Detector/classes.txt',
         'weights': '/home/cichons/Smart-Motion-Detector/yolov3.weights',
         'config': '/home/cichons/Smart-Motion-Detector/yolov3.cfg'}


################################################
# Configure Log Handler
################################################

logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)

logname = '/home/cichons/unifi-video/logs/object_detected.log'
handler = TimedRotatingFileHandler(logname, when="midnight", interval=1)
handler.suffix = "%Y%m%d"
logger.addHandler(handler)






################################################
# Load Stuff
################################################

# Load Pushover Key File
data=json.loads(open('/home/cichons/Smart-Motion-Detector/keys.json').read())

# Load Camera UUIDs from file:
with open('/home/cichons/Smart-Motion-Detector/support_dbdevices.json') as f:
    support_dbdevices = json.load(f)

cameras = {}
for cam in support_dbdevices['cameras']:
    print('Found Camera: ' + cam['name'] + ' ' + cam['uuid'] )
    cameras[cam['name']] = cam['uuid']
del cam


classes = None

with open(args['classes'], 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))    






################################################
# Settings
################################################



LOG_FILE = '/home/cichons/unifi-video/logs/recording.log'
WATCH_FOR = 'STOPPING REC'

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
scale = 0.00392
time_between_push_notifications = 300 




target_classes = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'bus',
 'train',
 'truck',
]


# Pushover Settings:
data['priority'] = 1 #2
data['retry'] = 30 
data['expire'] = 300


################################################
# Defining Functions
################################################

# basic Python implementation of Unix tail for getting Tail of Log File
def tail(file, n):
    with open(file, "r") as f:
        f.seek (0, 2)           # Seek @ EOF
        fsize = f.tell()        # Get Size
        f.seek (max (fsize-1024, 0), 0) # Set pos @ last n chars
        lines = f.readlines()       # Read to end
    lines = lines[-n:]    # Get last n lines
    return lines




def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + ': ' + str(round(confidence*100, 2))
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)














print(
    'Watching of ' + LOG_FILE + ' for ' + '*' + WATCH_FOR + '*' +
    ' started at ' + time.strftime('%Y-%m-%d %I:%M:%S %p'))








################################################
# Watch Logfile for new Motion Recording
################################################


mtime_last = 0
dtime_last = 0
boxes_last =  []
net = cv2.dnn.readNet(args['weights'], args['config'])
mtime_cur = os.path.getmtime(LOG_FILE)



i = 1
while True:
    
    
    if mtime_cur != mtime_last:
        # Monitor Log File:
        for i in tail(LOG_FILE, 1):
            if WATCH_FOR.lower() in i.lower():
                camera = i.split("|")[1].split("]")[0]
                start = i.split("START:")[1].split(" ")[0]
                ts = datetime.datetime.fromtimestamp(int(start) / 1e3)
                recording_id = i.split("motionRecording:")[1].split(" ")[0]
                print(str(ts) + '   Found Motion Recording!')
                mtime_last = mtime_cur


                #print(start, cam)
                
                confidences = []
                
                for root, directories, filenames in os.walk('/home/cichons/unifi-video/videos/'):
                    for filename in filenames: 
                        if fnmatch.fnmatch(filename, '*' + start + '*.mp4'):
                            video_path = root +'/' + filename
                        if fnmatch.fnmatch(filename, '*' + recording_id + '*full.jpg'):
                            img_path = root +'/' + filename
                            print(str(ts) + '   Running Object detection on: \'' + img_path + '\' from Camera: \'' + camera +'\' ...')
                            logger.info(str(ts) + '   Trigger Count: ' + str(i) + 'Motion detected on Camera: \'' + camera +'\'   Running Object detection on: \'' + img_path + '\'')
    
                            # Detect Objects:
                            image = cv2.imread(img_path)
                            if image is not None:
                                #image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)
                                (Height, Width) = image.shape[:2]
                                blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
                                net.setInput(blob)
                                outs = net.forward(get_output_layers(net))
                                
                                class_ids = []
                                for out in outs:
                                    for detection in out:
                                        scores = detection[5:]
                                        class_id = np.argmax(scores)
                                        confidence = scores[class_id]
                                        if confidence > conf_threshold and classes[class_id] in target_classes:
                                            center_x = int(detection[0] * Width)
                                            center_y = int(detection[1] * Height)
                                            w = int(detection[2] * Width)
                                            h = int(detection[3] * Height)
                                            x = center_x - w / 2
                                            y = center_y - h / 2
                                            class_ids.append(class_id)
                                            confidences.append(float(confidence))
                                            boxes.append([x, y, w, h])
                                            
    
                                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                                if boxes != boxes_last:
                                    for i in indices:
                                        i = i[0]
                                        box = boxes[i]
                                        x = box[0]
                                        y = box[1]
                                        w = box[2]
                                        h = box[3]
                                        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
                                        
                                    boxed_img_path = '/home/cichons/unifi-video/object_detections/'+ str(start) +'_'+ str(camera) +'_'+ str(recording_id) + '.jpg' 
                                    cv2.imwrite(boxed_img_path, image)
                                    boxes_last = boxes
                                    
                                    cv2.waitKey()
                                    
                                    # Send Push Notification:
                                    if len(confidences):
                                        dtime_cur = time.time()
                                        if boxes == boxes_last:
                                            if (dtime_cur - dtime_last) > time_between_push_notifications:
                                            
                                                # Write to Log File
                                                logger.info(str(ts) + '   ' + ', '.join(list(set([classes[i] for i in class_ids]))) + 'Trigger Count: ' + str(i) + ' detected on Camera: \'' + camera +'\'   Video path is: \'' + video_path + '\'')
                                                
                                                print(', '.join(list(set([classes[i] for i in class_ids]))) + ' detected!')
                                                
                                                # Sent Push Notification via Pushover:
                                                data['message'] = ', '.join(list(set([classes[i] for i in class_ids]))) + ' detected!'
                                                r = requests.post("https://api.pushover.net/1/messages.json", data = data,
                                                files = {
                                                  "attachment": (filename, open(boxed_img_path, "rb"), "image/jpeg")
                                                })
                                                print(r.text)
                                                dtime_last = dtime_cur
                                            else:
                                                print("Detected: " + ', '.join(list(set([classes[i] for i in class_ids]))) + ". Last Notification too recently.")
                                            
                                    boxes = []
                                else:
                                    print("No Object Detected.")
    
                            #cv2.imshow("object detection", image)
    
                            else:
                                print('image: ' + img_path + ' not found')
    
    

#cv2.destroyAllWindows()
