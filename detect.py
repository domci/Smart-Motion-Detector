#!/usr/bin/env python




################################################
# Import Libraries
################################################

import os
import time
import json
import fnmatch
import cv2
import numpy as np
import requests
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import datetime
import shelve


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
conf_threshold = 0.6
nms_threshold = 0.4
scale = 0.00392
time_between_push_notifications = 0 
px_dist = 30 # Minumum Distance in Pixels between current and prevouis Detection



target_classes = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'bus',
 'truck'
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














print('Watching of ' + LOG_FILE + ' for ' + '*' + WATCH_FOR + '*' +
    ' started at ' + time.strftime('%Y-%m-%d %I:%M:%S %p'))








################################################
# Watch Logfile for new Motion Recording
################################################


mtime_last = 0
dtime_last = 0

net = cv2.dnn.readNet(args['weights'], args['config'])
mtime_cur = datetime.datetime.now()
recording_id = 'None'
recording_id_last = ''
centers_last_3rd = [-10, -10]
centers_last = [-10, -10]
boxes_last = []

    
    
i = 1
while True:
    try:
        now = datetime.datetime.now()
        date_path = str(now.year) + '/' + (now.month if len(str(now.month)) == 2 else '0' + str(now.month)) + '/' + str(now.day)
        # Monitor Log File:
        for i in tail(LOG_FILE, 1):
            if WATCH_FOR.lower() in i.lower():
                mtime_cur = datetime.datetime.fromtimestamp(float(i.split(' ')[0]))
                camera_name_id = i.split("[")[1].split("]")[0].split("|")
                start = i.split("START:")[1].split(" ")[0]
                ts = datetime.datetime.fromtimestamp(int(start) / 1e3)
                recording_id = i.split("motionRecording:")[1].split(" ")[0]
                #recording_id = '5ba25ea1e4b0a01868310e29'
                if recording_id == recording_id_last:
                    continue
                
                print('-------------------------------------------------------------------------------------------------------------------------------------')    
                print(str(ts) + '   Found Motion Recording on Camera ' + ' '.join(camera_name_id) + '. Recording ID is: ' + recording_id)



                confidences = []

                for root, directories, filenames in os.walk('/home/cichons/unifi-video/videos/'+ cameras[camera_name_id[1]] + '/' + date_path):
                    for filename in filenames: 
                        if fnmatch.fnmatch(filename, '*' + start + '*.mp4'):
                            video_path = root +'/' + filename
                        if fnmatch.fnmatch(filename, '*' + recording_id + '*full.jpg'):
                            img_path = root +'/' + filename
                            print(str(ts) + '   Running Object detection on: \'' + img_path + '\' from Camera: \'' + ' '.join(camera_name_id) +'\' ...')
                            logger.info(str(ts) + '   Trigger Count: ' + str(i) + 'Motion detected on Camera: \'' + ' '.join(camera_name_id) +'\'   Running Object detection on: \'' + img_path + '\'')

                            # Detect Objects:
                            image = cv2.imread(img_path)
                            if image is not None:
                                try:
                                    (Height, Width) = image.shape[:2]
                                    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
                                    net.setInput(blob)
                                    outs = net.forward(get_output_layers(net))

                                    boxes = []
                                    class_ids = []
                                    dtime_cur = time.time()
                                    confidences = []
                                    centers = []
                                    for out in outs:
                                        for detection in out:
                                            scores = detection[5:]
                                            class_id = np.argmax(scores)
                                            confidence = scores[class_id]
                                            if confidence > conf_threshold and classes[class_id] in target_classes:
                                                print('confidence and class good')
                                                center_x = int(detection[0] * Width)
                                                center_y = int(detection[1] * Height)
                                                centers.append([center_x, center_y])
                                                w = int(detection[2] * Width)
                                                h = int(detection[3] * Height)
                                                x = center_x - w / 2
                                                y = center_y - h / 2
                                                class_ids.append(class_id)
                                                confidences.append(float(confidence))
                                                boxes.append([x, y, w, h])
                                    if boxes == []:
                                        print("No Object Detected.")
                                        continue

                                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                                    if boxes != boxes_last:
                                        print(boxes)
                                        print(boxes_last)
                                        for i in indices:
                                            i = i[0]
                                            box = boxes[i]
                                            x = box[0]
                                            y = box[1]
                                            w = box[2]
                                            h = box[3]
                                            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

                                        boxed_img_path = '/home/cichons/unifi-video/object_detections/'+ str(start) +'_'+ str(camera_name_id[1]) +'_'+ str(recording_id) + '.jpg' 
                                        cv2.imwrite(boxed_img_path, image)


                                        cv2.waitKey()
                                        # Calculate relative Distances between centers of new and previous detections:                                        
                                        distances = abs(np.array([(centers_last) - np.array(centers), abs((centers_last_3rd) - np.array(centers))]))

                                        
                                        print('centers', centers)
                                        print('centers_last', centers_last)
                                        
                                        if boxes != boxes_last and confidences and np.max(distances) > px_dist and (dtime_cur - dtime_last) > time_between_push_notifications:
                                            print('boxes differ', boxes != boxes_last)
                                            print('confidences good', confidences)
                                            print('distances ok', np.max(distances) > px_dist, distances)
                                            print('time between notifications?', (dtime_cur - dtime_last) > time_between_push_notifications)


                                            # Write to Log File
                                            logger.info(str(ts) + '   ' + ', '.join(list(set([classes[i] for i in class_ids]))) + 'Trigger Count: ' + str(i) + ' detected on Camera: \'' + ' '.join(camera_name_id) +'\'   Video path is: \'' + video_path + '\'')

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
                                            print("Detected: " + ', '.join(list(set([classes[i] for i in class_ids]))) + ". Notification unwanted.")
                                            print('boxes differ', boxes != boxes_last)
                                            print('confidences good', confidences)
                                            print('distances to previous detections ok', np.max(distances) > px_dist, distances)
                                            print('time between notifications?', (dtime_cur - dtime_last) > time_between_push_notifications)
                                            continue

                                    boxes_last = boxes
                                    centers_last_3rd = centers_last
                                    centers_last = centers

                                except Exception as err:
                                    print(err)
                                    continue

                            else:
                                print('image: ' + img_path + ' not found')
                                continue
                recording_id_last = recording_id
                recording_id = ''
    except Exception as err:
        print(err)
        continue