#!/usr/bin/env python




################################################
# Import Libraries
################################################
from matplotlib import pyplot as plt
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
import glob
from imutils.video import VideoStream
from IPython.display import clear_output



args =  {'classes': '/home/cichons/Smart-Motion-Detector/classes.txt',
         'weights': '/home/cichons/Smart-Motion-Detector/yolov3.weights',
         'config': '/home/cichons/Smart-Motion-Detector/yolov3.cfg',
         'stream':'rtsp://172.17.0.2:7447/5b75de05e4b0a018229f268f_2',
         'cam_name': 'Carport'
        }


################################################
# Configure Log Handler
################################################

logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)

logname = '/home/cichons/motioneye/logs/object_detected.log'
handler = TimedRotatingFileHandler(logname, when="midnight", interval=1)
handler.suffix = "%Y%m%d"
logger.addHandler(handler)






################################################
# Load Stuff
################################################

# Load Pushover Key File
data=json.loads(open('/home/cichons/Smart-Motion-Detector/keys.json').read())




classes = None

with open(args['classes'], 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))    






################################################
# Settings
################################################

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.6
nms_threshold = 0.4
scale = 0.00392
time_between_push_notifications = 0 
px_dist = 70 # Minumum Distance in Pixels between current and prevouis Detection



target_classes = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'bus',
 'truck'
]


# Pushover Settings:
data['pushover']['priority'] = 1 #2
data['pushover']['retry'] = 30 
data['pushover']['expire'] = 300


################################################
# Defining Functions
################################################




def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + ': ' + str(round(confidence*100, 2))
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




mtime_last = 0
dtime_last = 0

net = cv2.dnn.readNet(args['weights'], args['config'])
centers_last = [-10, -10]
boxes_last = []




print("[INFO] warming up camera...")
vs = VideoStream(src=args['stream'], framerate=3).start()
time.sleep(2.0)


print('cam_name: ', args['cam_name'])



while True:
    image = vs.read()
    
    now = datetime.datetime.now()
    camera_name = args['cam_name']
    confidences = []
    

    if image is not None:
        print('detecting Objects...')
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
                        print(confidence)
                        try:
                            requests.get('http://192.168.1.22:8080/object_detected')
                        except Exception as e:
                            print(e)
                            continue
                        
                        print(classes[class_id] + ' detected')
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
                boxes_last = boxes
                centers_last = centers
                continue

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            
            if boxes == boxes_last:
                logger.info(str(datetime.datetime.now()) + '   ' + 'Objects where previously detected on: \'' + camera_name + '\'')
                print('Objects where previously detected.')
            else:
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

                boxed_img_path = '/home/cichons/motioneye/videos/detections/'+ str(now) +'_'+ str(camera_name) + '.jpg' 
                cv2.imwrite(boxed_img_path, image)
                    

                # Calculate relative Distances between centers of new and previous detections:                                        
                distances = abs(np.array([np.array(centers) - np.array(obj) for obj in centers_last]))
                if len(distances) == 0:
                    distances = 1000
                    distances

                print('distances ', distances)
                print('centers ', centers)
                print('centers_last ', centers_last)
                print('boxes differ ', boxes != boxes_last)
                print('confidences good ', confidences)
                print('distances ok ', np.max(distances) > px_dist, distances)
                print('max distance ', np.max(distances))
                print('time between notifications? ', (dtime_cur - dtime_last) > time_between_push_notifications)
                

                if boxes != boxes_last and confidences and np.max(distances) > px_dist and (dtime_cur - dtime_last) > time_between_push_notifications:

                    # Write to Log File
                    logger.info(str(datetime.datetime.now()) + '   ' + ', '.join(list(set([classes[i] for i in class_ids]))) + ' detected on Camera: \'' + camera_name + '\'')

                    print(', '.join(list(set([classes[i] for i in class_ids]))) + ' detected!')

                    # Sent Push Notification via Pushover:
                    data['pushover']['message'] = ', '.join(list(set([classes[i] for i in class_ids]))) + ' detected!'

                    r = requests.post("https://api.pushover.net/1/messages.json", data = data['pushover'],
                    files = {
                      "attachment": (str(now) +'_'+ str(camera_name) + '.jpg', open(boxed_img_path, "rb"), "image/jpeg")
                    })

                    print(r.text)
                    dtime_last = dtime_cur
                else:
                    print("Detected: " + ', '.join(list(set([classes[i] for i in class_ids]))) + ". Notification unwanted.")
                    print('boxes differ?', boxes != boxes_last)
                    print('confidences good?', confidences)
                    print('distances to previous detections ok?', np.max(distances) > px_dist, distances)
                    print('time between notifications?', (dtime_cur - dtime_last) > time_between_push_notifications)
                    continue

            boxes_last = boxes
            centers_last = centers
            #centers = []

        except Exception as e:
            print(e)
            continue

    else:
        print('image not found')
        continue