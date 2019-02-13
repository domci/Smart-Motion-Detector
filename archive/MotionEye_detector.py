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
import glob
import pytz



# YOLO v3
args =  {'classes': '/playground/Smart-Motion-Detector/classes.txt',
         'weights': '/playground/Smart-Motion-Detector/yolov3.weights',
         'config': '/playground/Smart-Motion-Detector/yolov3.cfg'}



################################################
# Configure Log Handler
################################################

logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)

logname = '/playground/motioneye/logs/object_detected.log'
handler = TimedRotatingFileHandler(logname, when="midnight", interval=1)
handler.suffix = "%Y%m%d"
logger.addHandler(handler)






################################################
# Load Stuff
################################################

# Load Pushover Key File
pushover=json.loads(open('/playground/Smart-Motion-Detector/keys.json').read())



with open(args['classes'], 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))    






################################################
# Settings
################################################


### Detection Settings:
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.6
nms_threshold = 0.4
scale = 0.00392
time_between_detections = 10 #Seconds between Detections 
px_dist = 70 # Minumum Distance in Pixels between current and prevouis Detection
target_classes = ['person']
dtime_last = 0
net = cv2.dnn.readNet(args['weights'], args['config'])
centers_last = [-10, -10]
boxes_last = []

# Pushover Settings:
pushover['priority'] = 1 #2
pushover['retry'] = 30 
pushover['expire'] = 300


params = {
'conf_threshold' : 0.6,
'nms_threshold' : 0.4,
'scale' : 0.00392,
'px_dist' : 70, # Minumum Distance in Pixels between current and prevouis Detection
'target_classes' : ['person'],
'net' : cv2.dnn.readNet(args['weights'], args['config']),
'centers_last' : [-10, -10],
'boxes_last' : []
}



################################################
# Define Functions
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





def detect(img_path='/playground/Smart-Motion-Detector/ShamblesYorkEngland.jpg', params=params, pushover=pushover, boxes_last = [], centers_last = [-10, -10]):
    print('----------------------- ' + str(datetime.datetime.now().astimezone(pytz.timezone("Europe/Berlin"))) + ' -----------------------')
    try:
        image = cv2.imread(img_path)
        if image is not None:
                
                #init stuff
                (Height, Width) = image.shape[:2]
                conf_threshold = params['conf_threshold']
                nms_threshold = params['nms_threshold']
                scale = params['scale']
                px_dist = params['px_dist']
                net = params['net']
                blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

                net.setInput(blob)
                outs = net.forward(get_output_layers(net))
                boxes = []
                class_ids = []

                confidences = []
                centers = []
                net.setInput(blob)                
                target_classes = params['target_classes']
                
                
                
                
                # Detect!
                outs = net.forward(get_output_layers(net))
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > conf_threshold and classes[class_id] in target_classes:
                            
                            # inform Home Dasboard:
                            # requests.get('http://192.168.1.229:8080/object_detected')
                            
                            print(classes[class_id], ' detected!')
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
                    centers = [-10, -10]
                    boxes_last = []
                    detection = False
                    return boxes_last, centers_last, detection 

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

                    boxed_img_path = '/playground/motioneye/detections/'+ os.path.basename('/playground/motioneye/videos/Camera1/2018-12-19/08-10-32.mp4.thumb').split('.')[0]+'.jpg'
                    cv2.imwrite(boxed_img_path, image)


                    cv2.waitKey()
                    # Calculate relative Distances between centers of new and previous detections:                                        
                    distances = abs(np.array([np.array(centers) - np.array(obj) for obj in centers_last]))


                    print('centers', centers)
                    print('centers_last', centers_last)

                    if boxes != boxes_last and confidences and np.max(distances) > px_dist:
                        print('boxes differ', boxes != boxes_last)
                        print('confidences good', confidences)
                        print('distances ok', np.max(distances) > px_dist, distances)
                        print(', '.join(list(set([classes[i] for i in class_ids]))) + ' detected!')

                        # Sent Push Notification via Pushover:
                        pushover['message'] = ', '.join(list(set([classes[i] for i in class_ids]))) + ' detected!'

                        r = requests.post("https://api.pushover.net/1/messages.json", data = pushover,
                        files = {
                          "attachment": (os.path.basename(img_path), open(boxed_img_path, "rb"), "image/jpeg")
                        })

                        print(r.text)
                        detection = True
                    else:
                        print("Detected: " + ', '.join(list(set([classes[i] for i in class_ids]))) + ". Notification unwanted.")
                        print('boxes differ', boxes != boxes_last)
                        print('confidences good', confidences)
                        print('distances to previous detections ok', np.max(distances) > px_dist, distances)
                        centers = [-10, -10]
                        boxes_last = []
                        detection = False
                        return boxes_last, centers_last, detection

                boxes_last = boxes
                centers_last = centers

        else:
            print('image: ' + img_path + ' not found')
            centers = [-10, -10]
            boxes_last = []
            detection = False
            return boxes_last, centers_last, detection
    
    except Exception as err:
        print(err)
        centers = [-10, -10]
        boxes_last = []
        detection = False
        return boxes_last, centers_last, detection

    return boxes_last, centers_last, detection
    
#detect()

    









####
last_image = None #list_of_images[3]
run_detection = False
####

while True: #for i in range(100):
    try:
        now = datetime.datetime.now().astimezone(pytz.timezone("Europe/Berlin"))
        date_path = str(now.year) + '-' + str(now.month) + '-' + ('0' + str(now.day) if now.day < 10 else str(now.day))
        media_path = '/playground/motioneye/videos/Camera*/' + date_path + '/*.jpg' # '/*.thumb'
        list_of_images = glob.glob(media_path) 
        list_of_images.sort(key=os.path.getmtime, reverse=True)


        #print(list_of_images[:10])

        # only keep files after the last seen one:
        if last_image == None:
            print('running initial detection on latest Snapshot.')
            new_images = [list_of_images[0]]
            run_detection = True
            last_image = list_of_images[0]
        else:
            if last_image in list_of_images:
                next_image_idx = [i for i, n in enumerate(list_of_images)  if last_image == n][0] -1
            else:
                next_image_idx = 0
            if next_image_idx < 0:
                #print('no new Snapshot available.')
                pass    
            else:
                if next_image_idx == 0:
                    new_images = [list_of_images[next_image_idx]]

                else:
                    new_images = list_of_images[:next_image_idx]

                print('found new Snapshot(s):',  new_images)
                run_detection = True
                last_image = new_images[0]

        if run_detection == True:
            # Do stuff with new_images here (Detect the shit out of it!)
            for img in new_images:
                dtime_cur = time.time() 
                if (dtime_cur - dtime_last) > time_between_detections:
                    boxes_last, centers_last, detection = detect(img_path=img, params=params)
                    if detection == True:
                        dtime_last = dtime_cur
                else:
                    print('last detection too recently!')
            run_detection = False
    except Exception as err:
        print(err)
        continue
    #time.sleep(1)