#!/usr/bin/env python




################################################
# Import Libraries
################################################
#from matplotlib import pyplot as 
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
from flask import Flask, request, abort
import glob



args =  {'classes': '/home/cichons/Smart-Motion-Detector/classes.txt',
         'weights': '/home/cichons/Smart-Motion-Detector/yolov3.weights',
         'config': '/home/cichons/Smart-Motion-Detector/yolov3.cfg'
         
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





app = Flask(__name__)


mtime_last = 0
dtime_last = 0

net = cv2.dnn.readNet(args['weights'], args['config'])
centers_last = [-10, -10]
boxes_last = []


@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
    print(str(datetime.datetime.now()))
    global class_ids
    global confidences
    global boxes
    global conf_threshold
    global nms_threshold
    global scale
    global time_between_push_notifications
    global px_dist
    global target_classes
    global boxes_last
    global mtime_last
    global dtime_last
    global net
    global centers_last
    global data
    global logger
    
    if request.remote_addr != '172.17.0.3':
        print('Access denied!')
        logger.info(str(datetime.datetime.now()) + '   ' + 'Access denied from: \'' + request.remote_addr + '\'')
        abort(403)
    
    now = datetime.datetime.now()
    camera_name_id = request.args.get('cam_id')
    camera_name = request.args.get('cam_name')
    confidences = []
    
    logger.info(str(datetime.datetime.now()) + '   ' + 'Motion detected on Camera: \'' + camera_name + '\'')
    
    print('Webhook received.')
    print('cam_id: ', str(request.args.get('cam_id')))
    print('cam_name: ', request.args.get('cam_name'))



    # Take Snapshot
    #print('taking snapshot...')
    #img_request = requests.get('http://192.168.1.233:7999/1/action/snapshot')
    print('reading image...')
    list_of_files = glob.glob('/home/cichons/motioneye/videos/Camera1/' +str(now.year) + '-' + str(now.month) + '-' + str(now.day) +'/*.jpg') 
    latest_file = max(list_of_files, key=os.path.getctime)
    image = cv2.imread(latest_file)
    #image = cv2.imread('/home/cichons/motioneye/videos/Camera' + str(camera_name_id) + '/lastsnap.jpg')    
    #image = cv2.imread('/home/cichons/tmp/pexels-photo-109919.jpeg') 
    # plot image
    # plt.imshow(image)
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
                            pass

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
                return 'No Object detected.'

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

                boxed_img_path = '/home/cichons/motioneye/videos/detections/'+ str(now) +'_'+ str(camera_name_id) + '.jpg' 
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
                
                logger.info(str(datetime.datetime.now()) + '   ' + 'Access denied from: \'' + request.remote_addr + '\'')

                if boxes != boxes_last and confidences and np.max(distances) > px_dist and (dtime_cur - dtime_last) > time_between_push_notifications:

                    # Write to Log File
                    logger.info(str(datetime.datetime.now()) + '   ' + ', '.join(list(set([classes[i] for i in class_ids]))) + ' detected on Camera: \'' + camera_name + '\'')

                    print(', '.join(list(set([classes[i] for i in class_ids]))) + ' detected!')

                    # Sent Push Notification via Pushover:
                    data['pushover']['message'] = ', '.join(list(set([classes[i] for i in class_ids]))) + ' detected!'

                    r = requests.post("https://api.pushover.net/1/messages.json", data = data['pushover'],
                    files = {
                      "attachment": (str(now) +'_'+ str(camera_name_id) + '.jpg', open(boxed_img_path, "rb"), "image/jpeg")
                    })

                    print(r.text)
                    dtime_last = dtime_cur
                else:
                    print("Detected: " + ', '.join(list(set([classes[i] for i in class_ids]))) + ". Notification unwanted.")
                    print('boxes differ', boxes != boxes_last)
                    print('confidences good', confidences)
                    print('distances to previous detections ok', np.max(distances) > px_dist, distances)
                    print('time between notifications?', (dtime_cur - dtime_last) > time_between_push_notifications)
                    pass

            boxes_last = boxes
            centers_last = centers
            #centers = []

        except Exception as e:
            print(e)
            pass

    else:
        print('image not found')
        pass
    return 'OK.'


print('Object Detector waiting for Webhook from MotionEye. ' + time.strftime('%Y-%m-%d %I:%M:%S %p'))

if __name__ == '__main__':
    #app.run()
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 5000, app)
