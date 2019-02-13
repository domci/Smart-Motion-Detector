

"""
##############
# TODO
#############

# Put paths/IPs into arguments
# Add Bounding Boxes
# Figure out additional Cameras
#

"""



from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
#import imutils
import time
import cv2
import json
import requests
import os
import datetime
import threading
import sys
from threading import Lock





    
args = {
    'model': '/home/cichons/Smart-Motion-Detector/MobileNetSSD_deploy.caffemodel',
    'prototxt': '/home/cichons/Smart-Motion-Detector/MobileNetSSD_deploy.prototxt.txt',
    'confidence': 0.6
}

classes_detected = []
labels_detected = []

stop_recording_after = 80
person_counter = 0
no_person_counter = 0
detections = None
last_frame = np.array(0)









# Make Thread Safe Print Function:
mylock = Lock()
p = print

def print(*a, **b):
	with mylock:
		p(*a, **b)





class detector:
    
    # Initiate Values
    def __init__(self):
        print('[INFO] initiating Detector variables')
        self.confidence = 0
        self.push_time = 0
        self.r = {'text':None}
        self.inputQueue = Queue(maxsize=1)
        self.outputQueue = Queue(maxsize=1)
        self.detections = None
        self.testing = False
        self.recording = False
        self.frame = []
        self.raw_frame = []
        self.person_counter = 0
        self.no_person_counter = 0
        self.fH = 0
        self.fW = 0
        self.img_path = ''
        
        
        
        # load API Keys
        self.keys=json.loads(open('/home/cichons/Smart-Motion-Detector/keys.json').read())
        self.pushover = self.keys['pushover']
        self.pushover['message'] = 'Person detected! (new)'
        
        self.shinobi = self.keys['shinobi']
        self.shinobi['GROUP_KEY'] = 'kcMz5HUxX4'
        self.shinobi['MONITOR_ID'] = '9zwr33ysRF'


        
        
    def set_test_mode(self):
        if self.testing:
            self.testing = False
            print("[INFO] ending test mode.")
        else:
            self.testing = True
            print("[INFO] running in test mode.")
            
        


    def init_model(self, prototxt, model_path):
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model_path)
        # initialize the list of class labels MobileNet SSD was trained to
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        # detect, then generate a set of bounding box colors for each class
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    
    def init_monitor(self,url):
        print("[INFO] starting video stream...")
        self.vs = cv2.VideoCapture(url)
        self.fps = FPS().start()
        self.ret, self.raw_frame = self.vs.read()
        print('[INFO] Stream open: ', self.vs.isOpened())
        return self.vs
    
    
      
    
    def send_push(self): #', '.join(list(set(classes))) + ' detected!'
        """
        print('[INFO] drawing boxes')
        # compute the (x, y)-coordinates
        # of the bounding box for the object
        dims = np.array([self.fW, self.fH, self.fW, self.fH])
        box = detections[0, 0, i, 3:7] * dims
        (startX, startY, endX, endY) = box.astype("int")

        # draw the prediction on the frame
        cv2.rectangle(detector.frame, (startX, startY), (endX, endY),
        detector.COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(detector.frame, label, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, detector.COLORS[idx], 2)
        """
        print('[INFO] storing boxed image to file')
        self.img_path = '/home/cichons/nvr/snapshots/' + time.strftime("%Y%m%d_%H%M%S") + '.jpg'
        cv2.imwrite(self.img_path, self.raw_frame)
        
        print('[INFO] sending Push Notification', time.strftime("%d-%m-%Y %H:%M:%S"))
        r = requests.post("https://api.pushover.net/1/messages.json", data = self.pushover,
        files = {
          "attachment": (os.path.basename(self.img_path), open(self.img_path, "rb"), "image/jpeg")
            })
        print(r.text)

        
    def classify_frame(self):
        print('[INFO] starting Classifier Process')
        # keep looping
        while True:
            """
            if self.testing:
                
                
            """    
            # check to see if there is a frame in our input queue
            if not self.inputQueue.empty():
                # grab the frame from the input queue, resize it, and
                # construct a blob from it
                self.frame = self.inputQueue.get()
                self.raw_frame = self.frame
                self.frame = cv2.resize(self.frame, (300, 300))
                (self.fH, self.fW) = self.frame.shape[:2]
                self.blob = cv2.dnn.blobFromImage(self.frame, 0.007843, (300, 300), 127.5)

                # set the blob as input to our deep learning object
                # detector and obtain the detections
                self.net.setInput(self.blob)
                self.detections = self.net.forward()

                # write the detections to the output queue
                self.outputQueue.put(self.detections)

    def start_recording(self):
        print('[INFO] starting recording')
        r = requests.get('http://192.168.1.233:8080/' + self.shinobi['key']+ '/monitor/' + self.shinobi['GROUP_KEY'] + '/' + self.shinobi['MONITOR_ID'] + '/record')
        self.recording = True
        print(r.text)
        return r 

    def stop_recording(self):
        print('[INFO] stopping recording')
        r = requests.get('http://192.168.1.233:8080/' + self.shinobi['key']+ '/monitor/' + self.shinobi['GROUP_KEY'] + '/' + self.shinobi['MONITOR_ID'] + '/start')
        self.recording = False
        print(r.text)
        return r 
    


#create detector Object:
detector = detector()



#initiate detector Model:
detector.init_model(args["prototxt"], args["model"])

vs = detector.init_monitor('rtsp://192.168.1.240:554/s1')


# start background processes
print("[INFO] starting process...")
classifier_process = Process(target=detector.classify_frame)
classifier_process.daemon = True
classifier_process.start()






print('[INFO] starting detector')


while True: #for i in range(100):
    try:
        now = datetime.datetime.now()

        # grab the frame from the threaded video stream, resize it, and
        # grab its dimensions
        ret, detector.raw_frame = vs.read()

        if ret:
            if np.array_equal(detector.raw_frame, last_frame):
                print('[INFO] Frame not new')
                continue

            no_frame_counter = 0
            if detector.testing:
                detector.frame = cv2.imread('/home/cichons/Smart-Motion-Detector/Pedestrian-Safety.jpg')
            else:
                detector.frame = detector.raw_frame
            
            

            # if the input queue *is* empty, give the current frame to
            # classify
            if detector.inputQueue.empty():
                detector.inputQueue.put(detector.frame)

            if not detector.outputQueue.empty():
                detections = detector.outputQueue.get()
                
            # check to see if our detectios are not None (and if so, we'll
            # draw the detections on the frame)
            if detections is not None:
                #print('[INFO] Checking detections')
                # reset detection lists:
                classes_detected = []
                labels_detected = []

                # loop over detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = detections[0, 0, i, 2]
                    # filter out weak detections by ensuring the `confidence`
                    # is greater than the minimum confidence
                    if confidence < args['confidence']:
                        continue

                    # extract the index of the class label from the `detections`
                    idx = int(detections[0, 0, i, 1])

                    label = "{}: {:.2f}%".format(detector.CLASSES[idx], confidence * 100)
                    #print(label)

                    labels_detected.append(label)
                    classes_detected.append(detector.CLASSES[idx])

                if 'person' not in classes_detected:
                    
                    #print('[INFO] No Person in Frame.')
                    detector.no_person_counter = detector.no_person_counter + 1
                    detector.person_counter = 0
                    
                    # if x frames without person, stop recording
                    if detector.no_person_counter == stop_recording_after:
                        print('[INFO] no person detected since ', detector.no_person_counter,' frames')
                        if detector.recording:
                            """
                            stop_recording_thread = threading.Thread(target=detector.stop_recording, args=())
                            stop_recording_thread.daemon = True  # Daemonize thread
                            stop_recording_thread.start()
                            """
                            detector.stop_recording()
                    continue

                print('[INFO] Person detected', time.strftime("%d-%m-%Y %H:%M:%S"))

                # set counters:
                detector.no_person_counter = 0
                detector.person_counter = detector.person_counter + 1

                # send Push if person_couner == 0 (New detection in this session)
                if detector.person_counter == 1 and not detector.recording:

                    # take snapshot and send push
                    """
                    send_push_thread = threading.Thread(target=detector.send_push, args=())
                    send_push_thread.daemon = True  # Daemonize thread
                    send_push_thread.start() 
                    """
                    detector.send_push()
                    # if camera is not recording, start recording:
                    print('[INFO] shinobi is recording:', detector.recording)
                    if not detector.recording:
                        """
                        start_recording_thread = threading.Thread(target=detector.start_recording, args=())
                        start_recording_thread.daemon = True # Daemonize thread
                        start_recording_thread.start() 
                        """
                        detector.start_recording()


        # If Video Stream returns no frame:
        else:
            no_frame_counter = no_frame_counter + 1
            print('[INFO] no_frame_counter: ', no_frame_counter)
            print('[INFO] Stream open?', vs.isOpened())

            if no_frame_counter >= 10 or vs.isOpened() == False:
                print('[INFO] Re-connecting Stream...'. time.strftime("%d-%m-%Y %H:%M:%S"))
                vs = cv2.VideoCapture('rtsp://192.168.1.240:554/s1')
                fps = FPS().start()
                ret, detector.raw_frame = vs.read()
                if ret:
                    no_frame_counter = 0
                continue

        last_frame = detector.raw_frame
    except KeyboardInterrupt:
        print('[INFO] exiting program', time.strftime("%d-%m-%Y %H:%M:%S"))
        if detector.recording:
            detector.stop_recording()
        sys.exit()
    except Exception as err:
        print(err)