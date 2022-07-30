import math
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import imutils
import time
import cv2
import os
import tensorflow as tf
import time

    
classifier = tf.keras.models.load_model('classifier_20epoch.h5')

import math


class TrackerCounter:
    
    people_count = 0
    mask_count = 0
    nomask_count = 0
    uncertain_count = 0
    
    def __init__(self):
       
        # Store the center positions of the objects
        # Form -->      id: [cx, cy, has_been_top_80%, has_been_tracked_at_bottom_20%, most_recent_label]
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect, labels, w, h, sensitivity=100):
        
        # TODO: change name: sensitivity. Should be the opposite bc the higher the sensitivity the less sensitive it is to false positives.
        # TODO: change sensitivity to s = (w + h) / 14 to be a function of the width / height of the frame (once you access them)
    
        '''
        object_rect: List of object coordinates in the form [(xstart1, ystart1, xend1, yend1), (xstart2, ystart2, xend2, yend2), ...]
    
        
        Sensitivity: Added by me
        - Sensitivity determines how far the detection has to be from the old one to be considered a new object. 
        - If sensitivity is too low we might get false positives when an object is moving. 
        - It also depends on the amount of pixels so needs different value when resolution changes.
        '''
            
        # Objects boxes and ids
        objects_bbs_ids = []
        

        # Get center point of new object
        for rect, label in zip(objects_rect, labels):
            xstart, ystart, xend, yend = rect
            cx = (xstart + xend) // 2
            cy = (ystart + yend) // 2 
            
            
 
            # Find out if that object has been detected already
            same_object_detected = False
            for face_id, pt in self.center_points.items():
                # find hypotenuse form
                dist = math.hypot(cx - pt[0], cy - pt[1])
                
                # if euclidean distance is lower than sensitivity it keeps the same object id 
                if dist < sensitivity:
                    self.center_points[face_id][0] = cx      # update center x
                    self.center_points[face_id][1] = cy      # update center y 
                    self.center_points[face_id][4] = label   # update to latest label

                    objects_bbs_ids.append([xstart, ystart, xend, yend, face_id])
                    same_object_detected = True 
                
                    break
                
            
          
            
            # New object is detected we assign the ID to that object
            if same_object_detected == False:
                self.id_count += 1
                self.center_points[self.id_count] = [cx, cy, False, False, label]
                
                
                
                # Cleanup ting
                objects_bbs_ids.append([xstart, ystart, xend, yend, self.id_count])
                
                    
        # Clean the dictionary by center points to remove IDS not used anymore
        # TODO: consider running this every 5 frames to fix disappearing face = new face problem.
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        
        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        
        
        
        # People counter 
        for face_id, pt in self.center_points.items():
            # Note if object has been in the top 80% of the frame
            if pt[1] < 0.8*h:
                self.center_points[face_id][2] = True
            
            # Note if the object 
            # 1. has been at the bottom 20% of the frame and
            # 2. has been at the top 80% of the frame (can remove this condition, it is just to ensure that someone who appears from the bottom is not counted) and
            # 3. has never been counted before 
            # Then update it to having been counted and increase people count by 1 
            if (pt[1] > 0.8*h) and (pt[2] == True) and pt[3] == False:
                self.center_points[face_id][3] = True
                self.people_count += 1 
                
                # labeled count
                if pt[4] == 'Mask':
                    self.mask_count += 1
                elif pt[4] == 'No Mask':
                    self.nomask_count += 1
                    print('AY YO SOMEONE WALKED IN WITHOUT THEIR MASK ON')
                if pt[4] == 'Uncertain':
                    self.uncertain_count += 1
        
        return objects_bbs_ids



def detect_and_classify(frame, detector, classifier):
    # detect faces in frame and perform mask classification
    # TODO: Change RGB Tuple
    # turn frame into blob (essentially preprocessing: 1. mean subtraction, 2. scaling, 3. optionally channel swapping)
    RGB = (104.0, 177.0, 123.0) # Consider changing. Read the following under "Deep Learning and Mean Subtraction for intuition: https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    pixels = 224
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(pixels, pixels), mean=RGB)
    
    '''
    DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION 
    DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION 
    DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION DETECTION 
    '''
    
    '''
    TODO: Understand
    pass the blob through the network and obtain the face detections in the form of a blob. Current shape: (1, 1, 200, 7).
    1st dim: ?
    2nd dim: ?
    3rd dim: number of detections
    4th dim: (?, ?, confidence, startX %, startY %, endX %, endY %)
    The output is a 200x7 "array of heatmaps", the probability of a face being in location x, y. 
    I think 200 (the third value) is the number of face detections. I.e. 200 faces where detected each with a different probability of being a face. 
    '''
    detector.setInput(blob)
    detections = detector.forward()
    num_of_detections = detections.shape[2]

    # initialise list of faces, their locations, list of predictions for our mask classifier, rectangles list for tracking
    faces = []
    locs = []
    preds = []
    rects = []
    
    # loop over all face detections
    for i in range(num_of_detections):
        
        # Live Counter
        num_of_masked = 0
        num_of_unmasked = 0
        num_of_uncertain = 0
        
        # extract the confidence (probability) in the detection
        # The 1st and 2nd seem to be either 0 or 1. 
        # The last 4 are boundaries of box respectively: startX %, startY %, endX %, endY %. For example 30% of width to 60% of width and 10% of height to 40% of height of image. 
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is  greater than the minimum confidence
        if confidence > 0.5:
            
            # compute the (x, y)-coordinates of the bounding box for the object 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            rects.append(box.astype('int'))
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame 
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract Face 
            face = frame[startY:endY, startX:endX]                          # 1. extract the face region of interest (ROI) 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)                    # 2. convert it from BGR to RGB channel ordering 
            face = cv2.resize(face, (224, 224))                             # 3. resize it to 224x224 
            face = tf.keras.preprocessing.image.img_to_array(face)          # 4. preprocess it for the mask classifier
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

            # append face and bounding box to lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference make batch predictions on *all* faces at the same time rather than one-by-one predictions in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = classifier.predict(faces, batch_size=32)
    else: 
        #3. Display Information
        x = 10
        y = 15
        cv2.rectangle(frame, pt1=(box1x, 0), pt2=(box2x, y+30), color=(255, 255, 255), thickness=-1)
        cv2.rectangle(frame, pt1=(box2x, 0), pt2=(box3x, y+30), color=(150, 150, 150), thickness=-1)
        cv2.rectangle(frame, pt1=(box3x, 0), pt2=(int(1.0*w), y + 80), color=(255, 255, 255), thickness=-1)        
        cv2.putText(img=frame, text='Masked | Unmasked | Uncertain', org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
        cv2.putText(img=frame, text='   0          0           0', org=(x, y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)

    return locs, preds, rects




# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt" # configurations file 
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel" # weights file
detector = cv2.dnn.readNet(prototxtPath, weightsPath) # load saved model

# initialise the video stream
from imutils.video import VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0) # Give camera 2 seconds to warm up before running the loop

# Instantiate tracker / counter
tracker=TrackerCounter()

# Initialise total tracked faces to 0 
idd=0 

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    # NB: cv2.resize lets you choose height and width, imutils.resize only width but preserves ratio
    frame = vs.read()
    #frame = cv2.resize(frame, dsize=(x, y)) 
    frame = imutils.resize(frame, width=900)
    h, w = frame.shape[:2] # used in detect_and_classify also
    
    # For visualisation later
    box1x = 0
    box2x = int(0.35*w)
    box3x = int(0.80*w)
    
    # Detect and Classify for each frame
    locs, preds, rects = detect_and_classify(frame, detector, classifier)

    '''
    VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS 
    VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS VISUALS 
    '''
    
    labels = []
    # Live Counter
    num_of_masked = 0
    num_of_unmasked = 0
    num_of_uncertain = 0
    
    # loop over face locations and mask predictions
    for box, pred in zip(locs, preds):
        
        # unpack the bounding box and predictions
        startX, startY, endX, endY = box
        mask, no_mask = pred

        # 1. Determine the class label 2. Add colour (BGR)
        if mask >= 0.7:
            label = 'Mask'
            colour = (0, 255, 0)
            num_of_masked +=1
        elif no_mask >= 0.7:
            label = 'No Mask'
            colour = (0, 0, 255)
            num_of_unmasked +=1
        else: 
            label = 'Uncertain'
            colour = (0, 255, 255)
            num_of_uncertain +=1
        
        probability = max(mask, no_mask) * 100
        label_text = f'{label}: {probability:.1f}%'
        
        # For tracker statistics
        labels.append(label)
        
        # 1. Display label  
        cv2.putText(img=frame, text=label_text, org=(startX, startY - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=colour, thickness=2)
        # 2. Display bounding boxQQQQQQ
        cv2.rectangle(img=frame, pt1=(startX, startY), pt2=(endX, endY), color=colour, thickness=2)
        #3. Display Information
        x = 10
        y = 15
        # Background boxes 1, 2, 3 TODO: consider border instead of diff colours 
        cv2.rectangle(frame, pt1=(box1x, 0), pt2=(box2x, y+30), color=(255, 255, 255), thickness=-1)
        cv2.rectangle(frame, pt1=(box2x, 0), pt2=(box3x, y+30), color=(150, 150, 150), thickness=-1)
        cv2.rectangle(frame, pt1=(box3x, 0), pt2=(int(1.0*w), y + 80), color=(255, 255, 255), thickness=-1)
        
        # Live Info
        cv2.putText(img=frame, text='Masked | Unmasked | Uncertain', org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
        cv2.putText(img=frame, text=f'   {num_of_masked}          {num_of_unmasked}           {num_of_uncertain}', org=(x, y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
        

    '''
    TRACKING / COUNTING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING 
    TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING 
    TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING TRACKING 
    '''
    
    
    # YOUTUBE https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/
    # boxes_ids contains [object1_info, object2_info, ...] where object1_info = [Xstart, Ystart, Xend, Yend, id_of_object]
    boxes_ids = tracker.update(rects, labels, w, h, sensitivity=(w + h / 14))
        
    # for all objects
    for box_id in boxes_ids: 
        Xstart, Ystart, Xend, Yend, idd = box_id
        
        # VISUALS        
        # Text
        # ID of face 
        cv2.putText(img=frame, text=f'Face {idd}', org=(Xstart, Ystart-40), fontScale=1.4, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(155, 149, 24), thickness=2)
    
    
    # 
    cv2.putText(img=frame, text=f'Total Faces (since start): {idd}', org=(box2x + 10, y+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
    # Draw Line of people count
    cv2.line(img=frame, pt1=(0, int(0.8*h)), pt2=(w, int(0.8*h)), color=(45, 174, 102), thickness=5) 
    # Live Counter
    cv2.putText(img=frame, text=f'People Count: {tracker.people_count}', org=(box3x + 10, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
    cv2.putText(img=frame, text=f'Masked: {tracker.mask_count}', org=(box3x + 10, y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
    cv2.putText(img=frame, text=f'Unmasked: {tracker.nomask_count}', org=(box3x + 10, y+40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
    cv2.putText(img=frame, text=f'Uncertain: {tracker.uncertain_count}', org=(box3x + 10, y+60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)

    '''
    END
    '''
    
    # Show the output frame in real-time
    cv2.imshow("Frame", frame)

    # Terminate if `q` is pressed. waitKey(0): keeps image still until a key is pressed. waitKey(x) it will wait x miliseconds each frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# TODO: run classifier multiple times on the same tracked face to average out results. More robust classification. 
# TODO: Alert when someone goes in with no mask or uncertain and save and display their face over the video (perhaps on the right hand side) for security to go get them
# TODO: export all collected data to csv here? 
# TODO: also add date and time for maybe all the detections or for when the programme ran 
        
# # Cleanup
cv2.destroyAllWindows()
vs.stop()