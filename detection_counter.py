import numpy as np
import pandas as pd
import seaborn as sns
import imutils
from imutils.video import VideoStream
from imutils.video import FileVideoStream
import time
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

classifier = tf.keras.models.load_model('mask_classifier_mobilenet.h5')

prototxtPath = r"face_detector_ssd_caffee/deploy.prototxt" # configurations file 
weightsPath = r"face_detector_ssd_caffee/res10_300x300_ssd_iter_140000.caffemodel" # weights file
detector = cv2.dnn.readNet(prototxtPath, weightsPath) # load saved model

def detect_and_classify(frame, detector, classifier, conf = 0.3):
    '''
    Input: video frame, face detection model, face mask classification model
    Output: detector bounding boxes, classifier predictions, bounding boxes for tracker
    
    detect faces in frame and perform mask classification
    turn frame into blob (essentially preprocessing: 1. mean subtraction, 2. scaling, 3. optionally channel swapping)
    '''
    
    RGB = (104.0, 177.0, 123.0) # Source and intuition: "Deep Learning and Mean Subtraction": https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    pixels = 224
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(pixels, pixels), mean=RGB)
    
    ''' DETECTION '''
    # Detect Faces
    detector.setInput(blob)
    detections = detector.forward()
    num_of_detections = detections.shape[2]

    # initialise list of faces, their locations, list of predictions for our mask classifier
    faces = []
    locs = []
    preds = []
    
    # loop over all face detections
    for i in range(num_of_detections):
        
        # Live Counter to display on video feed
        num_of_masked = 0
        num_of_unmasked = 0
        num_of_uncertain = 0
        
        # extract the confidence (probability) in the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is  greater than the minimum confidence
        if confidence > conf:
            
            # compute the (x, y)-coordinates of the bounding box for the object 
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
    
            # ensure the bounding boxes fall within the dimensions of the frame 
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(W - 1, endX), min(H - 1, endY))

            # Extract Face 
            face = frame[startY:endY, startX:endX]                          # 1. extract the face region of interest (ROI) 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)                    # 2. convert it from BGR to RGB channel ordering 
            face = cv2.resize(face, (224, 224))                             # 3. resize it to 224x224 
            face = tf.keras.preprocessing.image.img_to_array(face)          # 4. preprocess it for the mask classifier
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

            # append face and bounding box to lists
            faces.append(face)
            locs.append(np.array([startX, startY, endX, endY]))

    ''' CLASSIFICATION '''
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference make batch predictions on *all* faces at the same time rather than one-by-one predictions in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = classifier.predict(faces, batch_size=32, verbose=0)

    return locs, preds

import math
from random import randrange

class TrackerCounter:
    
    people_count = 0
    mask_count = 0
    nomask_count = 0
    uncertain_count = 0
    
    def __init__(self):
       
        # Store information for each object in a dictionary
        # values: [cx (int), cy (int), has_been_before_boundary (boolean), has_been_counted_before (boolean), list_of_up_to_10_predictions (float [-1, 1])]
        self.center_points = {}
        
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect, label_probs, W, H, dist_same_obj=100):
            
        '''
        object_rect: List of objects' coordinates = [(xstart1, ystart1, xend1, yend1), (xstart2, ystart2, xend2, yend2), ...]
        label_probs: List of objects' label "probabilities" = [-0.9, 0.7, ....]
        W, H: frame width and height
        
        dist_same_obj:
        - It is the maximum eucledian distance from the previous object in order to be considered the same object
        - dist_same_obj determines how far the detection has to be from the old one to be considered a new object. 
        - If dist_same_obj is too low we might get false positives when an object is moving. 
        - It also depends on the amount of pixels so needs different value when resolution changes.
        '''
            
        # List of info of all objects in the frame (this is what the method returns)
        objects_infos = []
        
        # NB: If dist_same_obj is too low then fast moving faces are seen as a new face in each frame and averaging classification is not done OR even worse they are not counted at all (because a new face appears after the boundary)
        # NB: If dist_same_obj is too high then if a face disappears at frame 15 and a new face appears at frame 16 (it is only a problem if this happens in consecutive frames since memory is not implemented i.e. we got 1 frame memory) 
        # then the new face will be seen as the old face and might not be tracked
        # NB: This is very unlikely to be an issue when memory = 1 as it is now so it is a lot safer to have a high dist_same_obj than a low one. 
        dist_same_obj=(W+H)/10
        
       
        counter=0 # This variable represents how many faces of previous frame have been matched to faces of the new frame (e.g. if counter = 2 it means that 2 faces from the previous frame have been matched to 2 of the new frame). This makes sure two or more faces are not labelled the same 
        
         # Loop through all faces and their labels
        for rect, prob in zip(objects_rect, label_probs):
            xstart, ystart, xend, yend = rect
            # Get center point of new object
            cx = (xstart + xend) // 2
            cy = (ystart + yend) // 2 
            
            
            '''Find out if same face was detected'''
            same_object_detected = False
            # For all objects in previous frame 
            for face_id, pt in self.center_points.items():
                
                # calculate eucledian distance from previously detected face
                dist = math.hypot(cx - pt[0], cy - pt[1])
                
                # 1st condition: If it is the same object (ditsance < dist_same_obj) update previous objects location to this one's (keeps the object id). 
                # 2nd condition: But create a NEW object if all faces from previous frame have been matched up already (i.e. we if we've run out of faces to match). 
                # otherwise youll get the bug where when 2 people show up it is Face 1 and Face 1 as if they are the same face. So current # of objects has to be <= previous number of objects to match to previous objects which makes sense. 
                if dist < dist_same_obj and counter < len(self.center_points):
                    self.center_points[face_id][0] = cx      # update center x
                    self.center_points[face_id][1] = cy      # update center y 
                    
                    # Append classification probability to this face's classification history
                    frames_to_avg_over = 10
                    if len(pt[4]) < frames_to_avg_over:
                        self.center_points[face_id][4].append(prob)
                        
                    # If there are already 10 probabilities in the list then replace one at random (because the code for cycling through them one by one seems too complex in this case)
                    else: 
                        self.center_points[face_id][4][randrange(frames_to_avg_over)] = prob 
                    
                    # Add object info to list to return 
                    objects_infos.append([xstart, ystart, xend, yend, face_id])
                    same_object_detected = True 
                
                    counter+=1
                    
                    break
                
            '''New face detected'''
            # New object is detected: assign ID to that object
            if same_object_detected == False:
                self.id_count += 1
                self.center_points[self.id_count] = [cx, cy, False, False, [prob]]
                
                # Add new object to list to return
                objects_infos.append([xstart, ystart, xend, yend, self.id_count])
            
        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for object_info in objects_infos:
            _, _, _, _, object_id = object_info
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        
        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        
        # People counter 
        for face_id, pt in self.center_points.items():
            # Note if object has been before the boundary line
            if pt[0] > left_boundary_line:
                self.center_points[face_id][2] = True
            
            # Note if the object 
            # 1. has been detected after the boundary
            # 2. has been detected before the boundary (can remove this condition, it is just to ensure that someone who appears from the bottom is not counted) and
            # 3. has never been counted before 
            # Then update it to having been counted and increase people count by 1 
            # NB: pt[0] = xcenter, pt[1] = ycenter
            if (pt[0] < left_boundary_line) and (pt[2] == True) and pt[3] == False:
                self.center_points[face_id][3] = True
                self.people_count += 1 
                
                div = 3 # the larger div is the smaller the perpetrator frame will be 
                
                # labeled count
                avg_prob = np.mean(pt[4])

                if avg_prob > 0. + uncertain_interval:
                    self.mask_count += 1
                    
                    # TODO: Only for testing, remove 
                    # Save and display image of masked person
                    print(f'MASK ({current_time})')
                    detection_data.append(['mask', current_time])
                    # Catch exception where face is moving too fast and towards bottom left and perpetrator[0] or perpetrator[1] (width or height or both) ends up being 0.
                    # perpetrator=frame[0:H, 0:left_boundary_line + W//5]
                    # perpetrator = cv2.cvtColor(perpetrator, cv2.COLOR_BGR2RGB)
                    # perpetrator = Image.fromarray(perpetrator, 'RGB')
                    # plt.imshow(perpetrator)
                    # plt.show()
                    
                elif avg_prob < -0. - uncertain_interval:
                    self.nomask_count += 1
                    
                    # Save and display image of unmasked person
                    print(f'NO MASK ({current_time})')
                    detection_data.append(['no_mask', current_time])
                    # Catch exception where face is moving too fast and towards bottom left and perpetrator[0] or perpetrator[1] (width or height or both) ends up being 0.
                    # perpetrator=frame[0:H, 0:left_boundary_line + W//5]
                    # perpetrator = cv2.cvtColor(perpetrator, cv2.COLOR_BGR2RGB)
                    # perpetrator = Image.fromarray(perpetrator, 'RGB')
                    # plt.imshow(perpetrator)
                    # plt.show()

                else:
                    self.uncertain_count += 1

                    # Save and display image of potentially unmasked person
                    print(f'UNCERTAIN ({current_time})')
                    detection_data.append(['uncertain', current_time])
                    # Catch exception where face is moving too fast and towards bottom left and perpetrator[1] (the height of the image) ends up being 0. 
                    # perpetrator=frame[0:H, 0:left_boundary_line + W//5]
                    # perpetrator = cv2.cvtColor(perpetrator, cv2.COLOR_BGR2RGB)
                    # perpetrator = Image.fromarray(perpetrator, 'RGB')
                    # plt.imshow(perpetrator)
                    # plt.show()
                                                
        return objects_infos
    
vs = VideoStream(src=0).start()

tracker = TrackerCounter()

# Initialise total tracked faces and fps to 0 
idd = 0
fps_start_time = 0
fps = 0

# Parameters
W = 900
left_boundary_line = int(0.35*W)

# Store starting time for periodic data extraction
start_time = datetime.now()
hrs = 1 # every how many hours should the data be exported

# initialise csv file with collected data
detection_data = []

# loop over the frames from the video stream
while True:

    frame = vs.read()

    '''VISUALS'''

    # Resize the frame
    # NB: cv2.resize lets you choose height and width, imutils.resize only lets you choose width but preserves ratio
    # frame = cv2.resize(frame, dsize=(x, y))
    frame = imutils.resize(frame, width=W)
    H, W = frame.shape[:2] # used in detect_and_classify also

    # Calculate ideal font scale
    scale = 0.030                      # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
    ideal_font_size = min(W,H)/(25/scale)

    # Define box locations
    yy = int(ideal_font_size*35)
    box1start = np.array((0, 0))
    box1end = np.array((int(0.40*W), yy*2))
    box2start = np.array((int(0.40*W), 0))
    box2end = np.array((W, int(yy*4.5)))

    cv2.rectangle(frame, pt1=box1start, pt2=box1end, color=(255, 255, 255), thickness=-1)
    cv2.rectangle(frame, pt1=box2start, pt2=box2end, color=(255, 255, 255), thickness=-1)

    '''DETECT & CLASSIFY'''
    # Detect and Classify for each frame
    locs, preds= detect_and_classify(frame, detector, classifier)

    '''VISUALS'''

    label_probs = [] # This is for tracker_counter to average over multiple classifications
    # Live Counter
    num_of_masked = 0
    num_of_unmasked = 0
    num_of_uncertain = 0

    # Define classification uncertainty interval
    uncertain_interval = 0.2 # 0.5 means 50+% probability of a class for classification. 

    # loop over face locations and mask predictions
    for box, pred in zip(locs, preds):

        # unpack the bounding box and predictions
        startX, startY, endX, endY = box
        mask, no_mask = pred

        # 1. Determine the class label 2. Add colour (BGR)
        if mask >= 0.5 + uncertain_interval:
            label = 'Mask'
            colour = (0, 255, 0)
            num_of_masked +=1
            # for tracker_counter to average over 
            label_probs.append(mask)

        elif no_mask >= 0.5 + uncertain_interval:
            label = 'No Mask'
            colour = (0, 0, 255)
            num_of_unmasked +=1
            # for tracker_counter to average over 
            label_probs.append(-no_mask)

        elif (mask >= 0.5) and (mask <= 0.5 + uncertain_interval):
            label = 'Uncertain'
            colour = (0, 255, 255)
            num_of_uncertain +=1
            # for tracker_counter to average over 
            label_probs.append(mask)

        elif (no_mask >= 0.5) and (no_mask <= 0.5 + uncertain_interval):
            label = 'Uncertain'
            colour = (0, 255, 255)
            num_of_uncertain +=1
            # for tracker_counter to average over 
            label_probs.append(-no_mask)

        # probability and text to display
        probability = max(mask, no_mask) * 100
        label_text = f'{label}: {probability:.1f}%'

        # 1. Display label  
        cv2.putText(img=frame, text=label_text, org=(startX, startY - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=colour, thickness=2)
        # 2. Display bounding box
        cv2.rectangle(img=frame, pt1=(startX, startY), pt2=(endX, endY), color=colour, thickness=2)

    '''TRACK & COUNT'''

    # objects_info = [object1_info, object2_info, ...] where object_info = [Xstart, Ystart, Xend, Yend, id_of_object]
    objects_info = tracker.update(locs, label_probs, W, H, dist_same_obj=(W + H / 14))

    '''VISUALS'''  

    # for all objects
    for object_info in objects_info: 
        Xstart, Ystart, Xend, Yend, idd = object_info

        # ID of face
        cv2.putText(img=frame, text=f'Face {idd}', org=(Xstart, Ystart-40), fontScale=1.4, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(155, 149, 24), thickness=2)

    # Calculate fps
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = int(1/time_diff)
    fps = f'FPS: {fps}'
    fps_start_time = fps_end_time

    # Calculate current time and export data
    current_time = datetime.now()
    current_time_str = str(current_time)[:-7]
    current_time_export = current_time_str.replace(':', '.')
    time_difference = (current_time - start_time).seconds/3600

    if  time_difference >= hrs:

        # Export csv and summarise results
        detection_data_exp = pd.DataFrame(data=detection_data, columns=['Label', 'Datetime'])
        detection_data_exp.to_csv(f'{current_time_export}_detection_data.csv', index=False)

        # Unburden memory
        detection_data = []

        # Summarise and visualise results
        num_of_people = detection_data_exp.shape[0]
        if num_of_people > 0:
            mask = detection_data_exp[detection_data_exp.Label == 'mask'].shape[0]
            no_mask = detection_data_exp[detection_data_exp.Label == 'no_mask'].shape[0]
            uncertain = detection_data_exp[detection_data_exp.Label == 'uncertain'].shape[0]

            print(f'{num_of_people} people.')
            print(f'{mask} mask.')
            print(f'{no_mask} no_mask.')
            print(f'{uncertain} uncertain.')

            # Export Pie Chart
            dpi=110
            fig = plt.figure(figsize=(8, 6), dpi=dpi)

            if uncertain != 0:
                plt.pie(x=[mask, no_mask, uncertain], labels=['Mask', 'No Mask', 'Uncertain'], colors=['green', 'red', 'yellow'], startangle=90, autopct='%1.1f%%', textprops={'fontsize': 14})
            if uncertain == 0: 
                plt.pie(x=[mask, no_mask], labels=['Mask', 'No Mask'], colors=['green', 'red'], startangle=90, autopct='%1.1f%%', textprops={'fontsize': 14})

            plt.title(f'Total People: {num_of_people}', fontweight='bold', fontsize=15)
            plt.legend()
            #plt.show()

            fig.savefig(f'{current_time_export}_face_covering_pie_chart.png', dpi = dpi)

        else:
            print('No people detected.')

        # reset time
        start_time = datetime.now()
        
    # VISUALS: Display FPS and Current Time
    cv2.putText(img=frame, text=current_time_str, org=(box1start[0] + box1start[0]//14, yy), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 0, 0), thickness=2)
    cv2.putText(img=frame, text=fps, org=(box1start[0] + box1start[0]//14, int(yy*1.8)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 0, 0), thickness=2)

    # VISUALS: Display boundary line
    cv2.line(img=frame, pt1=(left_boundary_line, 0), pt2=(left_boundary_line, H), color=(45, 174, 102), thickness=5) 

    # VISUALS: Display live people counter 
    cv2.putText(img=frame, text=f'People Count: {tracker.people_count}', org=(box2start[0] + box2start[0]//15, yy), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 0, 0), thickness=2)
    percent_masked = 0
    percent_unmasked = 0
    percent_uncertain = 0
    if tracker.people_count != 0:
        percent_masked = np.round(tracker.mask_count / tracker.people_count * 100, 1) 
        percent_unmasked = np.round(tracker.nomask_count / tracker.people_count * 100, 1) 
        percent_uncertain = np.round(tracker.uncertain_count / tracker.people_count * 100, 1) 
    cv2.putText(img=frame, text=f'Masked:       {tracker.mask_count} ({percent_masked}%)', org=(box2start[0] + box2start[0]//15, yy*2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 155, 0), thickness=2)
    cv2.putText(img=frame, text=f'Unmasked:    {tracker.nomask_count} ({percent_unmasked}%)', org=(box2start[0] + box2start[0]//15, int(yy*3)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 0, 155), thickness=2)
    cv2.putText(img=frame, text=f'Uncertain:     {tracker.uncertain_count} ({percent_uncertain}%)', org=(box2start[0] + box2start[0]//15, int(yy*4)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 155, 155), thickness=2)

    '''END STREAM'''

    # Show the output frame in real-time
    cv2.imshow("Frame", frame)

    # Terminate if `q` is pressed. waitKey(0): keeps image still until a key is pressed. waitKey(x) it will wait x miliseconds each frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
        
        

'''EXPORT RESULTS'''

#Export csv file and summarise results
detection_data_exp = pd.DataFrame(data=detection_data, columns=['Label', 'Datetime'])
detection_data_exp.to_csv(f'{current_time_export}_detection_data.csv', index=False)

# Unburden memory
detection_data = []

# Summarise and visualise results
num_of_people = detection_data_exp.shape[0]
if num_of_people > 0:
    mask = detection_data_exp[detection_data_exp.Label == 'mask'].shape[0]
    no_mask = detection_data_exp[detection_data_exp.Label == 'no_mask'].shape[0]
    uncertain = detection_data_exp[detection_data_exp.Label == 'uncertain'].shape[0]

    print(f'{num_of_people} people.')
    print(f'{mask} mask.')
    print(f'{no_mask} no_mask.')
    print(f'{uncertain} uncertain.')

    # PIE CHART
    dpi=110
    fig = plt.figure(figsize=(8, 6), dpi=dpi)

    if uncertain != 0:
        plt.pie(x=[mask, no_mask, uncertain], labels=['Mask', 'No Mask', 'Uncertain'], colors=['green', 'red', 'yellow'], startangle=90, autopct='%1.1f%%', textprops={'fontsize': 14})
    if uncertain == 0: 
        plt.pie(x=[mask, no_mask], labels=['Mask', 'No Mask'], colors=['green', 'red'], startangle=90, autopct='%1.1f%%', textprops={'fontsize': 14})

    plt.title(f'Total People: {num_of_people}', fontweight='bold', fontsize=15)
    plt.legend()
    plt.show()

    fig.savefig(f'{current_time_export}_face_covering_pie_chart.png', dpi = dpi)

else:
    print('No people detected.')


# Cleanup
vs.stop()
cv2.destroyAllWindows()