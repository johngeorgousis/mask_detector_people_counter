import cv2
import numpy as np
import tensorflow as tf


def detect_and_classify(frame, detector, classifier, W, H, conf = 0.3):
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
