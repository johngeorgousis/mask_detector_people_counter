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
            if pt[0] > 0.3*w:
                self.center_points[face_id][2] = True
            
            # Note if the object 
            # 1. has been at the bottom 20% of the frame and
            # 2. has been at the top 80% of the frame (can remove this condition, it is just to ensure that someone who appears from the bottom is not counted) and
            # 3. has never been counted before 
            # Then update it to having been counted and increase people count by 1 
            if (pt[0] < 0.3*w) and (pt[2] == True) and pt[3] == False:
                self.center_points[face_id][3] = True
                self.people_count += 1 
                
                # labeled count
                if pt[4] == 'Mask':
                    self.mask_count += 1
                elif pt[4] == 'No Mask':
                    self.nomask_count += 1
                    print('AY YO SOMEONE WALKED IN WITHOUT THEIR MASK ON')
                    perpetrator = frame[pt[1] - h//5:pt[1] - h//5, pt[0] - h//5:pt[0] - h//5]
                    plt.imshow(perpetrator)
                if pt[4] == 'Uncertain':
                    self.uncertain_count += 1
                    print('Someone might have no mask')
                    perpetrator = frame[pt[1] - h//5:pt[1] - h//5, pt[0] - h//5:pt[0] - h//5]
                    plt.imshow(perpetrator)
        
        return objects_bbs_ids



