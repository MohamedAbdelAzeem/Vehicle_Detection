#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import time
import cv2
import os
import glob
import matplotlib.pyplot as plt
from moviepy.editor import *

input_path= sys.argv[1]
output_path = sys.argv[2]
# ### Load Weights and CFG

# In[11]:


#weights_path = os.path.join("yolo","C:\\Users\\THE WHALE\\Vechile_Detection_YOLO_GPU\\yolov3.weights")
#cfg_path = os.path.join("yolo","C:\\Users\\THE WHALE\\Vechile_Detection_YOLO_GPU\\yolov3.cfg")
labels = open('coco.names').read().strip().split('\n')     #Load COCO Dataset
print(labels)


# ### Load Neural Netowrk

# In[12]:


import cv2
net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')   #Load Neural Netowrk


# In[6]:


##Getting Layer Names and Output Layers
names = net.getLayerNames()
outputlayers = list(net.getUnconnectedOutLayersNames())  
print(outputlayers)


# ### Inference

# In[7]:


def process(image):
    boxes = []
    confidences = []
    classIDs = []

    H,W = image.shape[:2]   #height and width of the image
    blob = cv2.dnn.blobFromImage(image,1/255.0,(416,416),crop=False,swapRB=False)  #preprocessing the image
    net.setInput(blob)  #input the image to the network
    layers_output = net.forward(outputlayers)
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)   #getting ID of the class that has maximum score
            confidence = scores[classID]
            if confidence > 0.75: 
                box = detection[:4] * np.array([W,H,W,H])    #Getting center,width and height of the box
                bx,by,bw,bh = box.astype("int")
                x = int(bx-(bw/2))
                y = int(by-(bh/2))
                
                boxes.append([x,y,int(bw),int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.75,0.6)  #Filtering boxes using non maximum suppression
    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y = [boxes[i][0],boxes[i][1]]         # top left corner point 
            w,h = [boxes[i][2],boxes[i][3]]         #width and height of the box
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,139,139),2)
            cv2.putText(image,"{}:{}".format(labels[classIDs[i]],confidences[i]),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,139,139),2)
    return image


# In[9]:


clip = VideoFileClip(input_path)
start = time.time()
final = clip.fl_image(process).subclip(0,30)
#final.ipython_display()
#print(time.time()-start)
final.write_videofile(output_path, audio=False)


# In[ ]:





# In[ ]:




