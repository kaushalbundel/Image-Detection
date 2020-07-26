# Importing Packages

import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
#construct the argument parse and parse the arguments
ap= argparse.ArgumentParser()
ap.add_argument("-i", "--image", required= True, help="path to the input image")
ap.add_argument("-p", "--prototxt", required= True, help="path to caffe prototxt file")
ap.add_argument("-m", "--model", required= True, help="path to caffe pretrained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,help="minimum probablity to filter weak detection")
args=vars(ap.parse_args())
# Initializing class labels
CLASSES=["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
# Initializing colors
COLORS=np.random.uniform(0,255, size=(len(CLASSES),3))
#Loading the model
print("[Info] model is loading..")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#Loading the input image
image = cv2.imread(args["image"])
#Creating input blob for the image
#resizing the image to 300 * 300 and then normalizing the same
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 0.007843, (300,300), 127.5)
#passing this blog through neural network and getting predictions
print("[Info] detecting objects...")
net.setInput(blob)
detections=net.forward()
#Identifying objects in the image
for i in np.arange(0, detections.shape[2]):
    #extracting the confidence associated with prediction
    confidence= detections[0,0,i,2]
    #Removing weak confidence detections ensuring that confidence is > min confidence
    if confidence > args["confidence"]:
        #extract the index from class label 
        #compute x, y coordinates for bounding of objects
        idx= int(detections[0,0,i,1])
        box=detections[0,0,i,3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY)=box.astype("int")
        #displaying predictions
        label= "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[Info] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY-15 if startY-15 >15 else startY+15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx],2)
#Showing output image
plt.matshow(image)
plt.show()
cv2.waitKey(0)