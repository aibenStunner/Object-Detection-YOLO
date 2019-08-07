__author_='8.Ball'

#USAGE
#python object_detection.py --yolo YoloV3

from imutils.video import VideoStream
import numpy as np
import argparse
import time
import cv2
import os

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to the YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


#load the COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

#paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

#Load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Initialize the video stream and warm up the camera sensor
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

#loop over the video frames
while 1:
    frame = vs.read()
    #getting the spatial dimensions
    (H, W) = frame.shape[:2]

    #determine only the *output* layer names that is needed from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #construct a blob from the input frame and then perform a forward pass of the YOLO object detector, giving the bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    #show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    #initialize the lists of detected bounding boxes, confidence, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    #loop over each of the layer outputs
    for output in layerOutputs:
        #loop over each of the detections
        for detection in output:
            #extract the class ID and confidence (i.e. probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > args["confidence"]:
                #scale the bounding box coordinates back relative to the size of the image, keeping in mind the YOLO actually returns the center (x, y)-coordinates of the bounding box followerd by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                #use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                #update the list of bounding box coordinates, confidences, classIDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    #apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    #YOLO does not apply non-maxima suppression, so it is explicitly applied
    #Applying non-maxima suppression suppresses significantly overlapping bounding boxes, keeping only the most confident ones
    #NMS also ensures that we do not have any redundant or exrtraneous bounding boxes


    #ensure at least one detection exists
    if len(idxs) > 0:
        #loop over the indexes we are keepinng
        for i in idxs.flatten():
            #extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #draw a bouding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #show the output frames
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) ==  27:
        break

cv2.destroyAllWindows()
