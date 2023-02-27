import os
import cv2 as cv
import numpy as np

yolo_path = "data/YOLO"

classes = []
with open("data/YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

yolo_net = cv.dnn.readNet(os.path.join(yolo_path,"yolov3.weights"), os.path.join(yolo_path,"yolov3.cfg"))
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

image = cv.imread('Images/image1.jpeg')
height, width, channels = image.shape
img_blob = cv.dnn.blobFromImage(image, 0.00392,(416,416),(0,0,0), True, crop=False )

yolo_net.setInput(img_blob)
outputs = yolo_net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outputs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv.FONT_HERSHEY_PLAIN
thickness = 2.2
indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv.putText(image, label, (x, y + 30), font, thickness, color, 1)


cv.imwrite('Images/yolo_cv_det.png',image)
