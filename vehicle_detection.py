import cv2
import numpy as np
import os

# YOLO modelinin konfigürasyon dosyası ve ağırlıklarının yolu
config_file = "yolov3.cfg"
weights_file = "yolov3.weights"

# YOLOv3 modelini yükleme
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Sınıf adlarını yükleme
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

folder_path = "data"

image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(".jpg")]

for image_path in image_paths:
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    # Nesneleri tespit etme
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Nesneleri resim üzerine çizme
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLO Output", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
