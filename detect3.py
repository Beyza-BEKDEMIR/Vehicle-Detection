import cv2
import numpy as np

# YOLO modelinin konfigürasyon dosyası ve ağırlıklarının yolu
config_file = "yolov3.cfg"
weights_file = "yolov3.weights"
class_names_file = "coco.names" 

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)

with open(class_names_file, "r") as f:
    classes = f.read().strip().split("\n")

video_path = "video/v1.mp4"  # Video dosyasının yolu
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    input_blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(input_blob)

    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layer_names)

    conf_threshold = 0.5
    nms_threshold = 0.4
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    cv2.imshow("Vehicle Detection", frame)
    # Herhangi bir tuşa basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF != 255:
        break

cap.release()
cv2.destroyAllWindows() 
