import cv2
import numpy as np
import requests
from io import BytesIO

# YOLO modelinin konfigürasyon dosyası ve ağırlıklarının yolu
config_file = "yolov3.cfg"
weights_file = "yolov3.weights"
class_names_file = "coco.names"

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)

with open(class_names_file, "r") as f:
    classes = f.read().strip().split("\n")

image_paths = [
    "https://melitime.com/files/blog/SYs3YC5vML1TuewOYUp2.jpg",
    "https://www.memurlar.net/common/news/images/759479/headline.jpg",
    "https://cdn1.ntv.com.tr/gorsel/uWIn7C2PiUeBJ6BYWGleVw.jpg?width=588&height=441&mode=crop&scale=both&v=1589456580000",
    "https://www.salomonstore.sk/wp-content/uploads/2022/11/otobus-bisiklet-aliyor-mu.jpg",
    "https://iaahbr.tmgrup.com.tr/2efe76/1200/627/0/0/800/418?u=https://iahbr.tmgrup.com.tr/2022/08/18/son-dakika-motosiklet-bisiklet-ve-skuter-suruculeri-dikkat-artik-zorunlu-oldu-1660850817754.jpg"
]

for image_path in image_paths:
    response = requests.get(image_path)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), -1)
        
    if image is None:
            print("Resim açılamadı:", image_path)
 
    height, width = image.shape[:2]
            
    input_blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(input_blob)

    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    cv2.imshow("Vehicle Detection", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()



