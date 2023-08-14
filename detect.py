import cv2
import numpy as np

# YOLO modelinin konfigürasyon dosyası ve ağırlıklarının yolu
config_file = "yolov3.cfg"
weights_file = "yolov3.weights"
class_names_file = "coco.names" 

# YOLO modelini yükleyin
net = cv2.dnn.readNetFromDarknet(config_file, weights_file)

# yolov3.weights dosyasını açma işlemi
try:
    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    print("yolov3.weights dosyası başarıyla açıldı.")
except cv2.error:
    print("yolov3.weights dosyası açılırken hata oluştu.")

# Sınıf adlarını okuma
with open(class_names_file, "r") as f:
    classes = f.read().strip().split("\n")

# Resim yolları
image_paths = [
    "traffic.jpg",
    "images\image275.jpg",
    "images\\54.jpg",
    "images\img1048.jpg",
    "images\imtest1.jpeg",
    "images\imtest8.jpeg",
    "images\imtest9.jpeg",
    "images\imtest10.jpeg",
    "images\imtest17.png"
]

for image_file in image_paths:
    
    image = cv2.imread(image_file)
    
    if image is None:
        print("Resim açılamadı:", image_file)
        continue
    
    # Resmin boyutları
    height, width = image.shape[:2]

    input_blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # YOLO modelini input_blob ile çalıştırma
    net.setInput(input_blob)

    # YOLO çıktısını alma
    layer_names = net.getLayerNames()
    output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layer_names)

    # Tespit edilen nesneleri listeye ekleme
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

    # Tespit edilen araçları çizdirme
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sonucu gösterme
    cv2.imshow("Vehicle Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
