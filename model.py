import numpy as np 
import zipfile as zf
import cv2
import os
import tensorflow as tf
from keras import layers, models
from keras.layers import Conv2D, MaxPooling2D, Flatten
from sklearn.preprocessing import LabelEncoder
#from object_detection.utils import config_util
#from object_detection.builders import model_builder

# Zip dosyasının yolu
file_path = 'C:\\Users\\şehitler ölmez\\Desktop\\Vehicle Detection\\dataset-vehicles.zip'
try:
    with zf.ZipFile(file_path, "r") as zip_ref:
        # Dosyayı açmak için yapılacak işlemler
        print("Dosya başarıyla açıldı.")
except FileNotFoundError:
    print("Dosya mevcut değil.")
except PermissionError:
    print("Dosyaya erişim izni yok.")

# YOLO modelinin konfigürasyon dosyası ve ağırlıklarının yolu
config_file = "yolov3_custom.cfg"

if os.path.exists(config_file):
    print("Dosya yolunda yolov3.cfg dosyası bulunuyor.")
else:
    print("Dosya yolunda yolov3.cfg dosyası bulunamadı.")
    
weights_file = "yolov3.weights"
if os.path.exists(weights_file):
    print("Dosya yolunda yolov3.weights dosyası bulunuyor.")
else:
    print("Dosya yolunda yolov3.weights dosyası bulunamadı.")
net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
# yolov3.weights dosyasını açma işlemi
try:
    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    print("dosya başarıyla açıldı.")
except cv2.error:
    print("dosya açılırken hata oluştu.")
# Eğitim ve test veri kümelerini hazırla
train_images = []
test_images = []
train_labels = []
test_labels = []

# Zip dosyasını açma
with zf.ZipFile(file_path, "r") as zip_ref:    
    file_list = zip_ref.namelist()
    for image_path in file_list:
        try:
            if image_path.startswith("dataset-vehicles/images/train") and (image_path.endswith(".jpg") or image_path.endswith(".png")):
                label_path = image_path.replace("images/train", "labels/train").replace(".jpg", ".txt").replace(".png", ".txt")

                # Resim dosyasını okuma
                with zip_ref.open(image_path) as image_file:
                    img_data = image_file.read()
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    train_images.append(img)

                # Etiket dosyasını okuma
                with zip_ref.open(label_path) as label_file:
                   label_data = label_file.read().decode("utf-8")
                   label_info = label_data.strip().split()
                
                if len(label_info) == 5:
                    # 5 değerli etiketi işleme
                    class_id, x_center, y_center, width, height = map(float, label_info)
                    label = [class_id, x_center, y_center, width, height]
                    train_labels.append(label)
                elif len(label_info) > 5:
                    # 9 değerli etiketi işleme
                    values = map(float, label_info)
                    label = list(values)
                    train_labels.append(label)
                else:
                    print("Hatalı etiket formatı(train):", label_info)
                    
            elif image_path.startswith("dataset-vehicles/images/test") and (image_path.endswith(".jpg")):
                label_path = image_path.replace("images/test", "labels/test").replace(".jpg", ".txt")

                # Resim dosyasını okuma
                with zip_ref.open(image_path) as image_file:
                    img_data = image_file.read()
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    test_images.append(img)

                # Etiket dosyasını okuma
                with zip_ref.open(label_path) as label_file:
                   label_data = label_file.read().decode("utf-8")
                   label_info = label_data.strip().split()
                
                if len(label_info) == 5:
                    # 5 değerli etiketi işleme
                    class_id, x_center, y_center, width, height = map(float, label_info)
                    label = [class_id, x_center, y_center, width, height]
                    test_labels.append(label)
                elif len(label_info) > 5:
                    # 9 değerli etiketi işleme
                    values = map(float, label_info)
                    label = list(values)
                    test_labels.append(label)
                else:
                    print("Hatalı etiket formatı(test):", label_info)

            """elif image_path.startswith("dataset-vehicles/images/val") and (image_path.endswith(".jpg") or image_path.endswith(".png")):
                label_path = image_path.replace("images/val", "labels/val").replace(".jpg", ".txt").replace(".png", ".txt")

                # Resim dosyasını okuma
                with zip_ref.open(image_path) as image_file:
                    img_data = image_file.read()
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    validation_images.append(img)

                # Resim dosyasını okuma
                with zip_ref.open(label_path) as label_file:
                   label_data = label_file.read().decode("utf-8")
                   label_info = label_data.strip().split()
                
                if len(label_info) == 5:
                    # 5 değerli etiketi işleme
                    class_id, x_center, y_center, width, height = map(float, label_info)
                    label = [class_id, x_center, y_center, width, height]
                    validation_labels.append(label)
                elif len(label_info) > 5:
                    # 9 değerli etiketi işleme
                    values = map(float, label_info)
                    label = list(values)
                    validation_labels.append(label)
                else:
                    print("Hatalı etiket formatı(validation):", label_info)"""
        except Exception as e:
            print("Hata:", e, "- Dosya:", image_path)
        
    
print("\nTraining Images Count:", len(train_images))
print("Testing Images Count:", len(test_images))
#print("Validation Images Count:", len(validation_images))

print("\nTraining Labels Count:", len(train_labels))
print("Testing Labels Count:", len(test_labels))
#print("Validation Labels Count:", len(validation_labels))

for layer in net.getLayerNames():
    layer_name = layer[0]
    if layer_name[-2:] == "-1":
        net.getLayer(layer).trainable = False

class_names = ['car', 'motorcycle', 'truck', 'bus', 'bicycle']
desired_size = (224, 224)
resized_images = []

for img in train_images:  
    img_resized = cv2.resize(img, desired_size)
    img_array = np.array(img_resized) / 255.0
    resized_images.append(img_array)

x_train = np.array(resized_images)
y_train = []

# Eğitim verilerinin etiketlerini sınıf indeksleri olarak oluşturma
for label in train_labels: 
    if len(label) == 5: 
        class_id, x_center, y_center, width, height = label
        y_train.append(class_names[int(class_id)])   # Sınıf indeksini ekleyin
    elif len(label) > 5:
        # Eğer 5'ten fazla özellik varsa, sadece sınıf indeksini alın
        class_id = label[0]
        y_train.append(class_names[int(class_id)])
    else:
        print("Hatalı etiket formatı(train):", label)

# Verileri Numpy dizilere dönüştürme
x_train = np.array(x_train)
y_train = np.array(y_train)

try:
    y_train_indices = np.array([class_names.index(label) for label in y_train])
except ValueError as e:
    print("Hata:", e)

print("X_train shape:", x_train.shape)
print("y_train_indices shape:", y_train_indices.shape)

if x_train.shape[0] == y_train_indices.shape[0]:
    print("X_train and y_train_indices have the same number of samples.")
else:
    print("X_train and y_train_indices have different number of samples.")

unique_labels = np.unique(y_train_indices)
num_unique_labels = len(unique_labels)
print("Number of unique labels:", num_unique_labels)
print("Unique labels:", unique_labels)

"""resized_validation_images = []

for img in validation_images:
    img_resized = cv2.resize(img, desired_size)
    img_array = np.array(img_resized) / 255.0
    resized_validation_images.append(img_array)

x_validation = np.array(resized_validation_images)
y_validation = []

for label in validation_labels:
    if len(label) == 5: 
       class_id, x_center, y_center, width, height = label
       y_validation.append(class_names[int(class_id)])   # Sınıf indeksini ekleyin
    elif len(label) > 5:
        class_id = label[0]
        y_validation.append(class_names[int(class_id)])
    else:
        print("Hatalı etiket formatı(validation):", label)

x_validation = np.array(x_validation)
y_validation = np.array(y_validation)  """

# Transfer learning için yeni model oluşturun
model = models.Sequential()

# Önceden eğitilmiş YOLO modelinin katmanlarını yeni modele ekleyin
for layer in net.getUnconnectedOutLayers():
    model.add(layers.Conv2D(255, (1, 1), strides=(1, 1), padding="same", use_bias=True))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Conv2D(255, (1, 1), strides=(1, 1), padding="same", use_bias=True))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Conv2D(255, (1, 1), strides=(1, 1), padding="same", use_bias=True))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))

# Yeni sınıf sayısına uygun olarak son sınıf katmanını ekleyin
num_output_classes = len(class_names)
model.add(layers.Conv2D(num_output_classes, (1, 1), strides=(1, 1), padding="same", activation="softmax"))
model.build((None, desired_size[0], desired_size[1], 3))
model.summary()
# Modeli eğitirken kullanılan sınıf indeksleri ile sınıf adlarını eşleyen bir sözlük
class_indices_to_names = {index: class_name for index, class_name in enumerate(class_names)}

# Modeli eğitim verisiyle eğitme
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_indices, batch_size=4, epochs=10, validation_split=0.2)

resized_test_images = []

for img in test_images:
    img_resized = cv2.resize(img, desired_size)
    img_array = np.array(img_resized) / 255.0
    resized_test_images.append(img_array)

x_test = np.array(resized_test_images)
y_test = []

for label in test_labels:
    if len(label) == 5: 
       class_id, x_center, y_center, width, height = label
       y_test.append(class_names[int(class_id)]) 
    elif len(label) > 5:
        class_id = label[0]
        y_test.append(class_names[int(class_id)])
    else:
        print("Hatalı etiket formatı(test):", label)
        
y_test_indices = np.array([class_names.index(label) for label in y_test])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Modeli kullanarak sınıflandırma sonuçlarını alın
predictions = model.predict(x_test)

# Her bir tahmin için en yüksek olasılığa sahip sınıfı seçin
predicted_indices = np.argmax(predictions, axis=1)

# Tahmin edilen sınıfları sınıf adlarına dönüştürün
predicted_labels = [class_indices_to_names[index] for index in predicted_indices]

# Gerçek sınıf etiketlerini sınıf adlarına dönüştürün
true_labels = [class_indices_to_names[index] for index in y_test_indices]

# Doğruluk değerlendirmesi
accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
print("Test accuracy:", accuracy)
