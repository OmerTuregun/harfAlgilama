import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Eğitim ve Doğrulama dizinleri
train_dir = 'dataset/asl_alphabet_train'
test_dir = 'dataset/asl_alphabet_test'

# Görüntü boyutları
img_width, img_height = 64, 64
batch_size = 32
epochs = 20

# Modelin yüklenmesi veya eğitimi
def load_or_train_model():
    if os.path.exists('model/hand_gesture_model.h5'):
        try:
            # Modeli yükle
            model = tf.keras.models.load_model('model/hand_gesture_model.h5')
            print("Eğitilmiş model yüklendi.")
        except OSError:
            print("Model dosyası açılamıyor. Yeni model eğitiliyor.")
            model = train_model()
    else:
        # Yeni model oluştur ve eğit
        model = train_model()

    return model

# Modeli eğit ve kaydet
def train_model():
    # Veri artırma seçenekleri
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)  # %20 doğrulama için ayrılacak

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training')  # Eğitim için

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation')  # Doğrulama için

    # Modeli oluştur
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Modeli eğit
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Modeli kaydet
    model.save('model/hand_gesture_model.h5')
    print("Yeni model eğitilip kaydedildi.")
    return model

# Modeli yükle veya eğit
model = load_or_train_model()

# Hareket sınıflarını dosya isimlerinden al
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)
class_names = sorted(train_generator.class_indices.keys())

# Görüntüyü işleme fonksiyonu
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_width, img_height))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, img_width, img_height, 1))
    return reshaped

# Yüz tespiti işlevi
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return len(faces) > 0  # Yüz tespit edilirse True, edilmezse False döner

# Hareket tespiti fonksiyonu
def detect_hand_movements(frame):
    if detect_face(frame):
        return None, None  # Yüz algılandığında hiçbir çıktı verme

    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    class_index = np.argmax(prediction)
    confidence_score = np.max(prediction)
    
    if confidence_score > 0.5:  # Güven eşiği
        return class_names[class_index], confidence_score
    else:
        return None, None

# Ana işlev
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gesture, confidence = detect_hand_movements(frame)
        if gesture:
            print(f"Hareket: {gesture}, Güven: {confidence}")
            cv2.putText(frame, f"{gesture} - {confidence:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
