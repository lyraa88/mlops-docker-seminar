# modeling.py

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Global variables
DATA_DIR = 'data'
MODEL_PATH = 'mnist_retrained_model.h5'

def create_model():
    """Defines the CNN model architecture."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_initial_model():
    """Trains the initial model using the original MNIST dataset."""
    print("초기 MNIST 모델 학습을 시작합니다...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
    model = create_model()
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save(MODEL_PATH)
    print(f"초기 모델 학습 완료 및 {MODEL_PATH}에 저장되었습니다.")

def create_augmented_mnist_data(num_samples=10000):
    """Generates 10,000 augmented MNIST images with balanced labels."""
    (all_images, all_labels), _ = tf.keras.datasets.mnist.load_data()
    indices = []
    for i in range(10):
        label_indices = np.where(all_labels == i)[0]
        # Use replacement to ensure a large enough pool for diverse data
        indices.extend(np.random.choice(label_indices, size=num_samples // 10, replace=True))

    images = all_images[indices]
    labels = all_labels[indices]
    images = images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    augmented_images = []
    for img in images:
        if random.random() > 0.5:
            # Invert the image color
            inverted_img = 1.0 - img
            augmented_images.append(inverted_img)
        else:
            # Rotate the image by 15 degrees
            angle = random.choice([-15, 15])
            augmented_images.append(tf.keras.preprocessing.image.apply_affine_transform(
                img, theta=angle, fill_mode='constant', cval=0.0
            ))
    return np.array(augmented_images), labels

def load_retraining_data():
    """Loads and prepares data for retraining."""
    user_images, user_labels = [], []
    for label_dir in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label_dir)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = Image.open(img_path).convert('L').resize((28, 28))
                img = np.array(img).astype('float32') / 255.0
                user_images.append(img)
                user_labels.append(int(label_dir))

    user_images = np.array(user_images)
    user_labels = np.array(user_labels)

    # Load 10,000 augmented MNIST images (80% for train, 20% for test)
    mnist_retrain_images, mnist_retrain_labels = create_augmented_mnist_data(num_samples=10000)
    train_mnist_count = int(len(mnist_retrain_images) * 0.8)
    X_train_mnist = mnist_retrain_images[:train_mnist_count]
    y_train_mnist = mnist_retrain_labels[:train_mnist_count]
    X_val_mnist = mnist_retrain_images[train_mnist_count:]
    y_val_mnist = mnist_retrain_labels[train_mnist_count:]

    # Prepare user data for training and testing
    X_train_user = user_images.reshape(-1, 28, 28, 1)
    y_train_user = user_labels
    
    user_test_images = []
    user_test_labels = []
    for img, label in zip(user_images, user_labels):
        # Invert user images for the test set
        inverted_img = 1.0 - img
        user_test_images.append(inverted_img)
        user_test_labels.append(label)
    X_val_user = np.array(user_test_images).reshape(-1, 28, 28, 1)
    y_val_user = np.array(user_test_labels)

    # Combine all data
    X_train = np.concatenate([X_train_mnist, X_train_user])
    y_train = np.concatenate([y_train_mnist, y_train_user])
    X_val = np.concatenate([X_val_mnist, X_val_user])
    y_val = np.concatenate([y_val_mnist, y_val_user])
    
    return (X_train, y_train), (X_val, y_val)

def retrain_model():
    """Retrains the model with augmented and user data, including tuning."""
    print("모델 재학습을 시작합니다...")
    if not os.path.exists(MODEL_PATH):
        print(f"오류: {MODEL_PATH} 파일이 존재하지 않습니다. 초기 모델을 먼저 학습시키세요.")
        return
    model = models.load_model(MODEL_PATH)
    (X_train, y_train), (X_val, y_val) = load_retraining_data()
    
    # Model tuning with Early Stopping and data augmentation
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=10,
              validation_data=(X_val, y_val),
              callbacks=[early_stopping])
              
    model.save(MODEL_PATH)
    print(f"모델 재학습 완료 및 {MODEL_PATH}에 덮어쓰기 저장되었습니다.")

    # Reset user data
    for label_dir in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label_dir)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                os.remove(os.path.join(label_path, filename))
    print("사용자 데이터가 초기화되었습니다.")

def predict_image(image_path):
    """Predicts a digit from an input image."""
    if not os.path.exists(MODEL_PATH):
        print("오류: 모델이 학습되지 않았습니다.")
        return None, None
    model = models.load_model(MODEL_PATH)
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return predicted_label, confidence