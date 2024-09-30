import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = 'dataset'  # Update with your dataset path

# Create a function to load images and labels
def load_data(dataset_path):
    images = []
    labels = []
    class_names = os.listdir(dataset_path)

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        for image_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, image_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), class_names

# Create the face recognition model
def create_facenet_model(input_shape=(224, 224, 3), num_classes=None):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

# Load data
images, labels, class_names = load_data(DATASET_PATH)

# Normalize images
images = images.astype('float32') / 255.0

# Create data generator for augmentation
datagen = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# Create and compile the model
model = create_facenet_model(num_classes=len(class_names))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(images, labels, batch_size=BATCH_SIZE), epochs=EPOCHS)

# Save the model
model.save('facenet_keras.h5')
print("Model saved as facenet_keras.h5")
