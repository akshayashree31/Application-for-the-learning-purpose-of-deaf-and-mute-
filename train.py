import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Image size
sz = 128 

# Step 1 - Building the CNN
# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding fully connected layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=26, activation='softmax'))  # softmax for multi-class classification

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
classifier.summary()

# Step 2 - Preparing the train/test data and training the model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Normalization for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Define paths
training_path = 'data2/train'
test_path = 'data2/test'

# Check if training and test directories exist and contain images
if not os.path.exists(training_path):
    raise ValueError(f"Training path {training_path} does not exist.")
if not os.path.exists(test_path):
    raise ValueError(f"Test path {test_path} does not exist.")

# Check if the directories contain images
def check_directory(path):
    for root, dirs, files in os.walk(path):
        if len(files) == 0:
            raise ValueError(f"No images found in directory {root}")
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                raise ValueError(f"Non-image file found in directory {root}: {file}")

check_directory("E:\shitted\Source Code\data2\train")
check_directory("E:\shitted\Source Code\data2\test")

# Training set
training_set = train_datagen.flow_from_directory(
    training_path,
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical')

# Test set
test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(sz, sz),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical')

# Training the model
classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=5,
    validation_data=test_set,
    validation_steps=len(test_set))

# Saving the model
model_json = classifier.to_json()
with open("E:/shitted/Source Code/model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')

classifier.save_weights('E:/shitted/Source Code/model-bw.h5')
print('Weights saved')
