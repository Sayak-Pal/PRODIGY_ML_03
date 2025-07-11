# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
gti_upm_leapgestrecog_path = kagglehub.dataset_download('gti-upm/leapgestrecog')

print('Data source import complete.')

"""# Downloading the dataset in colab"""

!pip install kaggle

!mkdir ~/.kaggle

!mv kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download gti-upm/leapgestrecog

!unzip /content/leapgestrecog.zip

"""# Importing libraries"""

import numpy as np
import pandas as pd
import os
import cv2

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt

"""# Reading images"""

dir = '/content/leapGestRecog'

dir = gti_upm_leapgestrecog_path

print(gti_upm_leapgestrecog_path)

images = []
labels = []
for directory in os.listdir(dir):
  for subDir in os.listdir(os.path.join(dir,directory)):
    for img in os.listdir(os.path.join(dir, directory, subDir)):
      img_path = os.path.join(dir, directory, subDir, img)
      images.append(img_path)
      labels.append(subDir)

# images = np.array(images)
# labels = np.array(labels)
# labels

"""# Converting the data to DataFrame"""

# Create a DataFrame with images and labels
hand_gesture_data = pd.DataFrame({'Images': images, 'labels': labels})

# Filter out rows with the invalid 'leapGestRecog' label
hand_gesture_df = hand_gesture_data[hand_gesture_data['labels'] != 'leapGestRecog'].copy()

# Reconstruct the full image paths explicitly using the correct structure
hand_gesture_df['Images'] = hand_gesture_df.apply(lambda row: os.path.join(dir, row['labels'], os.path.basename(row['Images'])), axis=1)

print(hand_gesture_df.head())

"""# counting the images in each class"""

pd.Series(labels).value_counts()

"""# Splitting the dataset into train and test"""

X_train, X_test = train_test_split(hand_gesture_df, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(hand_gesture_df, test_size=0.3, random_state=42)

"""# Data Preprocessing"""

# Get the list of unique class labels after filtering
valid_classes = sorted(hand_gesture_df['labels'].unique())

image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)

train = image_gen.flow_from_dataframe(dataframe= train_set,x_col="Images",y_col="labels",
                                      target_size=(224,224), # Adjusted target_size
                                      color_mode='grayscale', # Changed color_mode
                                      classes=valid_classes, # Explicitly set classes
                                      class_mode="categorical",
                                      batch_size=4,
                                      shuffle=False
                                     )

test = image_gen.flow_from_dataframe(dataframe= X_test,x_col="Images", y_col="labels",
                                     target_size=(224,224), # Adjusted target_size
                                     color_mode='grayscale', # Changed color_mode
                                     classes=valid_classes, # Explicitly set classes
                                     class_mode="categorical",
                                     batch_size=4,
                                     shuffle= False
                                    )

val = image_gen.flow_from_dataframe(dataframe= val_set,x_col="Images", y_col="labels",
                                    target_size=(224,224), # Adjusted target_size
                                    color_mode= 'grayscale', # Changed color_mode
                                    classes=valid_classes, # Explicitly set classes
                                    class_mode="categorical",
                                    batch_size=4,
                                    shuffle=False
                                   )

classes=list(train.class_indices.keys())
print (classes)

"""# Data visualization"""

def show_hand_gesture(image_gen):
    test_dict = test.class_indices
    classes = list(test_dict.keys())
    images, labels=next(image_gen)
    plt.figure(figsize=(20,20))
    length = len(labels)
    if length<25:
        r=length
    else:
        r=25
    for i in range(r):
        plt.subplot(5,5,i+1)
        image=(images[i]+1)/2
        plt.imshow(image)
        index=np.argmax(labels[i])
        class_name=classes[index]
        plt.title(class_name, color="green",fontsize=16)
        plt.axis('off')
    plt.show()

show_hand_gesture(train)

"""# Building my model"""

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(8, 8), strides=(3, 3), activation='relu', input_shape=(224, 224, 1)),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3)),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

from keras.utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

"""# Training my mdoel"""

# Inspect the first few image paths
print("First 5 image paths in hand_gesture_df:")
print(hand_gesture_df['Images'].head())

# Check if the first image file exists
import os

first_image_path = hand_gesture_df['Images'].iloc[0]
print(f"\nChecking if the first image file exists at: {first_image_path}")
if os.path.exists(first_image_path):
    print("File exists.")
else:
    print("File does NOT exist.")

history = model.fit(train_dataset, epochs=3, validation_data=val_dataset, verbose=1)

"""# Testing my model"""

model.evaluate(test_dataset, verbose=1)

"""# Saving my model"""

model.save("hand_gesture_Model.h5")

"""# Getting results"""

pred = model.predict(test_dataset)
pred = np.argmax(pred, axis=1)

# Use the unique_labels list to create the label mapping
labels = {i: label for i, label in enumerate(unique_labels)}
pred2 = [labels[k] for k in pred]

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

# Get the true one-hot encoded labels from the test_dataset
y_true_one_hot = np.concatenate([y for x, y in test_dataset], axis=0)

# Convert the one-hot encoded labels to numerical labels
y_true_numerical = np.argmax(y_true_one_hot, axis=1)

# Use the 'labels' mapping dictionary (created in cell g4tBqb2CF8VQ) to convert numerical labels back to class names
y_true_class_names = [labels[k] for k in y_true_numerical]

print(classification_report(y_true_class_names, pred2))
print("Accuracy of the Model:","{:.1f}%".format(accuracy_score(y_true_class_names, pred2)*100))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class_labels = ['Palm', 'l', 'Fist', 'Fist_moved', 'Thumb', 'Index', 'Ok', 'Palm_moved', 'C', 'Down']

# Use the correct true labels (derived from test_dataset)
cm = confusion_matrix(y_true_class_names, pred2)

plt.figure(figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues')

# Use the unique_labels for xticks and yticks if needed, but the class_labels list should work fine
# unique_labels are ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
# class_labels maps these to descriptive names

plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=25)
plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()
