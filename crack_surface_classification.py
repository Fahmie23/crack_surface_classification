#%%
# 1. Import Package
import cv2
import shutil
import random
import numpy as np
import os,datetime
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications,models

# %%
# 2. Load Datasets
current_dir = os.getcwd()

def load_and_display_images(folder, num_images=4):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        if len(images) >= num_images:
            break

    fig, axes = plt.subplots(2, 2, figsize=(10, 12))
    fig.suptitle("Sample Images", fontsize=16)

    for img, ax in zip(images, axes.ravel()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Shape: {img.shape}")

    plt.show()

# %%
# Define the positive crack path and display some images for surface that have crack
positive_folder = os.path.join(current_dir, 'surface_crack', 'Positive')
load_and_display_images(positive_folder, num_images=4)

# %%
# Define the negative crack path and display some images for surface that do not have crack
negative_folder = os.path.join(current_dir, 'surface_crack', 'Negative')
load_and_display_images(negative_folder, num_images=4)

#%%
# Define the combined dataset folder
combined_dataset_folder = os.path.join(current_dir, 'surface_crack', 'combined')

# %%
# Split the dataset into train and validation

# Create the combined dataset folder
os.makedirs(combined_dataset_folder, exist_ok=True)

# Define the ratio of images for the validation set (e.g., 20%)
validation_ratio = 0.2

# Create train and validation subdirectories in the combined dataset folder
train_folder = os.path.join(combined_dataset_folder, 'train')
validation_folder = os.path.join(combined_dataset_folder, 'validation')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# Function to split and copy images to train and validation directories
def split_and_copy_images(class_folder, class_name):
    images = os.listdir(class_folder)
    random.shuffle(images)
    num_validation_images = int(len(images) * validation_ratio)
    
    train_images = images[num_validation_images:]
    validation_images = images[:num_validation_images]

    for image in train_images:
        src = os.path.join(class_folder, image)
        dst = os.path.join(train_folder, class_name, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for image in validation_images:
        src = os.path.join(class_folder, image)
        dst = os.path.join(validation_folder, class_name, image)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

# Split and copy images for the "positive" class
split_and_copy_images(positive_folder, 'positive')

# Split and copy images for the "negative" class
split_and_copy_images(negative_folder, 'negative')


# %%
train_dir = os.path.join(current_dir, 'surface_crack', 'combined', 'train')
validation_dir = os.path.join(current_dir, 'surface_crack', 'combined', 'validation')
BATCH_SIZE = 64
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)

# %%
# 3. Inspect some data examples after combining the datasets.
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#%%
#4. Further split validation and test dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

#%%
#5. Converting the tensorflow datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
#6. Create a Sequential 'model' for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
#7. Repeatedly apply data augmentation on a single image
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

#%%
#8. Define a layer for data normalization/rescaling
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#%%
"""
The plan:

data augmentation > preprocess input > transfer learning model
"""

IMG_SHAPE = IMG_SIZE + (3,)
#(A) Load the pretrained model using keras.applications module
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
#Display summary of the model
base_model.summary()
#Display model structure
keras.utils.plot_model(base_model)

#%%
#(B) Freeze the entire feature extractor
base_model.trainable = False

# %%
#Display the model summary to show that most parameters are non-trainable
base_model.summary()

# %%
#(C) Create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
#(D) Create the output layer
output_layer = layers.Dense(len(class_names),activation='softmax')
#(E) Build the entire pipeline using functional API
#a. Input
inputs = keras.Input(shape=IMG_SHAPE)
#b. Data augmentation model
x = data_augmentation(inputs)
#c. Data rescaling layer
x = preprocess_input(x)
#d. Transfer learning feature extractor
x = base_model(x,training=False)
#e. Final extracted features
x = global_avg(x)
#f. Classification layer
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)
#g. Build the model
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

# %%
#10. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# %%
#Create TensorBoard callback object
base_log_path = r"tensorboard_logs\image_crack_classification"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)

# %%
#11. Model training
#Evaluate the model before training
loss0,acc0 = model.evaluate(test_dataset)
print("Evaluation before training:")
print("Loss = ", loss0)
print("Accuracy = ",acc0)

# %%
#12. Proceed with model training
early_stopping = callbacks.EarlyStopping(patience=2)
EPOCHS = 10
history = model.fit(train_dataset,validation_data=validation_dataset,epochs=EPOCHS,callbacks=[tb,early_stopping])

#%%
# plotting for loss and validation loss
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

#%%
# plotting for accuracy and validation accuracy
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuarcy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# %%
#13. Further fine tune the model
# Let's take a look to see how many layers are in the base model
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# %%
# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# %%
#14. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

# %%
#15. Model training
fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch
history_fine = model.fit(train_dataset,validation_data=validation_dataset,epochs=total_epoch,initial_epoch=history.epoch[-1],callbacks=[tb,early_stopping])

# %%
#Evaluate the model after training
loss1,acc1 = model.evaluate(test_dataset)
print("Evaluation After Training:")
print("Loss = ",loss1)
print("Accuracy = ",acc1)

# %%
# 16. Saving and loading the model
# Save the model
model.save('crack_classification.h5')

# %%
