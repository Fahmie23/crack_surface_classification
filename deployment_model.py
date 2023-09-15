#%%
# 1. Import packages
import numpy as np
import os,datetime
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications,models

#%%
# 2. Split the data into train and validation
current_dir = os.getcwd()
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
new_model = keras.models.load_model('crack_classification.h5')
new_model.evaluate

# %%
#6. Deployment
#(A) Retrieve a batch of images from the test set and perform predictions
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = new_model.predict_on_batch(image_batch)

# %%
prediction_index = np.argmax(predictions,axis=1)
#(B) Create a label map for the classes
label_map = {i:names for i,names in enumerate(class_names)}
prediction_label = [label_map[i] for i in prediction_index]
label_class_list = [label_map[i] for i in label_batch]

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Adjust spacing between subplots

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(f"Label: {label_class_list[i]}\nPrediction: {prediction_label[i]}")  # Use \n for new line
    plt.axis('off')
    plt.grid(False)

plt.show()
