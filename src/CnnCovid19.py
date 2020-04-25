import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# Logging something
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Declare paths
path = '../resources'
train_dir = '../resources/train'
validation_dir = '../resources/validation'
train_normal_dir = '../resources/train/normal'
train_pneumonia_dir = '../resources/train/pneumonia'
validation_normal_dir = '../resources/validation/normal'
validation_pneumonia_dir = '../resources/validation/pneumonia'

# Count how many pictures we have
num_normal_tr = len(os.listdir(train_normal_dir))
num_pneumonia_tr = len(os.listdir(train_pneumonia_dir))
num_normal_val = len(os.listdir(validation_normal_dir))
num_pneumonia_val = len(os.listdir(validation_pneumonia_dir))
total_train = num_normal_tr + num_pneumonia_tr
total_val = num_normal_val + num_pneumonia_val

print("------- MY DATASET ------- ")
print('total train normal images:', num_normal_tr)
print('total train pneumonia images:', num_pneumonia_tr)
print('total validation normal images:', num_normal_val)
print('total validation pneumonia images:', num_pneumonia_val)
print("Total train images:", total_train)
print("Total validation images:", total_val)
print("------- MY DATASET ------- ")

# Global Variables
batch_size = 10
epochs = 15 
image_height = 64
image_width = 64

####### BUILD CNN STARTS HERE #######

# Step 1
# Load images from the disk.
# Resize the images to 64x64.
# Convert them into floating point tensors.
# Rescale the tensors from values between 0 and 255 to values between 0 and 1.
train_image_generator = ImageDataGenerator(rescale=1. / 255)

validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=False,
                                                           target_size=(image_height, image_width),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(image_height, image_width),
                                                              class_mode='binary')
# Step 2
# This function will plot images in the form of a grid with
# 1 row and 5 columns where images are placed in each column.
sample_training_images, _ = next(train_data_gen)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(sample_training_images[:5])

# Step 3
# Create the Model
model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D(),  # pool size: 2x2
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)
])

# Step 4
# Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Step 5
# Print the layers
model.summary()

# Step 6
# Train the model
model_trained = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

####### BUILD CNN ENDS HERE #######

# Plot accuracy and loss
acc = model_trained.history['accuracy']
val_acc = model_trained.history['val_accuracy']
loss = model_trained.history['loss']
val_loss = model_trained.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Print Confusion Matrix and Classification Report
predictions = model.predict_generator(
    val_data_gen,
    steps=np.ceil(val_data_gen.samples / val_data_gen.batch_size),
    verbose=1,
    workers=0)

# returns 1 if prediction > 0.5 else returns 0.
y_pred = np.where(predictions > 0.5, 1, 0)

print('Confusion Matrix')
cm = metrics.confusion_matrix(val_data_gen.classes, y_pred)
print(cm)
print('Classification Report')
print(metrics.classification_report(val_data_gen.classes, y_pred))
