import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


img_height, img_width = 128, 128
batch_size = 32
train_dir = "./datasets"

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(train_dir, "train_aug"),
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(train_dir, "val"),
    image_size=(img_height, img_width),
    batch_size=batch_size
)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Rescaling(1./255))
model.add(tf.keras.applications.ResNet50(include_top=False, pooling='max', weights='imagenet'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)

EPOCHS = 50

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[earlyStopping]
)

model.summary()

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
model.evaluate(val_ds)

fig2 = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.axis(ymin=0,ymax=1)
plt.grid()
plt.title('Model Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()