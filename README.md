<a name="top"></a>

### Fashion Mnist CNN model

```python

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape
#((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))

X_train[0]
#ndarray (28, 28)

X_train.max()
#np.uint8(255)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(5, 5))
for idx in range(0, 15):
  plt.subplot(3, 5, idx + 1)
  plt.imshow(X_train[idx])
  plt.title(class_names[y_train[idx]])
  plt.axis('off')
  plt.tight_layout();

```
<img width="671" height="515" alt="image" src="https://github.com/user-attachments/assets/061064f2-e767-4c3e-a175-4c6c425cbf82" />

```python

X_train.shape
#(60000, 28, 28)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

model = tf.keras.Sequential([
    
    layers.Input(shape=(28, 28, 1)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.AvgPool2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.AvgPool2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(10, activation='softmax')

])

model.summary()

```
<img width="563" height="768" alt="image" src="https://github.com/user-attachments/assets/8bee20bf-c445-4e6c-abf0-317e6dcbc63a" />

```python

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stop], epochs=100)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}'
      f'\nLoss: {loss:.3f}')

#313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9273 - loss: 0.2169
#Accuracy: 0.93
#Loss: 0.217

y_prob = model.predict(X_test)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout();

```
<img width="1277" height="631" alt="image" src="https://github.com/user-attachments/assets/fe6be68c-8a0b-4c58-a06c-72c5d3b7411d" />

```python

plt.figure(figsize=(10, 5))

for i in range(0, 15):
  plt.subplot(3, 5, i + 1)
  plt.imshow(X_test[i])
  pred_idx = np.argmax(y_prob[i])

  color = 'green' if pred_idx == y_test[i] else 'red'

  plt.title(f'Label: {class_names[y_test[i]]}'
            f'\nPredicted: {class_names[y_test[pred_idx]]}', color=color)
  plt.tight_layout();

```

<img width="1086" height="626" alt="image" src="https://github.com/user-attachments/assets/c2c50c0e-1494-4b67-805d-eb1aab6106fe" />



[Нагору ↑](#top)
