<a name="top"></a>

### Mnist dataset CNN

```python

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape
#((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))

plt.figure(figsize=(5, 2))
for idx in range(0, 15):
  plt.subplot(3, 5, idx + 1)
  plt.imshow(X_train[idx])
  plt.axis('off')
  plt.title(y_train[idx])
  plt.tight_layout();

```
<img width="483" height="261" alt="image" src="https://github.com/user-attachments/assets/df00b64e-05b8-494e-b579-85c36d47a3c9" />

```python

X_train[0].max()
#np.uint8(255)

X_train[0]
#ndarray (28, 28)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_train.shape, X_test.shape
#((60000, 28, 28, 1), (10000, 28, 28, 1))

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
<img width="573" height="813" alt="image" src="https://github.com/user-attachments/assets/846e2dc0-b377-48b5-9b42-c31b2267f3c3" />

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
    restore_best_weights=True,
)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stop], epochs=150)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f},'
      f'\nLoss: {loss:.3f}.')

#313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9945 - loss: 0.0170
#Accuracy: 0.99,
#Loss: 0.017.


plt.figure(figsize=(10, 4))

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
plt.legend();

```
<img width="1092" height="478" alt="image" src="https://github.com/user-attachments/assets/46dc098d-76b9-40be-ae90-2a48dd4efbb6" />

```python

y_prob = model.predict(X_test)
plt.figure(figsize=(10, 4))
for idx in range(0, 15):
  plt.subplot(3, 5, idx + 1)
  plt.imshow(X_test[idx])
  index_pred = np.argmax(y_prob[idx])
  color = 'green' if index_pred == y_test[idx] else 'red'
  
  plt.title(f'True label: {y_test[idx]}'
            f'\nPredicted label: {index_pred}', color=color)

  plt.axis('off')
  plt.tight_layout();

```

<img width="1037" height="483" alt="image" src="https://github.com/user-attachments/assets/2c7c37bc-9acc-4b1f-9a3f-bd9197c7616c" />

[Нагору ↑](#top)
