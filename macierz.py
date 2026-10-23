import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

model = tf.keras.models.load_model('model_nowotwory.keras')

test_ds = tf.keras.utils.image_dataset_from_directory(
    'data/Testing',
    image_size=(150, 150),
    batch_size=32,
    shuffle=False 
)

class_names = test_ds.class_names

y_true = np.concatenate([y for x, y in test_ds], axis=0)
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1) 

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Przewidziana klasa (Decyzja modelu)', labelpad=20)
plt.ylabel('Rzeczywista klasa (Stan faktyczny)', labelpad=20)
plt.title('Macierz pomyłek dla rozpoznawania nowotworów mózgu')

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/macierz_pomylek.png')