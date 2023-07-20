import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm
import cv2

# Cargar el archivo csv con las posiciones y nombres
posiciones_df = pd.read_csv('prueba/archive/posiciones.csv')
posiciones_dict = posiciones_df.set_index('Posicion').T.to_dict('records')[0]


def img_prepoces(ruta, tupla):
    img = cv2.imread(ruta)
    if img is None:
        print(f"Couldn't read the image at {ruta}.")
        return None
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (tupla[0], tupla[1])) / 255

train = pd.read_csv('prueba/archive/Training_set.csv')
test = pd.read_csv('prueba/archive/Testing_set.csv')

test_img = []
train_img = []
for name in tqdm(train['filename']):
    train_img.append(img_prepoces("prueba/archive/train/"+name,(224,224)))
for name in tqdm(test['filename']):
    img = img_prepoces("prueba/archive/test/"+name,(224,224))
    if img is not None:
        test_img.append(img)
names = train['label']
print("train_img",len(train_img))
print("nombres",len(names))
X_train_img, X_val_img, y_train_names, y_val_names = train_test_split(train_img, names, test_size=0.2, random_state=42)
X_train_img = np.array(X_train_img).reshape(-1, 224, 224, 3)
X_val_img = np.array(X_val_img).reshape(-1, 224, 224, 3)

modelo = keras.models.load_model('prueba/modelos/con_data/mejor_modelo.keras')
X_test_img = np.array(test_img).reshape(-1, 224, 224, 3)
predictions = modelo.predict(X_test_img)

# Convertir las predicciones a etiquetas numéricas
predicted_labels_num = [np.argmax(pred) for pred in predictions]

# Mapear las etiquetas numéricas a sus correspondientes nombres
predicted_labels = [posiciones_dict[label] if label in posiciones_dict else 'Unknown' for label in predicted_labels_num]

for i in range(5):  # Muestra 5 imágenes
    plt.imshow(X_test_img[i])
    plt.title(f"Predicted label: {predicted_labels[i]}")
    plt.show()

# Guardar las predicciones en un archivo csv
predicted_labels_df = pd.DataFrame(predicted_labels, columns=['predicted_label'])
predicted_labels_df.to_csv('predicted_labels.csv', index=False)