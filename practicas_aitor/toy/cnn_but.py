#Importar librerias"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from collections import Counter
from tqdm import tqdm

"""Para este cuaderno haremos uso de un [dataset](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)obtenido de kaggle"""

"""Paso 3: Cargar datos de entrenamiento"""
#Verificamos si se tiene GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

train = pd.read_csv('prueba/archive/Training_set.csv')
test = pd.read_csv('prueba/archive/Testing_set.csv')

# Crear un diccionario donde las claves son las clases  y los valores son 0
nombres = sorted(train['label'].unique())
counter = Counter(train['label'])

assert sum(counter.values()) == len(train['label']), "La suma no está bien hecha."

"""Graficamos el dataset"""

plt.figure(figsize=(20, 10))
plt.bar(counter.keys(), counter.values())
plt.title('Distribución de clases en el conjunto de datos')
plt.xlabel('Clases')
plt.ylabel('Cantidad')
plt.xticks(rotation=90) # Esto rota los nombres de las clases en el eje x para mejorar la visibilidad
plt.show()
def calcular_porcentajes(counter):
    total = sum(counter.values())
    return {nombre: (valor / total) * 100 for nombre, valor in counter.items()}

def imprimir_porcentajes(porcentajes):
    for nombre, valor in porcentajes.items():
        print(f'{nombre}: {valor:.2f}%')

"""Imprimos el porcentaje, es decir que porcentaje representa cada clase del dataset"""
porcentajes = calcular_porcentajes(counter)
imprimir_porcentajes(porcentajes)

"""Empezamos con la CNN"""

def img_prepoces(ruta, tupla):
    """
    Función para leer, convertir y redimensionar una imagen utilizando OpenCV.

    Parámetros:
    - ruta (str): Una cadena que representa la ruta del archivo de imagen a procesar.
    - tupla (tuple): Un tuple que contiene dos elementos que representan la altura y el ancho (respectivamente) a los que se debe redimensionar la imagen.

    Devoluciones:
    - ndarray: Una matriz Numpy que representa la imagen procesada y normalizada.

    Esta función realiza los siguientes pasos:
    1. Lee la imagen de la ruta especificada usando `cv2.imread()`.
    2. Convierte la imagen de BGR a RGB usando `cv2.cvtColor()`.
    3. Redimensiona la imagen al tamaño especificado por la tupla usando `cv2.resize()`.
    4. Normaliza la imagen dividiéndola por 255. Esto se hace para cambiar los valores de los píxeles de la imagen de 0-255 a 0-1, un rango más adecuado para el entrenamiento de modelos de red neuronal.
    """
    img = cv2.imread(ruta)
    if img is None:
        print(f"Couldn't read the image at {ruta}.")
        return None
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (tupla[0], tupla[1])) / 255



def construir_modelo_fc(forma_entrada, func_acti, num_capas, filtro_inicial, n_clases, n_neuronas_fc):
    """
    Función para construir una red neuronal convolucional (CNN) con Keras.

    Parámetros:
    - forma_entrada (tuple): Un tuple que define la forma de los datos de entrada (altura, ancho, canales).
    - func_acti (str): Una cadena que define la función de activación a utilizar en las capas convolucionales y densas. Ejemplos: 'relu', 'sigmoid', 'tanh'.
    - num_capas (int): Un entero que define el número de capas convolucionales a añadir a la red.
    - filtro_inicial (int): Un entero que define el número de filtros en la primera capa convolucional. Las capas convolucionales subsiguientes duplicarán el número de filtros.
    - n_clases (int): Un entero que define el número de clases objetivo para la clasificación.
    - n_neuronas_fc (int): Un entero que define el número de neuronas en la capa totalmente conectada (capa densa).

    Devoluciones:
    - model (keras.Model): El modelo de la red neuronal convolucional.
    """
    model = Sequential()
    model.add(Conv2D(filtro_inicial, (3, 3), activation=func_acti, input_shape=forma_entrada))
    model.add(MaxPooling2D((2, 2)))
    for i in range(1, num_capas):
        model.add(Conv2D(filtro_inicial*(2**i), (3, 3), activation=func_acti))
        model.add(MaxPooling2D((2, 2)))
    #Añadir capas densas
    model.add(Flatten())#importante esta capa ya que convierte 2d a 1d para capas densas
    model.add(Dense(n_neuronas_fc, activation=func_acti))
    model.add(Dropout(0.5))
    model.add(Dense(n_clases, activation='softmax'))

    return model



def convertir_labels_one_hot(ytrain, yval):
    """
    Esta función convierte las etiquetas categóricas de texto a formato one-hot.

    Argumentos:
        ytrain: Serie de pandas o array de NumPy que contiene las etiquetas de las muestras de entrenamiento.
        yval: Serie de pandas o array de NumPy que contiene las etiquetas de las muestras de validación.

    Retorna:
        Dos arrays de NumPy que representan las etiquetas de las muestras de entrenamiento y validación,
        respectivamente, en formato one-hot.
    """

    # Crear el codificador
    le = LabelEncoder()

    # Ajustar el codificador a las etiquetas de entrenamiento y transformar las etiquetas a números enteros
    y_train_int = le.fit_transform(ytrain)

    # Utilizar el codificador ajustado para transformar las etiquetas de validación a números enteros
    y_val_int = le.transform(yval)

    # Convertir los números enteros a formato one-hot y devolverlos
    return to_categorical(y_train_int), to_categorical(y_val_int)

def dimensiones(ruta):
    """
    Función para obtener las dimensiones de una imagen utilizando OpenCV.

    Parámetros:
    - ruta (str): Una cadena que representa la ruta del archivo de imagen a procesar.

    Devoluciones:
    - tuple: Un tuple que contiene las dimensiones de la imagen. El formato de la tupla devuelta es (altura, ancho, canales), donde 'canales' generalmente es 3 para imágenes en color (RGB).

    Esta función realiza los siguientes pasos:
    1. Lee la imagen de la ruta especificada usando `cv2.imread()`.
    2. Convierte la imagen de BGR a RGB usando `cv2.cvtColor()`.
    3. Devuelve la forma de la matriz de la imagen usando el atributo `.shape` de numpy, que representa las dimensiones de la imagen.
    """
    return cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB).shape

def crear_diccionario_posiciones(lista, serie):
        diccionario = {}
        for i in range(len(lista)):
            if i < len(serie):
                valor = serie.iloc[i]
                if valor not in diccionario:
                    indice = np.where(lista[i] == 1.0)[0][0]
                    diccionario[valor] = indice

        return diccionario

def guardar_diccionario_en_csv(diccionario, nombre_archivo):
        with open(nombre_archivo, 'w', newline='') as archivo:
            writer = csv.writer(archivo)
            writer.writerow(['Nombre', 'Posicion'])
            for clave, valor in diccionario.items():
                writer.writerow([clave, valor])


def procesar_datos(y_train_names, y_temp):

    if len(y_train_names) == len(y_temp):
        diccionario_posiciones = crear_diccionario_posiciones(y_train_names, y_temp)
        guardar_diccionario_en_csv(diccionario_posiciones, "posiciones.csv")
    else:
        print("La longitud de la lista y el Series no coincide.")



"""1-Dividir el dataset en train y validation"""

#crear el array donde guardaremos las imagenes en forma de matrix
train_img = []

"""2-Convertir las imagenes a una matrix"""

for name in tqdm(train['filename']):
    train_img.append(img_prepoces("prueba/archive/train/"+name,(224,224)))


"""3-Guardar las etiquetas"""

names = train['label']

"""4-Dividir los datos"""

X_train_img, X_val_img, y_train_names, y_val_names = train_test_split(train_img, names, test_size=0.2, random_state=42)

#X_train_img, X_val_img, y_train_names, y_val_names = train_test_split(train_img, names, test_size=0.2, random_state=42, stratify=names)

"""Resize:
 se espera que las imagenes de entrada esten en un array 4d, con forma (numero de imagenes,altura,ancho,canales)
"""

X_train_img = np.array(X_train_img).reshape(-1, 224, 224, 3)
X_val_img = np.array(X_val_img).reshape(-1, 224, 224, 3)

"""Definimos los parametros para hacer uso del data augmentation"""

# Crear un generador de datos de imagen con aumentación
datagen = ImageDataGenerator(
        rotation_range=30,  # Rotar aleatoriamente las imágenes
        zoom_range = 0.1, # Zoom aleatoriamente las imágenes dentro del rango
        width_shift_range=0.1,  # Desplazar aleatoriamente las imágenes horizontalmente
        height_shift_range=0.1,  # Desplazar aleatoriamente las imágenes verticalmente
        horizontal_flip=True,  # Invertir aleatoriamente las imágenes horizontalmente
        vertical_flip=False
        )  # No invertir las imágenes verticalmente

"""Ajustamos el generador de datos a los datos de entrenamiento"""

datagen.fit(X_train_img)

"""Convertimos las etiquetas a formato one-hot"""
#print("ANTES: ",y_train_names)


y_temp = y_train_names.copy()

y_train_names, y_val_names = convertir_labels_one_hot(y_train_names, y_val_names)


procesar_datos(y_train_names, y_temp)


"""Callbacks"""

callbacks = [
    ModelCheckpoint("mejor_model_pre.h5", save_best_only=True,verbose=1),  # Guarda el mejor modelo como 'mejor_modelo_pre.h5'
    EarlyStopping(patience=10, restore_best_weights=True),  # Detén el entrenamiento si el modelo deja de mejorar
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)  # Reduce el learning rate si la pérdida de validación no mejora
]


"""Creando el modelo"""

modelo = construir_modelo_fc(
    forma_entrada=(224,224,3), # Forma de entrada
    func_acti='relu', # Función de activación
    num_capas=4, # Número de capas
    filtro_inicial=16, # Número de filtros en la primera capa
    n_clases=75, # Número de clases
    n_neuronas_fc=512 # Número de neuronas en la capa totalmente conectada
)

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = modelo.fit(datagen.flow(X_train_img, y_train_names, batch_size=32),
                    validation_data=(X_val_img, y_val_names),
                    steps_per_epoch=len(X_train_img) // 32,
                    epochs=65)

modelo.save('mejor_modelo.keras')

