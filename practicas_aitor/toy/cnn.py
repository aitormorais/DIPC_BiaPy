

import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout

train = pd.read_csv('prueba/archive/Training_set.csv')
test = pd.read_csv('prueba/archive/Testing_set.csv')


def ver_imagen(ruta):
    img = cv2.imread(ruta)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV utiliza el espacio de colores BGR, por lo que necesitamos convertirlo a RGB para mostrarlo correctamente con matplotlib
    plt.imshow(img)
    plt.show()

def dimensiones(ruta):
    """retorna las dimensiones de la imagen pasandole la ruta"""
    return cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB).shape


def img_prepoces(ruta,tupla):
    return cv2.resize(cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB),(tupla[0],tupla[1]))/255



"""Dividimos las imagenes en train y test"""

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


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

from sklearn.preprocessing import LabelEncoder


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

"""**Paso 1-Dividir las imagenes etiquetadas en train y validation**"""

#primero alamacenar todas las imagenes en un array
train_img_dir = []#aqui almacenare las rutas de las imagenes train

#creamos el array donde guardaremos las imagenes en forma de matrix
train_img=[]

#segundo añadir al array la url de cada imagen y convertirla a una matrix
for name in train['filename']:
    #train_img_dir.append('/kaggle/input/butterfly-image-classification/train/'+name)
    train_img.append(img_prepoces("prueba/archive/train/"+name,(224,224)))

#tercero, guardar las etiquetas
names = train['label']



X_train_img, X_val_img, y_train_names, y_val_names = train_test_split(train_img, names, test_size=0.2, random_state=42)

"""**Resize:**
se espera que las imágenes de entrada estén en un array 4D, con forma (número_de_imágenes, altura, ancho, canales)
"""

X_train_img = np.array(X_train_img).reshape(-1, 224, 224, 3)
X_val_img = np.array(X_val_img).reshape(-1, 224, 224, 3)

len(X_val_img)

from keras.preprocessing.image import ImageDataGenerator

# Crear un generador de datos de imagen con aumentación
datagen = ImageDataGenerator(
        rotation_range=30,  # Rotar aleatoriamente las imágenes en el rango (grados, 0 a 180)
        zoom_range = 0.1, # Zoom aleatoriamente las imágenes dentro del rango
        width_shift_range=0.1,  # Desplazar aleatoriamente las imágenes horizontalmente
        height_shift_range=0.1,  # Desplazar aleatoriamente las imágenes verticalmente
        horizontal_flip=True,  # Invertir aleatoriamente las imágenes horizontalmente
        vertical_flip=False)  # No invertir las imágenes verticalmente

# Ajustar el generador de datos a tus datos de entrenamiento
datagen.fit(X_train_img)





#convertimos los nombres a formato one-hot
y_train_names,y_val_names = convertir_labels_one_hot(y_train_names, y_val_names)

"""**Paso 2 Creamos el modelo fc**"""

modelo = construir_modelo_fc((224,224,3),'relu',4,16,75,512)

"""**Paso3-Callbacks**"""

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint("mejor.h5", save_best_only=True),  # Guarda el mejor modelo como 'mejor_modelo.h5'
    EarlyStopping(patience=10, restore_best_weights=True),  # Detén el entrenamiento si el modelo deja de mejorar
    ReduceLROnPlateau(patience=5)  # Reduce la tasa de aprendizaje si el modelo deja de mejorar
]


"""**COMPILAR & TRAIN**"""

# Compila el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#entrenar
# Entrena el modelo
#history = modelo.fit(X_train_img, y_train_names, validation_data=(X_val_img, y_val_names), epochs=20, callbacks=callbacks)

# Supongamos que 'model' es tu modelo de aprendizaje profundo
history = modelo.fit(datagen.flow(X_train_img, y_train_names, batch_size=32),
                    validation_data=(X_val_img, y_val_names),
                    steps_per_epoch=len(X_train_img) // 32,
                    epochs=1)



""""

def guardar_csv(history,numero,nombre):
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Genera un nombre para el modelo
    model_name = "Modelo"+str(numero)

    # Crea un DataFrame con las columnas especificadas
    new_data = pd.DataFrame({
        'Model': [model_name] * len(train_loss),
        'Epoch': range(1, len(train_loss) + 1),
        'Loss': train_loss,
        'Accuracy': train_accuracy,
        'Val_loss': val_loss,
        'Val_Accuracy': val_accuracy
    })

    # Lee el archivo CSV existente
    existing_data = pd.read_csv(nombre+'.csv')

    # Combina el DataFrame existente con la nueva información
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Guarda el DataFrame combinado en un archivo CSV
    combined_data.to_csv('INFOR.csv', index=False)

    print("Nueva información agregada al archivo CSV exitosamente.")

import pandas as pd
import matplotlib.pyplot as plt

def mostrar_graficas_modelos(csv_file):
    # Lee el archivo CSV en un DataFrame
    data = pd.read_csv(csv_file)

    # Obtiene los nombres únicos de los modelos
    modelos = data['Model'].unique()

    # Crea una figura para mostrar las gráficas
    fig, axs = plt.subplots(len(modelos), 2, figsize=(10, 5 * len(modelos)))

    # Itera sobre los modelos y muestra las gráficas correspondientes
    for i, modelo in enumerate(modelos):
        # Filtra los datos del modelo actual
        modelo_data = data[data['Model'] == modelo]

        # Obtiene las métricas
        epochs = modelo_data['Epoch']
        accuracy = modelo_data['Accuracy']
        val_accuracy = modelo_data['Val_Accuracy']
        loss = modelo_data['Loss']
        val_loss = modelo_data['Val_loss']

        # Grafica Accuracy
        axs[i, 0].plot(epochs, accuracy, label='Train accuracy')
        axs[i, 0].plot(epochs, val_accuracy, label='Test accuracy')
        axs[i, 0].set_ylabel('Accuracy')
        axs[i, 0].legend(loc='lower right')
        axs[i, 0].set_title(f'{modelo} - Accuracy eval')

        # Grafica Loss
        axs[i, 1].plot(epochs, loss, label='Train error')
        axs[i, 1].plot(epochs, val_loss, label='Test error')
        axs[i, 1].set_ylabel('Error')
        axs[i, 1].set_xlabel('Epoch')
        axs[i, 1].legend(loc='upper right')
        axs[i, 1].set_title(f'{modelo} - Error eval')

    # Ajusta los espacios entre las subfiguras
    plt.tight_layout()

    # Muestra las gráficas
    plt.show()

#mostrar_graficas_modelos('/scratch/aitor.morais/INFOR.csv')

import pandas as pd

def borrar_modelo_csv(csv_file, modelo):
    # Lee el archivo CSV en un DataFrame
    data = pd.read_csv(csv_file)

    # Filtra los datos para excluir el modelo especificado
    data = data[data['Model'] != modelo]

    # Guarda los datos actualizados en el archivo CSV
    data.to_csv(csv_file, index=False)

    print(f"Información del modelo '{modelo}' eliminada exitosamente del archivo CSV.")

# Llama a la función para borrar el modelo3 del archivo CSV
borrar_modelo_csv('INFOR.csv', 'Modelo3')

import matplotlib.pyplot as plt


def plot_training_history(history):
    fig, axs = plt.subplots(2)

    #  accuracy
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    #  error
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def ver_graficos(history):
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    history_data = {
        "accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "loss": train_loss,
        "val_loss": val_loss
    }

    history = type('', (), {})()
    history.history = history_data

    plot_training_history(history)

ver_graficos(history)



import time

hr, minute = map(int, time.strftime("%H %M").split())

type(minute)

import csv

def abrir_csv(direccion):
    # Abre el archivo CSV en modo lectura
    with open(direccion, 'r') as archivo:
        # Lee el contenido del archivo CSV
        contenido = archivo.readlines()

    # Lee el archivo CSV utilizando csv.reader
    reader = csv.reader(contenido)

    # Convierte el lector en una lista para facilitar su manipulación
    datos = list(reader)

    return datos




abrir_csv('/scratch/aitor.morais/INFOR.csv')"""
"""
import csv
from datetime import datetime

# Supongamos que tienes el historial de métricas almacenado en el objeto `History` de Keras
# y lo has asignado a la variable `history`
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Genera un nombre para el modelo basado en la hora actual
model_name = "asdasd"

# Especifica la ruta del archivo CSV
ruta_archivo = "/kaggle/working/history.csv"

# Abre el archivo CSV en modo escritura
with open(ruta_archivo, 'a', newline='') as file:
    writer = csv.writer(file)

    # Si el archivo está vacío, escribe los encabezados
    if file.tell() == 0:
        writer.writerow(['epoch', 'Modelo', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])

    # Escribe las métricas del modelo actual en el archivo CSV por cada época
    for epoch in range(len(train_loss)):
        writer.writerow([epoch + 1, model_name, train_loss[epoch], train_accuracy[epoch],
                         val_loss[epoch], val_accuracy[epoch]])

print("Métricas guardadas en el archivo CSV.")

import csv
from datetime import datetime

# Supongamos que tienes el historial de métricas almacenado en el objeto `History` de Keras
# y lo has asignado a la variable `history`
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Genera un nombre para el modelo basado en la hora actual
model_name = "Modelo02"

# Abre el archivo CSV en modo escritura
with open('info.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    # Si el archivo está vacío, escribe los encabezados
    if file.tell() == 0:
        writer.writerow(['epoch', 'Modelo', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])

    # Escribe las métricas del modelo actual en el archivo CSV por cada época
    for epoch in range(len(train_loss)):
        writer.writerow([epoch + 1, model_name, train_loss[epoch], train_accuracy[epoch],
                         val_loss[epoch], val_accuracy[epoch]])

print("Métricas guardadas en el archivo CSV.")

# Convierte el objeto history a un DataFrame
history_df = pd.DataFrame(history.history)

# Añade una columna para el número de corrida
history_df['run'] = time.time()  # Supón que 'run_number' es un identificador único para cada corrida

# Añade el DataFrame al archivo CSV existente sin sobrescribir los datos existentes
with open('history.csv', 'a') as f:
    history_df.to_csv(f, header=f.tell()==0)

def cargar_datos_csv(url):
    history_df = pd.read_csv(url)
    history = type('', (), {})()
    history.history = history_df.to_dict(orient='list')
    # Plot the training history
    plot_training_history(history)

cargar_datos_csv('/kaggle/working/history.csv')"""