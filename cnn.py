# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 08:28:52 2024

@author: KGP
"""

# Parte 1 - Construir el modelo de CNN

# Cómo importar las librerías
from keras.models import Sequential
from keras.layers import Conv2D #crea una capa de convolución 2D, damos imágenes bidimensionales
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = "relu")) # a la hora de agregar filtros se aconseja un número potencia de 2: 16-32-64...por un tema de rendimiento del sistema. Si no sabemos cuántos se suele escoger 32
        # es importante decir el formato de imagen, si es cuadrada o no. Intentar tamaño pequeño (64x64, 128x128, 256x256...)por el coste de tiempo computacional.
        # input_shape = (128, 128, 3): primero nº de filas, nº de columnas y nº canales de color (1 byn, 3 es RGB)

# Paso 2 - Maxpooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# tras hacer el test y obtener un test un poco pobre, añado una segunda capa de convolución y maxpooling
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu")) #borro el input_shape porque ya he dicho antes el tamaño de entrada que además tras el primer filtrado ha cambiado
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Paso 3 - Flattering
classifier.add(Flatten())

# Paso 4 - Full Connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid")) #Ponemos dim =1 porque lo consideramos un problema binario, no son dos problemas separados, estan relacionados

# Compilar la CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])



# Parte 2 - Ajustaremos la CNN a las imágenes para entrenar

from keras.preprocessing.image import ImageDataGenerator #vamos a reescalar, hacer simetrias, giros de las mismas imágenes para tener muchas mas imagenes para entrenar

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) #reescalado en el conjunto de test

training_dataset = train_datagen.flow_from_directory( #procede a cargar toda una carpeta de trabajo, en este caso de training
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

testing_dataset = test_datagen.flow_from_directory( #procede a cargar toda una carpeta de trabajo, en este caso de set
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_dataset,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=testing_dataset,
        validation_steps=2000)

# Una vez hecho el testing si no estamos satisfechos con los resultados, podemos ir probando cambiando filtros, nº de convoluciones etc etc