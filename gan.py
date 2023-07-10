import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from google.colab import drive
import os
drive.mount('/content/gdrive/')

import keras
from keras.layers import Dense, Conv2DTranspose, LeakyReLU, Reshape, BatchNormalization, Activation, Conv2D
from keras.models import Model, Sequential


def generador_de_imagenes():

    generador = Sequential()

    generador.add(Dense(256*4*4, input_shape = (100,)))
    #generador.add(BatchNormalization())
    generador.add(LeakyReLU())
    generador.add(Reshape((4,4,256)))

    generador.add(Conv2DTranspose(128,kernel_size=3, strides=2, padding = "same"))
    #generador.add(BatchNormalization())
    generador.add(LeakyReLU(alpha=0.2))


    generador.add(Conv2DTranspose(128,kernel_size=3, strides=2, padding = "same"))
    #generador.add(BatchNormalization())
    generador.add(LeakyReLU(alpha=0.2))

    generador.add(Conv2DTranspose(128,kernel_size=3, strides=2, padding = "same"))
    #generador.add(BatchNormalization())
    generador.add(LeakyReLU(alpha=0.2))

    generador.add(Conv2D(3,kernel_size=3, padding = "same", activation='tanh'))

    return(generador)

modelo_generador = generador_de_imagenes()

modelo_generador.summary()

import matplotlib.pyplot as plt
import numpy as np

# Definir datos de entrada
def generar_datos_entrada(n_muestras):
  X = np.random.randn(100 * n_muestras)
  X = X.reshape(n_muestras, 100)
  return X

def crear_datos_fake(modelo_generador, n_muestras):
  input = generar_datos_entrada(n_muestras)
  X = modelo_generador.predict(input)
  y = np.zeros((n_muestras, 1))
  return X,y

numero_muestras = 4
X,_ = crear_datos_fake(modelo_generador, numero_muestras)

# Visualizamos resultados
for i in range(numero_muestras):
    plt.subplot(2, 2, 1 + i)
    plt.axis('off')
    plt.imshow(X[i])



from keras.layers import Conv2D, Flatten, Dropout
from keras.optimizers import Adam

def discriminador_de_imagenes():

    discriminador = Sequential()
    discriminador.add(Conv2D(64, kernel_size=3, padding = "same", input_shape = (32,32,3)))
    discriminador.add(LeakyReLU(alpha=0.2))
    #discriminador.add(Dropout(0.2))

    discriminador.add(Conv2D(128, kernel_size=3,strides=(2,2), padding = "same"))
    discriminador.add(LeakyReLU(alpha=0.2))
    #discriminador.add(Dropout(0.2))

    discriminador.add(Conv2D(128, kernel_size=3,strides=(2,2), padding = "same"))
    discriminador.add(LeakyReLU(alpha=0.2))
    #discriminador.add(Dropout(0.2))

    discriminador.add(Conv2D(256, kernel_size=3, strides=(2,2), padding = "same"))
    discriminador.add(LeakyReLU(alpha=0.2))
    #discriminador.add(Dropout(0.2))

    discriminador.add(Flatten())
    discriminador.add(Dropout(0.4))
    discriminador.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=0.0002 ,beta_1=0.5)
    discriminador.compile(loss='binary_crossentropy', optimizer= opt , metrics = ['accuracy'])

    return(discriminador)

modelo_discriminador = discriminador_de_imagenes()
modelo_discriminador.summary()

# from keras.datasets import cifar10

# def cargar_imagenes():
#     (Xtrain, Ytrain), (_, _) = cifar10.load_data()

#     # Nos quedamos con los perros
#     indice = np.where(Ytrain == 0)
#     indice = indice[0]
#     Xtrain = Xtrain[indice, :,:,:]

#     # Normalizamos los datos
#     X = Xtrain.astype('float32')
#     X = (X - 127.5) / 127.5

#     return X

# print(cargar_imagenes().shape)



import numpy as np
from PIL import Image

def cargar_imagenes():
    # Define the directory path where your images are stored
    directory_path = '/content/generated/real_images/'

    # Load your images from the directory
    image_files = os.listdir(directory_path)

    # Create an empty list to store the loaded images
    images = []

    for file in image_files:
        # Read each image file
        image = Image.open(os.path.join(directory_path, file))

        # Resize the image to the desired shape (e.g., 32x32)
        image = image.resize((32, 32))

        # Convert the image to numpy array
        image = np.array(image)

        # Append the image to the list
        images.append(image)

    # Convert the list of images to a numpy array
    X = np.array(images)

    # Normalize the data
    X = X.astype('float32')
    X = (X - 127.5) / 127.5

    return X

print(cargar_imagenes().shape)


import random

def cargar_datos_reales(dataset, n_muestras):
  ix = np.random.randint(0, dataset.shape[0], n_muestras)
  X = dataset[ix]
  y = np.ones((n_muestras, 1))
  return X,y

def cargar_datos_fake(n_muestras):
  X = np.random.rand(32 * 32 * 3 * n_muestras)
  X = -1 + X * 2
  X = X.reshape((n_muestras, 32,32,3))
  y = np.zeros((n_muestras, 1))
  return X,y

def entrenar_discriminador(modelo, dataset, n_iteraciones=20, batch = 128):
  medio_batch = int(batch/2)

  for i in range(n_iteraciones):
    X_real, y_real = cargar_datos_reales(dataset, medio_batch)
    _, acc_real = modelo.train_on_batch(X_real, y_real)

    X_fake, y_fake = cargar_datos_fake(medio_batch)
    _, acc_fake = modelo.train_on_batch(X_fake, y_fake)

    print(str(i+1) + ' Real:' + str(acc_real*100) + ', Fake:' + str(acc_fake*100))


dataset = cargar_imagenes()
entrenar_discriminador(modelo_discriminador, dataset)

def crear_gan(discriminador, generador):
    discriminador.trainable=False
    gan = Sequential()
    gan.add(generador)
    gan.add(discriminador)

    opt = Adam(lr=0.0002,beta_1=0.5) 
    gan.compile(loss = "binary_crossentropy", optimizer = opt)

    return gan

gan = crear_gan(modelo_discriminador,modelo_generador)
gan.summary()



import pandas as pd
import matplotlib.pyplot as plt
import os
%matplotlib inline
from datetime import datetime

def mostrar_imagenes_generadas(datos_fake, epoch):
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    
    # Define the directory path where you want to save the images
    save_directory = '/content/generated/real_images/generated'
    
    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Hacemos que los datos vayan de 0 a 1
    datos_fake = (datos_fake + 1) / 2.0

    for i in range(10):
        plt.imshow(datos_fake[i])
        plt.axis('off')
        nombre = str(epoch) + '_imagen_generada_' + str(i) + '.png'
        filepath = os.path.join(save_directory, nombre)  # Create the file path
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

def evaluar_y_guardar(modelo_generador, epoch, medio_dataset):
    # We save the model
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    nombre = str(epoch) + '_' + str(now) + "_modelo_generador_" + '.h5'
    modelo_generador.save(nombre)

    # We generate new data
    X_real, Y_real = cargar_datos_reales(dataset, medio_dataset)
    X_fake, Y_fake = crear_datos_fake(modelo_generador, medio_dataset)

    # We evaluate the model
    _, acc_real = modelo_discriminador.evaluate(X_real, Y_real)
    _, acc_fake = modelo_discriminador.evaluate(X_fake, Y_fake)

    print('Acc Real:' + str(acc_real * 100) + '% Acc Fake:' + str(acc_fake * 100) + '%')

def entrenamiento(datos, modelo_generador, modelo_discriminador, epochs, n_batch, inicio=0):
    dimension_batch = int(datos.shape[0] / n_batch)
    medio_dataset = int(n_batch / 2)

    # We iterate over the epochs
    for epoch in range(inicio, inicio + epochs):
        # We iterate over all batches
        for batch in range(n_batch):

            # We load all the real data
            X_real, Y_real = cargar_datos_reales(dataset, medio_dataset)

            # We train the discriminator withEnrenamos discriminador con datos reales
            coste_discriminador_real, _ = modelo_discriminador.train_on_batch(X_real, Y_real)
            X_fake, Y_fake = crear_datos_fake(modelo_generador, medio_dataset)

            coste_discriminador_fake, _ = modelo_discriminador.train_on_batch(X_fake, Y_fake)

            # We generate input images for the GAN
            X_gan = generar_datos_entrada(medio_dataset)
            Y_gan = np.ones((medio_dataset, 1))

            # We train the GAN with fake data
            coste_gan = gan.train_on_batch(X_gan, Y_gan)

        # Every 10 epochs we show the results and cost
        if (epoch + 1) % 10 == 0:
            evaluar_y_guardar(modelo_generador, epoch=epoch, medio_dataset=medio_dataset)
            mostrar_imagenes_generadas(X_fake, epoch=epoch)

entrenamiento(dataset, modelo_generador, modelo_discriminador, epochs=10, n_batch=128, inicio=0)



X_fake, _ = crear_datos_fake(n_muestras=49, modelo_generador=modelo_generador)
X_fake = (X_fake+1)/2

for i in range(36):
  plt.subplot(6,6,i+1)
  plt.axis('off')
  plt.imshow(X_fake[i])
