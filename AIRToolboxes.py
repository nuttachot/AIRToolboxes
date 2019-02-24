##############################################1
import time

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json

import pickle as p

##############################################2
batch_size = 128
num_classes = 10
epochs = 4

img_rows, img_cols = 28, 28

##############################################3
(x_train, y_train), (x_test, y_test) = mnist.load_data()

##############################################4
plt.figure(figsize=(5, 4))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

##############################################5
x_train.min(), x_train.max()

##############################################6
temp_x_test = x_test

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

##############################################7
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

##############################################8
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

##############################################9
model.summary()

##############################################9
plotly.offline.iplot(fig1, filename="testMNIST")

##############################################10
start = time.time()

his = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

done = time.time()
print(done - start)

##############################################11
with open('history_model', 'wb') as file_pi:
    p.dump(his.history, file_pi)

##############################################12
with open('history_model', 'rb') as file:
     his = p.load(file)
    
##############################################13
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##############################################14
h1 = go.Scatter(y=his.history['loss'], 
                    mode="lines", line=dict(
                    width=2,
                    color='blue'),
                    name="loss"
                   )
h2 = go.Scatter(y=his.history['val_loss'], 
                    mode="lines", line=dict(
                    width=2,
                    color='red'),
                    name="val_loss"
                   )

##############################################15  
data = [h1,h2]
layout1 = go.Layout(title='Loss',
                   xaxis=dict(title='epochs'),
                   yaxis=dict(title=''))
fig1 = go.Figure(data, layout=layout1)
plotly.offline.iplot(fig1, filename="testMNIST")

##############################################15  
filepath='model1.h5'
model.save(filepath)

##############################################16
predict_model = load_model(filepath) 
predict_model.summary()

##############################################17
plt.figure(figsize=(5, 4))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(temp_x_test[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

##############################################18
x_test[:1].shape
y_test[:1].shape
np.argmax(y_test[0], axis=0)

##############################################19
result = predict_model.predict_classes(x_test[:1])
print(result[0])

##############################################20
filepath_model = 'model1.json'
filepath_weights = 'weights_model.h5'

model_json = model.to_json()
with open(filepath_model, "w") as json_file:
    json_file.write(model_json)
    
    model.save_weights('weights_model.h5')
    print("Saved model to disk")

##############################################21
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model = model_from_json(loaded_model_json)
    predict_model.load_weights(filepath_weights)    
    print("Loaded model from disk")

##############################################21
#predict_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
score = predict_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##############################################22
from google.colab import drive
drive.mount('/content/gdrive')

##############################################23
with open('/content/gdrive/My Drive/AIRtest/foo.txt', 'w') as f:
    f.write('Hello Google Drive!')




