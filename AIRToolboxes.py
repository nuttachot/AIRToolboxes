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
epochs = 3

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

##############################################10
start = time.time()

his = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

done = time.time()
print(done - start)

##############################################11
with open('history_model', 'wb') as file:
    p.dump(his.history, file)

##############################################12
with open('history_model', 'rb') as file:
     his = p.load(file)
    
##############################################13
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##############################################14
h1 = go.Scatter(y=his['loss'], 
                    mode="lines", line=dict(
                    width=2,
                    color='blue'),
                    name="loss"
                   )
h2 = go.Scatter(y=his['val_loss'], 
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

##############################################16
filepath='model1.h5'
model.save(filepath)

##############################################17
predict_model = load_model(filepath) 
predict_model.summary()

##############################################18
plt.figure(figsize=(5, 4))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(temp_x_test[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

##############################################19
x_test[:1].shape
y_test[:1].shape
np.argmax(y_test[0], axis=0)

##############################################20
result = predict_model.predict_classes(x_test[:1])
print(result[0])

##############################################21
filepath_model = 'model1.json'
filepath_weights = 'weights_model.h5'

model_json = model.to_json()
with open(filepath_model, "w") as json_file:
    json_file.write(model_json)
    
    model.save_weights('weights_model.h5')
    print("Saved model to disk")

##############################################22
with open(filepath_model, 'r') as f:
    loaded_model_json = f.read()
    predict_model = model_from_json(loaded_model_json)
    predict_model.load_weights(filepath_weights)    
    print("Loaded model from disk")

##############################################23
#predict_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
score = predict_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##############################################24
predicted_classes = predict_model.predict_classes(x_test)

##############################################25
y_true = np.argmax(y_test, axis=1)

##############################################26
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]

print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])

##############################################27
labels = {0 : "0", 1: "1", 2: "2", 3: "3", 4: "4",
          5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}

target_names = ["Class {} ({}) :".format(i,labels[i]) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names, digits=4))


##############################################28
1. ให้นักศึกษาแสดงกราฟ Accuacy ขณะที่ Train Model
2. ให้นักศึกษารัน Train Model บน Colab โดยปรับ Epoch เป็น 100 แล้วแสดงกราฟ Accuracy/Loss/Precision/Recall/f1-score
3. ให้นักศึกษารัน Train Model บน Colab โดยปรับ batch_size เป็น 256 แล้วแสดงกราฟ Accuracy/Loss/Precision/Recall/f1-score และความเร็วในการ Train Model



