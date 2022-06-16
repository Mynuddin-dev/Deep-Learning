from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import *
from keras.models import *
import numpy as np
import matplotlib.pyplot as plt
(train_images , train_labels) , (validation_images , validation_labels) = mnist.load_data()

print("Training Dataset : ")
print(train_images.shape)
print(train_labels.shape)

print("Validation dataset : ")
print(validation_images.shape)
print(validation_labels.shape)


train_labels = to_categorical(train_labels)

model = Sequential()
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters = 6, 
                 kernel_size = 5, 
                 strides = 1, 
                 activation = 'sigmoid', 
                 input_shape = (28,28,1)))
#Pooling layer 1
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters = 16, 
                 kernel_size = 5,
                 strides = 1,
                 activation = 'sigmoid',
                 input_shape = (14,14,6)))
#Pooling Layer 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Flatten
model.add(Flatten())
#Layer 3
#Fully connected layer 1
model.add(Dense(units = 120, activation = 'sigmoid'))

#Layer 4
#Fully connected layer 2
model.add(Dense(units = 84, activation = 'sigmoid'))
model.add(Dropout(0.2))

#Layer 5
#Output Layer
model.add(Dense(units = 10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())

model.fit(train_images,train_labels , epochs =10 , batch_size = 10 , validation_split = 0.2)

y_pred = model.predict(validation_images)

#Converting one hot vectors to labels
labels = np.argmax(y_pred, axis = 1)

print(labels)
