from keras import layers
from keras import models





def cnn_model(shape=(80,80,3),dropout=0.5,last_activation='softmax'):
    model=models.Sequential()
    model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=shape))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(4,activation=last_activation))
    model.summary()
    return 'cnn', model


def dense_model(shape=(80*80,),last_activation='softmax'):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu',input_shape=shape))
    model.add(layers.Dense(4, activation=last_activation))
    model.summary()
    return 'simple_dense',model

def inception_model():
    pass




