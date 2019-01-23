import os
import vectorization as vr
from keras import layers
from keras import models
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# 超参数优化

def data():
    x_train, y_train = vr.ImgVectorization('train').vec_all()
    x_test, y_test = vr.ImgVectorization('val').vec_all()
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    model = models.Sequential()
    model.add(layers.Conv2D({{choice([16, 32, 64, 128])}}, (3, 3), activation='relu', input_shape=(80,80,3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D({{choice([16, 32, 64, 128])}}, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D({{choice([16, 32, 64, 128])}}, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout({{uniform(0, 1)}}))
    model.add(layers.Dense({{choice([16, 32, 64, 128])}}, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='rmsprop')
    result = model.fit(x_train, y_train,
                       batch_size={{choice([32, 64, 128])}},
                       epochs={{choice([10,20])}},
                       validation_data=(x_test, y_test))
    val_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', val_acc)
    return {'loss': -val_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    this_dir= os.getcwd()
    result_dir = this_dir + os.sep + 'result'
    X_train, Y_train, X_test, Y_test = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())
    best_model.save(result_dir + os.sep + 'cnn_hyper.h5')
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
