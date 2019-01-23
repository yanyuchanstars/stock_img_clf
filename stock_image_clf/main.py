import os
import somemodels as md
import vectorization as vr
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


class FitModel():
    def __init__(self,data_enhance=False,img_size=(80,80),epochs=10,batch_size=32):
        self.img_size=img_size
        self.epochs=epochs
        self.batch_size=batch_size
        if data_enhance:
            self.l=[0,0.1,0.1,0.1]
        else:
            self.l=[0,0,0,0]

    def fm(self):
        train_input = vr.ImgVectorization('train',img_size=self.img_size,batch_size=self.batch_size,
                                          rotation=self.l[0],hs=self.l[1],ws=self.l[2],zr=self.l[3]).vec_generator()
        val_input = vr.ImgVectorization('val',img_size=self.img_size,batch_size=self.batch_size).vec_generator()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])  # 损失函数为分类交叉熵
        history = model.fit_generator(train_input, steps_per_epoch=50, epochs=self.epochs, validation_data=val_input, validation_steps=12)
        best_acc=np.amax(history.history['val_acc'])
        print(best_acc)
        model.save(result_dir + os.sep + model_name + '_'+str(pic_size)+'_'+str(dropout)+'_'+str(batch_size)+'_'+str(self.epochs)+'.h5')
        self.plot(history)

    def plot(self,history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(epochs, acc, 'bo', label='train')
        plt.plot(epochs, val_acc, 'b', label='val')
        plt.title('accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(epochs, loss, 'bo', label='train')
        plt.plot(epochs, val_loss, 'b', label='val')
        plt.title('loss')
        plt.legend()
        plt.savefig(result_dir + os.sep + model_name + str(pic_size)+'_'+str(self.epochs)+'_plot')
        plt.show()


def evaluate_on_test(model_file,img_size=(80,80),batch_size=32):
    read_model = load_model( model_file)
    X_test,Y_test = vr.ImgVectorization('test',img_size=img_size,batch_size=batch_size).vec_all()  # 评估验证集
    test_score = read_model.evaluate(X_test,Y_test)
    print(test_score)

if __name__ == '__main__':
    this_dir = os.getcwd()
    result_dir = this_dir+os.sep+'result'
    pic_size=80
    epochs = 10
    batch_size=16
    dropout=0.2
    # model_name, model = md.dense_model()
    model_name, model = md.cnn_model(shape=(pic_size,pic_size,3),dropout=dropout)          # 选择模型
    FitModel(data_enhance=False,img_size=(pic_size,pic_size),epochs=epochs,batch_size=batch_size).fm()
    model_file=result_dir + os.sep + model_name + '_'+str(pic_size)+'_'+str(dropout)+'_'+str(batch_size)+'_'+str(epochs)+'.h5'
    evaluate_on_test(model_file,img_size=(pic_size,pic_size),batch_size=batch_size)                 # 在测试集评估所有模型



