import os
from keras.models import load_model
from keras import models
from keras import backend as K
from keras_preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2

# cnn可视化，参考deep learning with python一书

class PredictImg():
    def __init__(self,model_path,img_path):
        self.model_path=model_path
        self.img_path=img_path

    def img_to_tensor(self):
        img = image.load_img(self.img_path, target_size=(80, 80))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255
        print(img_tensor.shape)
        return img_tensor

    def predict(self):
        model = load_model(self.model_path)
        pred=model.predict(self.img_to_tensor())
        print(pred)
        result=np.argmax(pred[0])
        print(result)
        if result==0:
            print('A_shape')
        elif result==1:
            print('down_shape')
        elif result==2:
            print('rise_shape')
        elif result==3:
            print('U_shape')
        return result

    def plot_img_tensor(self):
        plt.imshow(self.img_to_tensor()[0])
        print(self.img_to_tensor().shape)
        plt.show()

    def plot_onelayer_onechannel(self,layerid,channel_id,layer_before):
        model = load_model(self.model_path)
        layer_outputs = [layer.output for layer in model.layers[:layer_before]]  # layer_before 前多少层，layer_before=5，获取前5层
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(self.img_to_tensor())
        choose_layer_activation = activations[layerid]  # layerid=0 查看第一层激活输出
        print(choose_layer_activation.shape)
        plt.matshow(choose_layer_activation[0, :, :, channel_id], cmap='viridis')   # channel_id 查看layerid层的第channel_id通道图片
        plt.show()

    def plot_layer_allchannel(self,begin_layer,end_layer):  # 查看层数越多，打印图片对内存要求越高
        model = load_model(self.model_path)
        layer_outputs = [layer.output for layer in model.layers[begin_layer:end_layer]]
        layer_names = []
        for layer in model.layers[begin_layer:end_layer]:
            layer_names.append(layer.name)
        images_per_row = 16
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(self.img_to_tensor())
        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0, :, :,col * images_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size,
                    row * size: (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()


    def plot_heatmaps(self,con2dlayer_name,con2dlayer_channel): # 打印某层的热力图
        model = load_model(self.model_path)
        predict_result=self.predict()
        shape_output = model.output[:, predict_result]
        last_conv_layer = model.get_layer(con2dlayer_name)
        grads = K.gradients(shape_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([self.img_to_tensor()])
        for i in range(con2dlayer_channel):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        # print(heatmap)
        plt.matshow(heatmap)
        plt.show()
        plt.matshow(heatmap)
        img = cv2.imread(self.img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.9 + img
        cv2.imwrite(result_dir+os.sep+str(predict_result)+'_heatmap.jpg', superimposed_img)


if __name__ == '__main__':
    this_dir= os.getcwd()
    result_dir = this_dir + os.sep + 'result'
    img_path=this_dir+os.sep+'U_shape.png'
    model_path=result_dir+os.sep+'cnn_80_0.2_16_10.h5'
    PI=PredictImg(model_path,img_path)
    PI.predict()
    # PI.plot_heatmaps('conv2d_5',64)
    # PI.plot_layer_allchannel(begin_layer=0,end_layer=6)
    # PI.plot_onelayer_onechannel(layerid=2,channel_id=5,layer_before=3)
    # PI.img_to_tensor()
    # PI.plot_img_tensor()