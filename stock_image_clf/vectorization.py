import os
from keras.preprocessing.image import ImageDataGenerator

this_dir=os.getcwd()


# 图像数据预处理
class ImgVectorization():
    def __init__(self,datatype='train',img_size=(80,80),batch_size=20,mode='categorical',rotation=0,ws=0,hs=0,zr=0,hf=False,shuff=True):
        self.data_dir=this_dir+os.sep+datatype
        self.img_size = img_size
        self.shuff=shuff
        self.batch_size=batch_size
        self.mode=mode
        self.rotation=rotation
        self.ws=ws
        self.hs=hs
        self.zr=zr
        self.hf=hf

    def check_img_amount(self):  # 检查各类图片数量一致
        list=[]
        for folder in os.listdir(self.data_dir):
            img_amount = len(os.listdir(self.data_dir + os.sep + folder))
            img_class = folder
            print(img_class, img_amount)
            list.append(img_amount)
        if len(set(list))==1:
            return list[0]*len(list)    # 返回该目录图片数量
        else:
            print('样本分类数量不一致')

    def lable_list(self):     # 随机森林使用的label_list
        list_lable=[]
        for folder in os.listdir(self.data_dir):
            img_amount = len(os.listdir(self.data_dir + os.sep + folder))
            for im in range(img_amount):
                list_lable.append(int(folder))
        return list_lable

    def vec_generator(self):      # 按批 向量化生成器
        data = ImageDataGenerator(rescale=1. / 255, rotation_range=self.rotation, width_shift_range=self.ws,
                                  height_shift_range=self.hs, zoom_range=self.zr,
                                  horizontal_flip=self.hf)
        data_generator = data.flow_from_directory(self.data_dir, target_size=self.img_size, batch_size=self.batch_size,
                                                  class_mode=self.mode,shuffle=self.shuff)
        return data_generator

    def check_input_type(self):   # 检查生成数据格式
        for data_batch,lables_batch in self.vec_generator():
            print(data_batch.shape)
            print(lables_batch.shape)
            break

    def vec_all(self):  # 向量化全部数据
        data_amount=self.check_img_amount()
        self.batch_size=data_amount
        list=[]
        for data_batch,data_label in self.vec_generator():
            list.append(data_batch)
            list.append(data_label)
            break
        return list[0],list[1]




# 文本数据预处理
class TextVectorization():
    pass