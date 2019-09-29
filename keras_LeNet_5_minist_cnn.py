from __future__ import print_function
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Dropout,Flatten
from keras.datasets import mnist
import keras
from keras import backend as k
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

# 归一化
x_train /= 255
y_train /= 255

# 独热编码
from keras.utils import np_utils
y_train_new = np_utils.to_categorical(num_classes=10, y=y_train)
y_test_new = np_utils.to_categorical(num_classes=10, y=y_test)


def LeNet5():
    model = Sequential()
    # 模型第一层要指定输入的维度，input_shape
    model.add(Conv2D(filters=6,kernel_size=(5,5),padding='valid',activation='tanh',input_shape=(1, 28, 28)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120,activation='tanh'))
    model.add(Dense(84,activation='tanh'))
    # softmax函数是用来计算该输入图片属于0-9的概率
    model.add(Dense(10,activation='softmax'))
    return model


def train_model():
    model = LeNet5()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=64,epochs=12,verbose=1,validation_split=0.2,shuffle=True)
    return model


# 返回测试集损失函数值和准确率
# loss,accuracy = model.evaluate(x_test,y_test_new)
