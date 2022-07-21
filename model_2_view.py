import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets
import numpy as np
import ReadData_2 as RD

epochs = 80


# STEP1 加载数据
def load_data(step, sample_len):
    path = r'data\1HP'
    rate = [0.7, 0.15, 0.15]
    x_train, y_train, x_validate, y_validate, x_test, y_test = RD.get_data(path, rate, step, sample_len)
    # 切片
    sample = tf.data.Dataset.from_tensor_slices((x_train, y_train))   # 按照样本数进行切片得到每一片的表述（2048+10，1）
    sample = sample.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    sample_validate = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
    sample_validate = sample_validate.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    sample_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    sample_test = sample_test.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    return sample, sample_validate, sample_test


# STEP2 设计网络结构，建立网络容器
def create_model():
    Con_net = keras.Sequential([  # 网络容器
        layers.Conv1D(filters=32, kernel_size=20, strides=1, padding='same', activation='relu'),  # 添加卷积层
        layers.BatchNormalization(),  # 添加正则化层
        layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 池化层
        layers.Conv1D(filters=32, kernel_size=20, strides=1, padding='same', activation='relu'),  # 添加卷积层
        layers.BatchNormalization(),  # 添加正则化层
        layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 池化层
        layers.Flatten(),  # 打平层，方便全连接层使用
        layers.Dense(100, activation='relu'),  # 全连接层，120个节点
        layers.Dense(10, activation='softmax'),  # 全连接层，10个类别节点
    ])
    return Con_net


def train(sample1, sample1_validate, sample1_test, sample_len):
    res = []
    Con_net = create_model()  # 建立网络模型
    Con_net.build(input_shape=(10, sample_len, 2))  # 构建一个卷积网络，输入的尺寸  ----------------------
    optimizer = optimizers.Adam(lr=1e-4)  # 设置优化器
    variables = Con_net.trainable_variables
    for epoch in range(epochs):  # 外循环，遍历多少次训练数据集
        for step, (x, y) in enumerate(sample1):  # 遍历一次训练集的所有样例
            with tf.GradientTape() as tape:  # 构建梯度环境 # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = Con_net(x)  # flatten, => [b, 512]
                loss = tf.losses.categorical_crossentropy(y, out)  # compute loss
                loss = tf.reduce_mean(loss)  # 求损失的平均值
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            if step % 1000 == 0:
                print(epoch, step, 'loss:', float(loss))
        # print("验证集正确率")
        # test(Con_net, sample1_validate)
        # print("测试集正确率")
        res.append(test(Con_net, sample1_test))
    result.append(res)
    # Con_net.save_weights('./cnn_save_weights_400')


def test(Con_net, sample_data):
    total_num = 0
    total_correct = 0
    for x, y in sample_data:
        # Con_net = create_model()  # 建立网络模型
        # Con_net.load_weights('./cnn_save_weights_400')
        out = Con_net(x)  # 前向计算
        predict = tf.argmax(out, axis=-1)  # axis=-1, 倒数第一维, 返回每行的最大值坐标
        # print("predict", predict)
        y = tf.cast(y, tf.int64)
        # print("y", y)
        m = predict == y
        m = tf.cast(m, dtype=tf.int64)   # tensor张量类型
        total_correct += int(tf.reduce_sum(m))
        total_num += x.shape[0]
        if total_num < total_correct:
            print("error---------------------------")
            print("正确",total_correct,"总数",total_num)
    acc = total_correct / total_num
    # print('acc:', acc)
    return acc


def run_step():  # epoch=10
    step_len = list(range(210,430,10))
    #step_len = [420]
    # step_len = [210]
    for i in step_len:
        sample1, sample1_validate, sample1_test = load_data(step=i, sample_len=420)
        train(sample1, sample1_validate, sample1_test,sample_len=420)


def run_sample():
    sample_len = list(range(1,7))
    # sample_len = [1]
    for i in sample_len:
        sample1, sample1_validate, sample1_test = load_data(step=210, sample_len=420*i)
        train(sample1, sample1_validate, sample1_test, sample_len=420*i)


# 当epoch=10时，随着步长的变化，实验结果的变化
result = []
run_step()
#run_sample()
print(result)

