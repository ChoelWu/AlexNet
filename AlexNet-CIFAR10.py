import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# 零均值归一化
def normalize(X_train, X_test):
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))  # 均值
    std = np.std(X_train, axis=(0, 1, 2, 3))  # 标准差
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


# 预处理
def preprocess(x, y):
    x = tf.image.resize(x, (227, 227))  # 将32*32的图片放大为227*227的图片
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    y = tf.squeeze(y, axis=1)  # 将(50000, 1)的数组转化为(50000)的Tensor
    y = tf.one_hot(y, depth=10)
    return x, y


# 零均值归一化
x_train, x_test = normalize(x_train, x_test)

# 预处理
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(50000).batch(128).map(preprocess)  # 每个批次128个训练样本

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(10000).batch(128).map(preprocess)  # 每个批次128个测试样本


# 可以自定义填充数量的卷积层
class ConvWithPadding(tf.keras.layers.Layer):
    def __init__(self, kernel, filters, strides, padding):
        super().__init__()
        self.kernel = kernel
        self.filters = filters
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.w = tf.random.normal([self.filters, self.filters, input_shape[-1], self.kernel])

    def call(self, inputs):
        return tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)


batch = 32

alex_net = keras.Sequential([
    # 卷积层1
    keras.layers.Conv2D(96, 11, 4),  # 输入为227*227@3的图片，通过96个大小为11*11@3的卷积核，步长为4，无填充，后得到55*55@96的特征图
    keras.layers.ReLU(),  # ReLU激活
    keras.layers.MaxPooling2D((3, 3), 2),  # 重叠最大池化，大小为3*3，步长为2，最后得到27*27@96的特征图
    keras.layers.BatchNormalization(),
    # 卷积层2
    #     ConvWithPadding(kernel=256, filters=5, strides=1, padding=[[0, 0], [2, 2], [2, 2], [0, 0]]),
    keras.layers.Conv2D(256, 5, 1, padding='same'),  # 输入27*27@96，卷积核256个，大小5*5@96，步长1，填充2，得到27*27@96(与输入等长宽)特征图
    keras.layers.ReLU(),
    keras.layers.MaxPooling2D((3, 3), 2),  # 重叠最大池化，大小为3*3，步长为2，最后得到13*13@256的特征图
    keras.layers.BatchNormalization(),
    # 卷积层3
    keras.layers.Conv2D(384, 3, 1, padding='same'),  # 输入13*13@256，卷积核384个，大小3*3@256，步长1，填充1，得到13*13@384(与输入等长宽)特征图
    keras.layers.ReLU(),
    # 卷积层4
    keras.layers.Conv2D(384, 3, 1, padding='same'),  # 输入13*13@384，卷积核384个，大小3*3@384，步长1，填充1，得到13*13@384(与输入等长宽)特征图
    keras.layers.ReLU(),
    # 卷积层5
    keras.layers.Conv2D(256, 3, 1, padding='same'),  # 输入13*13@384，卷积核256个，大小3*3@384，步长1，填充1，得到13*13@256(与输入等长宽)特征图
    keras.layers.ReLU(),
    keras.layers.MaxPooling2D((3, 3), 2),  # 重叠最大池化，大小为3*3，步长为2，最后得到6*6@256的特征图
    # 全连接层1
    keras.layers.Flatten(),  # 将6*6@256的特征图拉伸成9216个像素点
    keras.layers.Dense(4096),  # 9216*4096的全连接
    keras.layers.ReLU(),
    keras.layers.Dropout(0.25),  # Dropout 25%的神经元
    # 全连接层2
    keras.layers.Dense(4096),  # 4096*4096的全连接
    keras.layers.ReLU(),
    keras.layers.Dropout(0.25),  # Dropout 25%的神经元
    # 全连接层3
    keras.layers.Dense(10, activation='softmax')  # 4096*10的全连接，通过softmax后10分类
])

alex_net.build(input_shape=[batch, 227, 227, 3])
alex_net.summary()

# 网络编译参数设置
loss = keras.losses.CategoricalCrossentropy()
alex_net.compile(optimizer=keras.optimizers.Adam(0.00001), loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

# 训练
history = alex_net.fit(train_db, epochs=10)

# 损失下降曲线
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

# 测试
alex_net.evaluate(test_db)

plt.show()
