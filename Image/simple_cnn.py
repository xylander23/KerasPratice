from keras import layers
from keras import models

class ImageNn():
    def buildModel(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))  # 卷积层(通道数， 卷积核大小， 激活函数，输入数据（图像高、图像宽、图像频道）)
        self.model.add(layers.MaxPooling2D((2,2)))  # 池化层
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(10, activation="softmax"))

    def train(self, train_data, train_label, epochs=5, batch_size=64):
        self.buildModel()
        self.model.compile(optimizer="rmsprop",
                           loss="categorical_crossentropy",
                           metrics= ["accuracy"]
                           )
        self.model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size)
