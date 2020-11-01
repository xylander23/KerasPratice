from keras.datasets import mnist
from keras.utils import to_categorical

from simple_cnn import ImageNn


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255 # 每个元素取值范围在255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype("float32") / 255

    train_labels = to_categorical(train_labels)  # 训练标签向量化
    test_labels = to_categorical(test_labels)

    model = ImageNn()
    model.train(train_images, train_labels)

