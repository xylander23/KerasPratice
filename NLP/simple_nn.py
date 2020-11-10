from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

class SimpleNN():
    def build(self, maxlen):
        self.model = Sequential()
        self.model.add(Embedding(10000, 8, input_length=maxlen))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation="sigmoid"))
    
    def train(self, train_data, train_label, maxlen, epochs=5, batch_size=64):
        self.build(maxlen)
        self.model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
        self.model.summary()
        self.model.fit(train_data, train_label, epochs=10, batch_size=32, validation_split=0.2)

