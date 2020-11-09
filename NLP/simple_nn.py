from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

def SimpleNN():
    def build(self):
        self.model = Sequential()
        self.model.add(Embedding(10000, 8, input_length=maxlen))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation="sigmoid"))
    
    def train(self, train_data, train_label, epochs=5, batch_size=64):
        self.model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
        self.model.summary()
        self.model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

