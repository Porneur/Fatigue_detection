from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
class RollModel(Model):
    def __init__(self, in_shape):
        super(RollModel, self).__init__(in_shape)
        self.in_shape = in_shape
        self.lstm1 = layers.LSTM(units = 50, input_shape = self.in_shape, return_sequences = True)
        self.lstm2 = layers.LSTM(units = 50, return_sequences = True)
        #self.lstm3 = layers.LSTM(units = 50, return_sequences = True)
        #self.lstm4 = layers.LSTM(units = 50, return_sequences = True)
        #self.lstm5 = layers.LSTM(units = 50, return_sequences = True)
        #self.lstm6 = layers.LSTM(units = 50, return_sequences = True)
        #self.lstm7 = layers.LSTM(units = 50, return_sequences = True)
        #self.lstm8 = layers.LSTM(units = 50, return_sequences = True)
        self.lstm9 = layers.LSTM(units = 50)
        self.d1 = layers.Dense(100, activation='relu')
        self.drop = layers.Dropout(0.5)
        self.d2 = layers.Dense(1)

    def call(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        #x = self.lstm3(x)
        #x = self.lstm4(x)
        #x = self.lstm5(x)
        #x = self.lstm6(x)
        #x = self.lstm7(x)
        #x = self.lstm8(x)
        x = self.lstm9(x)
        x = self.d1(x)
        x = self.drop(x)
        x = self.d2(x)
        return x
    
    
    def build_graph(self):
        x = Input(shape=self.in_shape)
        return Model(x,outputs = self.call(x))












