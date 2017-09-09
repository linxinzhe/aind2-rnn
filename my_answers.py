import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series)-window_size):
        x=series[i:i+window_size]
        label=series[i+window_size]
        X.append(x)
        y.append(label)
    

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size,1)))
    model.add(Dense(1,activation="linear"))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    english_characters=['a','b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's','t', 'u', 'v', 'w', 'x', 'y', 'z',' ']
    allowed=punctuation+english_characters
    
    cleaned_text=[c for c in text if c in allowed]
        
    return "".join(cleaned_text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    #full_steps_text = text[:len(text) // (step_size*window_size) * (step_size*window_size)]
    full_steps_text=text
    
    i=0
    while((i+window_size) < len(full_steps_text)):
        x = full_steps_text[i:i + window_size]
        y = full_steps_text[i + window_size]
        i+=step_size
        inputs.append(x)
        outputs.append(y)

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    return model

