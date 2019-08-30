
from flask import Flask
from flask import request, jsonify
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


app = Flask(__name__)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def model_train():
    raw_seq = [1,2,3,4,5,6,7,8,9]
    #number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=1000, verbose=0)
    # prediction
    #input_length = int(len(raw_seq)/2)
    x_input = array(raw_seq[-n_steps:].copy())
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    return yhat


@app.route('/', methods=['GET'])
def result():
    #req_data = request.get_json()
    #y = json.dumps(req_data)
    res = model_train()
    return str(res)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    #app.run()
    
    
    
"""
http://127.0.0.1:5000

{
	"data": [1,2,3,4,5,6,7,8,9]
}
"""
