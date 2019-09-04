
from flask import Flask
from flask import request, jsonify
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import backend as K


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


def anomaly_elimination(arr):
    arr_mean = np.mean(arr)
    arr_cleaned = []
    for i in arr:
        if i < 2 * arr_mean:
            arr_cleaned.append(i)
    return arr_cleaned


def model_train(raw_seq):
    #to float
    temp_arr = []
    for i in raw_seq:
        temp_arr.append(float(i))
    raw_seq = temp_arr.copy()
    
    #remove anomalies
    raw_seq = anomaly_elimination(raw_seq).copy()
    
    raw_seq_len = int(len(raw_seq)/3)
    #number of time steps
    n_steps = raw_seq_len
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
    model.add(Dense(raw_seq_len*3*4, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    model.fit(X, y, epochs=350, verbose=0)
    
    # prediction
    Pred = []
    full_seq = raw_seq.copy()
    for i in range(0,raw_seq_len):
        x_input = array(full_seq[-n_steps:].copy())
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)[0][0]
        Pred.append(float(yhat))
        full_seq.append(yhat)
		
    K.clear_session()
        
    return jsonify({"prediction":Pred,
                    "data":temp_arr})



@app.route('/', methods=['POST'])
def result():
    req_data = request.get_json()
    #y = json.dumps(req_data)
    res = model_train(req_data["data"])
    return res


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    
    
    
"""
http://0.0.0.0:80

{
	"data": [1,2,3,4,5,6,7,8,9]
}
"""
