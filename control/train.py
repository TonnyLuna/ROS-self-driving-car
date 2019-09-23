import pandas as pd
import numpy as np
import tensorflow as tf
import argparse, os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Lambda, Convolution2D, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, LSTM
from keras.layers.wrappers import TimeDistributed
from keras import metrics, backend

from keras.optimizers import Adam # base
#from keras.optimizers import AdamNoise # proposal

from rnn_utils import INPUT_SHAPE, STEPS,batch_generator



np.random.seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def relu_index(index, top, bottom):
    if index > top:
        return top
    elif index < bottom:
        return bottom
    else:
        return index


def time_distributed(X_train, X_valid):
    t_X_train = np.empty([X_train.shape[0], STEPS], dtype=object)
    t_X_valid = np.empty([X_valid.shape[0], STEPS], dtype=object)
    top = X_train.shape[0]
    for i in range(0, X_train.shape[0]):
        for j in range(0, STEPS):
            t_X_train[i][j] = X_train[relu_index(i-(STEPS-(j+1)), top, 0)]

    for i in range(0, X_valid.shape[0]):
        for j in range(0, STEPS):
            t_X_valid[i][j] = X_valid[relu_index(i-(STEPS-(j+1)), top, 0)]

    return t_X_train, t_X_valid


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'steering.csv'))
    data_df['steering'] = pd.to_numeric(data_df['steering'], errors='coerce')
    X = data_df['image'].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    X_train, X_valid = time_distributed(X_train, X_valid)
    print "Training dataset: ", X_train.shape
    return X_train, X_valid, y_train, y_valid



def show_shapes(model):
    print("======================================")
    print("INPUT SHAPE")
    for layer in model.layers:
        print(layer.input_shape)
    print("======================================")

    print("======================================")
    print("OUTPUT SHAPE")
    for layer in model.layers:
        print(layer.output_shape)
    print("======================================")

    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png')


def build_model(args):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(24, (5, 5)), input_shape=INPUT_SHAPE))
    model.add(TimeDistributed(Conv2D(32, 5, 5, activation='elu', subsample=(2, 2))))
    model.add(TimeDistributed(Conv2D(48, 3, 3, activation='elu', subsample=(2, 2))))
    model.add(TimeDistributed(Conv2D(64, 3, 3, activation='elu', subsample=(2, 2))))
    model.add(TimeDistributed(Conv2D(128, 3, 3, activation='elu', subsample=(1, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=256, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1))

    show_shapes(model)
    return model



def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    from time import gmtime, strftime
    csv_logger = CSVLogger(strftime("%Y-%m-%d %H:%M:%S", gmtime())+'.csv', append=True, separator=',')


    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate), metrics=['mse', 'logcosh', 'cosine_proximity']) # base
    #model.compile(loss='mean_squared_error', optimizer=AdamNoise(eta=0.01, gamma=0.55, lr=args.learning_rate),  metrics=['mse', 'logcosh', 'cosine_proximity']) # proposal


    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint, csv_logger],
                        verbose=1)




def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-p', help='Optimizer',             dest='opt',               type=str,   default='Adam')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='../database')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=15)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=10000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-5)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
