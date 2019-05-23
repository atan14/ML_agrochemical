import argparse
import os
import numpy as np
import sys
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras import regularizers
from keras.optimizers import Adam

from functions import model_functions as model_func
from functions import general_functions as general

parser = argparse.ArgumentParser()
parser.add_argument('featurizer', type=str, help='type of descriptor (choice: rdk, ecfp)')
parser.add_argument('--kernel_regularizer', type=str, default='None', help='Choice of weight decay (choice: l1, l2, l1_l2, None)')
parser.add_argument('--regularizer_param', type=float, default=0.00, help='Regularization parameter lambda')
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--loss', type=str, default='binary_crossentropy')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=7,
                    help='seed number for random shuffling.')
args = parser.parse_args()

# Check if inputs are available
if args.featurizer not in ['rdk', 'ecfp']:
    raise Exception("Descriptor %s not available. Choose from rdk or ecfp." % args.featurizer)
if args.kernel_regularizer not in ['l1', 'l2', 'l1_l2', 'None']:
    raise Exception("Kernel regularizer %s not available. Choose from l1, l2, l1_l2 or None" % args.regularizer_param)

lr = 0.0001

filename = "NN_training_history/%s/%s/batchsize%s/%s_epochs%s_dropout%s_lr%s_random_2" % (args.featurizer, args.kernel_regularizer, args.batch_size, args.regularizer_param, args.epochs, args.dropout_rate, lr)
logfile = "./logfile/" + filename + ".log"
save_model_path = "./saved_model/" + filename + ".h"

general.check_path_exists(logfile)
sys.stdout = open(logfile, 'wt')

# Import data
temp_path = os.path.join(os.getcwd(), 'data/ft_train_random.pkl')
if os.path.exists(temp_path):
    data = general.import_pandas_dataframe(temp_path)
    print("Shape of data:", data.shape)
else:
    raise Exception("%s does not exist." % temp_path)

# Input and target
X = np.stack(data[args.featurizer])
Y = LabelBinarizer().fit_transform((data['agrochemical']))


# Build model
r = args.regularizer_param
reg_dict = {'l1': regularizers.l1(r), 'l2': regularizers.l2(r), 'l1_l2': regularizers.l1_l2(l1=r, l2=r), 'None': None}
regularizer = reg_dict[args.kernel_regularizer]

input_dropout_rate = 0.1
if args.dropout_rate == 0.0:
    input_dropout_rate = 0.0

a = tf.placeholder(dtype=tf.float32, shape=(None, 2048))

model = Sequential()
model.add(Dropout(input_dropout_rate, input_shape=(X.shape[1],)))
model.add(Dense(512, kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(args.dropout_rate))
model.add(Dense(128, kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(args.dropout_rate))
model.add(Dense(32, kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(args.dropout_rate))
model.add(Dense(8, kernel_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(Y.shape[1], kernel_regularizer=regularizer))
model.add(Activation('sigmoid'))

adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.0)
model.compile(loss=args.loss, optimizer=adam, metrics=['accuracy'])

output = model(a)

print("Featurizer: %s" % args.featurizer)
print("\n=================================")
print("Simple Neural Network Model")
print("=================================\n")

model.summary()

print("Kernel regularizer: %s %s" % (args.kernel_regularizer, args.regularizer_param))
print("Dropout rate: %s" % args.dropout_rate)
print("Loss function: %s" % args.loss)
print("Optimizer functions: %s" % args.optimizer)
print("Number of epochs: %s" % args.epochs)
print("Learning rate: %s" % lr)
print ()

model = model_func.nn_training_history(model, X, Y, args.epochs, args.batch_size)


general.check_path_exists(save_model_path)
model_func.save_model(model, save_model_path, neural_network=1)

print("\nExiting program.")

sys.stdout.close()

