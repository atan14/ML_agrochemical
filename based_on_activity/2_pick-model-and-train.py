import numpy as np
import argparse
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

from functions import model_functions as model_func
from functions import general_functions as general

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='model to train')
parser.add_argument('featurizer', type=str, help='type of descriptor (choice: rdk, ecfp)')
parser.add_argument('split_type', type=str, help='for loading testing sets created using different splitting methods (choice: random or scaffold)')
args = parser.parse_args()

# Check if inputs are available
if args.featurizer not in ['rdk', 'ecfp']:
    raise Exception("Descriptor %s not available. Choose from rdk or ecfp." % args.featurizer)
if args.model not in ['NN', 'RF', 'GB', 'KNN']:
    raise Exception("Model %s not available." % args.model)

# Import data
temp_path = os.path.join(os.getcwd(), 'data/ft_train_%s.pkl' % args.split_type)
if os.path.exists(temp_path):
    data = general.import_pandas_dataframe(temp_path)
    print("Shape of data:", data.shape)

X = np.stack(data[args.featurizer])
Y = LabelBinarizer().fit_transform((data['agrochemical']))

X, Y = shuffle(X, Y, random_state=0)

if args.model == 'NN':
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, BatchNormalization
    from keras.optimizers import Adam
    from keras import regularizers

    reg = regularizers.l1(5e-07)
    epochs = 15
    input_dropout_rate = 0.1
    dropout_rate = 0.3
    lr = 0.0001
    batch_size = 16

    model = Sequential()
    model.add(Dropout(input_dropout_rate, input_shape=(X.shape[1],)))
    model.add(Dense(512, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(Y.shape[1], kernel_regularizer=reg))
    model.add(Activation('sigmoid'))

    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)

elif args.model == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier

    nn = 3
    weights = 'distance'
    model = KNeighborsClassifier(n_neighbors=nn, weights=weights, n_jobs=-1)

    model.fit(X, Y)

elif args.model == 'GB':
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier

    lr = 0.2
    num_trees = 500
    depth = 3

    gb = GradientBoostingClassifier(learning_rate=lr, n_estimators=num_trees, max_depth=depth, random_state=0)
    model = OneVsRestClassifier(gb)

    model.fit(X, Y)

elif args.model == 'RF':
    from sklearn.ensemble import RandomForestClassifier

    depth = 5
    num_trees = 200

    model = RandomForestClassifier(n_estimators=num_trees, max_depth=depth)

    model.fit(X, Y)

neural_network = 0
if args.model == 'NN':
    neural_network = 1
save_model_path = "./best_models/%s_%s_%s.h" % (args.model, args.featurizer, args.split_type)
general.check_path_exists(save_model_path)
model_func.save_model(model, save_model_path, neural_network=neural_network)

print("%s model saved to %s" % (args.model, save_model_path))
print("\nExiting program.")
