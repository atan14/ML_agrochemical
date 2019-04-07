import argparse
import os
import numpy as np
import sys

from sklearn.preprocessing import StandardScaler

from functions import model_functions as model_func
from functions import general_functions as general
from functions import featurizing_functions as ft

parser = argparse.ArgumentParser(description='Plotting validation accuracy and loss against '
                                             'number of epochs for simple neural network model.')

parser.add_argument('featurizer', type=str, help='type of feature (choice: daylight, ecfp)')
parser.add_argument('dataset', type=str, help='dataset to be used (choice: dataset1.pkl, '
                                              'dataset2.pkl)')
parser.add_argument('num_layers', type=int, help='number of layers to use for training (choice: '
                                                 '3, 4, 5).')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--loss', type=str, default='binary_crossentropy', help='type of loss '
                                                                            'function to use')
parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer function')
args = parser.parse_args()

# Check if the parameters are available for this file
if args.featurizer not in ['daylight', 'ecfp']:
    raise Exception("Descriptor %s not available. Choose from 'daylight' or 'ecfp'."
                    % args.featurizer)
if args.num_layers not in [3, 4, 5]:
    raise Exception("Number of layers not available. Choose from 3, 4 or 5, or add them below. ")


filename = 'simplenn_epoch_image/%s_%s_%slayers.log' % \
           (args.featurizer, args.dataset[:args.dataset.rfind('.')], args.num_layers)

# check if directory exists
general.check_path_exists(filename)

sys.stdout = open(filename,'wt')

# Import data
temp_path = general.file_pathway(args.dataset)
if os.path.exists(temp_path):
    data = general.import_pandas_dataframe(temp_path)
    print("Shape of data:", data.shape)
else:
    raise Exception("%s does not exist." % args.dataset)

# Featurization
if args.featurizer == 'daylight':
    print("Calculating Daylight fingerprint...")
    data['fingerprint'] = data['mol'].apply(ft.daylight_fingerprint)
    data['fingerprint'] = data['fingerprint'].apply(ft.daylight_fingerprint_padding)
    print("Daylight fingerprint calculation done.")

if args.featurizer == 'ecfp':
    print("Calculating ECFP...")
    data['fingerprint'] = data['mol'].apply(ft.get_ecfp)
    print("ECFP calculation done.")

# Input (x) and label (y)
X = data['fingerprint']
Y = data['agrochemical']

X = np.array(np.stack(X), dtype=float)
Y = np.array(Y, dtype=float)

# Standard scaling
print("Standard scaling...")
X = StandardScaler().fit_transform(X)
print("Standard scaling done.")

# Build neural network model
if args.num_layers == 3:
    layers_dim = [X.shape[1], 128, 8, 1]
    activation = ['relu', 'softmax', 'sigmoid']
elif args.num_layers == 4:
    layers_dim = [X.shape[1], 512, 128, 8, 1]
    activation = ['relu', 'tanh', 'softmax', 'sigmoid']
elif args.num_layers == 5:
    layers_dim = [X.shape[1], 512, 128, 16, 4, 1]
    activation = ['relu', 'tanh', 'softmax', 'tanh', 'sigmoid']


image_name = filename[:filename.rfind('.')] + '.png'

training_acc, training_loss, validation_acc, validation_loss = \
    model_func.plot_nn_loss_against_epoch(X, Y, layers_dim, activation, args.epochs, image_name,
                                          loss=args.loss, optimizer=args.optimizer)

print("Number of epochs for maximum training accuracy:", np.argmax(training_acc))
print("Number of epochs for minimum training loss:", np.argmin(training_loss))
print("Number of epochs for maximum validation accuracy:", np.argmax(validation_acc))
print("Number of epochs for minimum validation loss:", np.argmin(validation_loss))

print ()
print ("Training accuracy:", training_acc)
print ("Training loss:", training_loss)
print ()
print ("Validation accuracy:", validation_acc)
print ("Validation loss:", validation_loss)
print ()


print("\nExiting program.")
sys.stdout.close()
