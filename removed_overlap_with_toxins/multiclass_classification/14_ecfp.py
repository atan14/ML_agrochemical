import argparse
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from functions import model_functions as model_func
from functions import general_functions as general
from functions import featurizing_functions as ft


parser = argparse.ArgumentParser(description='Training using compound properties features.')

parser.add_argument('dataset', type=str,
                    help='dataset to be used (choice: dataset1.pkl, dataset2.pkl)')
parser.add_argument('method', type=str,
                    help='ML methods to use for training (choice: [simplenn, randomforest, '
                         'knearest, gradientboosting, svm, convnn])')
parser.add_argument('--num_split', type=int, default=10,
                    help='Number of splits for averaging of multiclass classification performance.')
parser.add_argument('--seed', type=int, default=7,
                    help='seed number for random shuffling.')
parser.add_argument('--save_model', type=bool, default=False,
                    help='Whether trained model is saved.')
parser.add_argument('--model_param', nargs='*', default=0,
                    help='Input parameter for model used [IMPORTANT: ORDERING OF INPUT PARAMETER '
                         'MATTERS](see function define_model for better understanding).')
parser.add_argument('--filename_append', type=str, default='',
                    help='Filename extension to distinct runs from each other.')
parser.add_argument('--layer_dim', type=int, default=3)
parser.add_argument('--loss', type=str, default='binary_crossentropy')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--epochs', type=int, default=0)
args = parser.parse_args()


# Import data
temp_path = general.file_pathway(args.dataset)
if os.path.exists(temp_path):
    data = general.import_pandas_dataframe(temp_path)
    print("Shape of data:", data.shape)
else:
    raise Exception("%s does not exist." % args.dataset)

if not 'ecfp' in data.columns:
    print ("Calculating ECFP...")
    data['ecfp'] = data['mol'].apply(ft.get_ecfp)
    print ("ECFP calculation done.")

# Input (x) and class (y)
X = data['ecfp']
X = np.stack(X)

mlb = MultiLabelBinarizer().fit(data['agrochemical'])
Y = mlb.transform(data['agrochemical'])

print ("Standard scaling...")
st = StandardScaler()
X = st.fit_transform(X)
print ("Standard scaling done.")


# Machine learning algorithms to use
method = args.method
model_param = args.model_param

num_split = args.num_split
seed = args.seed
save = args.save_model

epochs = 0

if method in ['simplenn']:
    if args.layer_dim == 3:
        layers_dim = [X.shape[1], 128, 8, Y.shape[1]]
        activation = ['relu', 'softmax', 'sigmoid']
    elif args.layer_dim == 4:
        layers_dim = [X.shape[1], 512, 128, 8, Y.shape[1]]
        activation = ['relu', 'tanh', 'softmax', 'sigmoid']
    elif args.layer_dim == 5:
        layers_dim = [X.shape[1], 512, 128, 16, 8, Y.shape[1]]
        activation = ['relu', 'tanh', 'softmax', 'tanh', 'sigmoid']

    epochs = model_func.plot_nn_loss_against_epoch(X, Y, layers_dim, activation, args.epochs,
                                            image_name='./image/14_ecfp/%s_%s_%s_NNLossAcc.png' % (
                                                method, args.dataset[:args.dataset.rfind('.')],args.filename_append))
    print("Number of epochs:", epochs)

    model_func.define_model(method)
    model = model_func.build_simplenn_model(layers_dim=layers_dim, activation=activation,
                                            loss=args.loss, optimizer=args.optimizer)

else:
    model = model_func.define_model(method, model_param)


model = model_func.train_model(model, num_split, seed, X, Y,
                               image_name='./image/14_ecfp/%s_%s_%s.png' %(method,
                               args.dataset[:args.dataset.rfind('.')],args.filename_append),
                               neural_network_epochs=epochs)
if save:
    save_filepath = './saved_model/14_ecfp/' + "%s_%s_%s.h" % (method, args.dataset[
                                                                    :args.dataset.rfind(
        '.')], args.filename_append)
    model_func.save_model(model, save_filepath, epochs)

print ("\nExiting program.")

