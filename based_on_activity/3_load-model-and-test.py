import numpy as np
import argparse
import os
import sys
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelBinarizer

from functions import model_functions as model_func
from functions import general_functions as general
from functions import metrics_functions as metrics

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help='path to saved model')
parser.add_argument('--split_type', type=str, default='random', help='for loading testing sets created using different splitting methods (choice: random or scaffold)')
parser.add_argument('--test', type=bool, default=True, help='whether test on testing set is desired. If not it is just for verification purpose on validation set.')
parser.add_argument('--num_split', type=int, default=None, help='for reproducing the exact results during training (only for testing on validation set). Only valid when --test is False.')
args = parser.parse_args()

if not os.path.exists(args.model_path):
    raise Exception("Pathway to model %s does not exist" % args.model_path)
if args.test:
    args.num_split = None

ml = ['GB', 'RF', 'KNN']
if any(x in args.model_path for x in ml):
    nn = False
else:
    nn = True

logfile = os.path.join(os.getcwd(), "best_models/%s.log" % args.model_path[args.model_path.find('/'):-2])
general.check_path_exists(logfile)
sys.stdout = open(logfile, 'wt')

print ("Loading model from %s..." % args.model_path)
model = model_func.load_model(args.model_path, nn)
print ("Finished loading model.")

if args.test:
    test_data_path = os.path.join(os.getcwd(), 'data/ft_test_%s.pkl' % args.split_type)
    test_data = general.import_pandas_dataframe(test_data_path)
    print("\nPrediction on testing data set of shape", test_data.shape, ": ")

    ft = None
    if 'ecfp' in args.model_path:
        ft = 'ecfp'
    elif 'rdk' in args.model_path:
        ft = 'rdk'
    x_test = np.stack(test_data[ft])
    y_test = LabelBinarizer().fit_transform((test_data['agrochemical']))

    pred_test = model.predict(x_test)
    pred_test = (pred_test == pred_test.max(axis=1, keepdims=1)).astype(float)

    _ = metrics.performance_metrics(y_test, pred_test)

else:
    data_path = os.path.join(os.getcwd(), 'data/ft_train_%s.pkl' % args.split_type)
    data = general.import_pandas_dataframe(data_path)
    print("\nPrediction on training data set of shape", data.shape, ": ")

    ft = None
    if 'ecfp' in args.model_path:
        ft = 'ecfp'
    elif 'rdk' in args.model_path:
        ft = 'rdk'
    X = np.stack(data[ft])
    Y = LabelBinarizer().fit_transform((data['agrochemical']))

    shuffle = ShuffleSplit(n_splits=args.num_split, test_size=0.2, random_state=7)
    for train_idx, val_idx in shuffle.split(X):
        train_idx, val_idx
    x_train, x_val, y_train, y_val = X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]

    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)

    pred_train = (pred_train == pred_train.max(axis=1, keepdims=1)).astype(float)
    pred_val = (pred_val == pred_val.max(axis=1, keepdims=1)).astype(float)

    _ = metrics.performance_metrics(y_val, pred_val, y_train, pred_train)

print("\nExiting program.")
sys.stdout.close()