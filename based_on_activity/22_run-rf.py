import argparse
import os
import numpy as np
import sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

from functions import model_functions as model_func
from functions import general_functions as general

parser = argparse.ArgumentParser()
parser.add_argument('featurizer', type=str, help='type of descriptor (choice: rdk, ecfp)')
parser.add_argument('--n_estimators', type=int, default=100, help='number of estimators')
parser.add_argument('--max_depth', type=int, default=3, help='maximum depth of a tree')
parser.add_argument('--max_features', type=str, default='None', help='maximum number of features')
parser.add_argument('--num_split', type=int, default=10,
                    help='Number of splits for averaging of classification performance.')
parser.add_argument('--seed', type=int, default=7,
                    help='seed number for random shuffling.')
args = parser.parse_args()

# Check if inputs are available
if args.featurizer not in ['rdk', 'ecfp']:
    raise Exception("Descriptor %s not available. Choose from rdk or ecfp." % args.featurizer)
if args.max_features not in ['auto', 'sqrt', 'log2', 'None']:
    raise Exception("Input %s for max features not available." % args.max_features)
if args.max_features == 'None':
    args.max_features = None

filename = "RF/%s/%s_maxft/%strees_%sdepth_random" % (args.featurizer, args.max_features, args.n_estimators, args.max_depth)
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
model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, max_features=args.max_features)

print("Featurizer: %s" % args.featurizer)
print("\n=================================")
print("Random Forest Classifier Model")
print("=================================\n")
print(model.get_params())
print()

model_func.cv_ml(model, X, Y, args.num_split)


general.check_path_exists(save_model_path)
model_func.save_model(model, save_model_path, neural_network=False)

print("\nExiting program.")

sys.stdout.close()
