import argparse
import os
import numpy as np
import sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier

from functions import model_functions as model_func
from functions import general_functions as general

parser = argparse.ArgumentParser()
parser.add_argument('featurizer', type=str, help='type of descriptor (choice: rdk, ecfp)')
parser.add_argument('--num_trees', type=int, default=500, help='Number of estimators')
parser.add_argument('--max_depth', type=int, default=3, help='maximum depth of individual regression estimators')
parser.add_argument('--learning_rate', type=float, default=0.1, help='step size')
parser.add_argument('--max_features', type=str, default='None', help='fraction of features used (choice: auto, sqrt, log2, None)')
parser.add_argument('--num_split', type=int, default=10,
                    help='Number of splits for averaging of classification performance.')
parser.add_argument('--seed', type=int, default=7,
                    help='seed number for random shuffling.')
parser.add_argument('--write_to_output', type=bool, default=True, help='whether to write to output file.')
args = parser.parse_args()

# Check if inputs are available
if args.featurizer not in ['rdk', 'ecfp']:
    raise Exception("Descriptor %s not available. Choose from rdk or ecfp." % args.featurizer)
if args.max_features not in ['auto', 'sqrt', 'log2', 'None']:
    raise Exception("Input %s for max features not available." % args.max_features)
if args.max_features == 'None':
    args.max_features = None


filename = "GB/%s/%s_maxft/%strees_%sdepth_%slr_random" % (args.featurizer, args.max_features, args.num_trees, args.max_depth, args.learning_rate)
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
gb = GradientBoostingClassifier(learning_rate=args.learning_rate, n_estimators=args.num_trees, max_depth=args.max_depth, max_features=args.max_features, random_state=args.seed)

model = OneVsRestClassifier(gb)

print("Featurizer: %s" % args.featurizer)
print("\n=================================")
print("Gradient Boosting Model")
print("=================================\n")
print(model.estimator)
print()

model_func.cv_ml(model, X, Y, args.num_split)


general.check_path_exists(save_model_path)
model_func.save_model(model, save_model_path, neural_network=False)

print("\nExiting program.")

sys.stdout.close()
