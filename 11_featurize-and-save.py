import pandas as pd
import os
import argparse

from functions.featurizing_functions import *
from functions.general_functions import *

parser = argparse.ArgumentParser(description='Featurize data and save into h5 files.')

parser.add_argument('dataset', type=str, help='dataset to be used (choice: dataset1, dataset2)')
args = parser.parse_args()

# Import data
temp_path = file_pathway(args.dataset, 'pkl')
if os.path.exists(temp_path):
    data = import_pandas_dataframe(temp_path)
    print("Shape of data:", data.shape)
else:
    raise Exception("%s does not exist." % args.dataset)


print("Molecular fingerprint featurization...")
data['fingerprint'] = data['mol'].apply(fingerprint)
data['fingerprint'] = data['fingerprint'].apply(molecular_fingerprint_padding)
print("Molecular fingerprint featurization done.")


print("Saving dataset...")
from sklearn.externals import joblib
joblib.dump(data, 'data/%s.sav' % args.dataset)

print("Exiting program.")
