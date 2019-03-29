import pandas as pd
import os
import argparse

from functions import featurizing_functions as ft
from functions import general_functions as general

parser = argparse.ArgumentParser(description='Featurize data and save into files.')
parser.add_argument('dataset', type=str, help='Dataset to be used (choice: dataset1, dataset2)')
parser.add_argument('descriptor', type=str, help='Type of descriptor to featurize compounds (choice: daylight, ecfp)')
args = parser.parse_args()

# Import data
temp_path = general.file_pathway(args.dataset, 'pkl')
if os.path.exists(temp_path):
    data = general.import_pandas_dataframe(temp_path)
    print("Shape of data:", data.shape)
else:
    raise Exception("%s does not exist." % args.dataset)

if args.descriptor == 'daylight':
    print("Daylight molecular fingerprint featurization...")
    data['fingerprint'] = data['mol'].apply(ft.daylight_fingerprint)
    data['fingerprint'] = data['fingerprint'].apply(ft.daylight_fingerprint_padding)
    print("Molecular fingerprint featurization done.")
elif args.descriptor == 'ecfp':
    print ("ECFP calculation...")
    data['ecfp'] = data['mol'].apply(ft.get_ecfp)
    print ("ECFP calculation done.")
else:
    raise Exception('No such descriptor.')

print("Saving dataset...")
data.to_pickle('./data/%s.pkl' % args.dataset)

print("Exiting program.")
