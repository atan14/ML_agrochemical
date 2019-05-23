def import_pandas_dataframe(pathway):
    import pandas as pd
    print("Importing data '%s'." % pathway)
    if pathway[-4:] == ".pkl":
        data = pd.read_pickle(pathway)
        data.reset_index(drop=True, inplace=True)
    elif pathway[-3:] == ".h5":
        store = pd.HDFStore(pathway)
        data = store.get(pathway[pathway.rfind('/') + 1:pathway.rfind('.')])
    else:
        raise Exception("File '%s' not found in pathway." % pathway)
    print("Done importing data.")
    return data


def check_path_exists(filename):
    import os
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
