import time
import numpy as np
from functions import metrics_functions as metrics


def kfold_scaffold_split(data, k, train_frac=0.8):
    from functions.featurizing_functions import generate_scaffold_set
    from sklearn.utils import shuffle

    scaffold_sets = generate_scaffold_set(data['smiles'])

    train_indices, valid_indices = [], []

    for fold in range(k):
        train_idx, valid_idx = [], []
        len1 = []
        for scaffold in scaffold_sets:
            if len(scaffold) < 2:
                len1 += scaffold
            else:
                cutoff = int(train_frac * len(scaffold))
                scaffold = shuffle(scaffold, random_state=fold)
                train_idx += scaffold[:cutoff]
                valid_idx += scaffold[cutoff:]
        len1 = shuffle(len1, random_state=fold)
        cutoff = int(train_frac * len(len1))
        train_idx += len1[:cutoff]
        valid_idx += len1[cutoff:]

        train_indices.append(train_idx)
        valid_indices.append(valid_idx)

    return train_indices, valid_indices


def cv_random_nn(model, X, Y, epochs, batch_size, num_split):
    print("Random Splitting of %s folds" % num_split)
    from sklearn.model_selection import ShuffleSplit

    shuffle = ShuffleSplit(n_splits=num_split, test_size=0.2, random_state=7)
    accuracy, fitting_time = 0.0, 0.0
    mac_precision, mac_recall, mac_f1, mac_roc_auc = 0.0, 0.0, 0.0, 0.0
    mic_precision, mic_recall, mic_f1, mic_roc_auc = 0.0, 0.0, 0.0, 0.0
    hamming = 0.0

    for fold, (train_idx, val_idx) in enumerate(shuffle.split(X)):
        x_train, x_val, y_train, y_val = X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]
        ml = model

        print("=== Split %s ===" % (fold + 1))
        start = time.perf_counter()

        history = ml.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
        print("Train accuracy per epoch:", history.history['acc'])
        print("Train loss per epoch:", history.history['loss'])
        print("Val accuracy per epoch:", history.history['val_acc'])
        print("Val loss per epoch:", history.history['val_loss'])

        end = time.perf_counter()

        pred_train_proba = ml.predict(x_train)
        pred_val_proba = ml.predict(x_val)

        pred_train = (pred_train_proba == pred_train_proba.max(axis=1, keepdims=1)).astype(float)
        pred_val = (pred_val_proba == pred_val_proba.max(axis=1, keepdims=1)).astype(float)

        acc_, mac_p_, mac_r_, mac_f1_, mac_auc_, mic_p_, mic_r_, mic_f1_, mic_auc_, hamming_ = metrics.performance_metrics(y_val, pred_val, y_train=y_train, pred_train=pred_train)

        accuracy += float(acc_) / num_split

        mac_precision += float(mac_p_) / num_split
        mac_recall += float(mac_r_) / num_split
        mac_f1 += float(mac_f1_) / num_split
        mac_roc_auc += float(mac_auc_) / num_split

        mic_precision += float(mic_p_) / num_split
        mic_recall += float(mic_r_) / num_split
        mic_f1 += float(mic_f1_) / num_split
        mic_roc_auc += float(mic_auc_) / num_split

        hamming += float(hamming_) / num_split

        fitting_time += (end - start) / num_split
        print("Fitting time", (end - start), "\n")

    print("===== Average results over %s splits =====" % num_split)
    print("Accuracy : %f" % accuracy)
    print("Macro Precision:", mac_precision)
    print("Macro Recall:", mac_recall)
    print("Macro F1 score:", mac_f1)
    print("Macro ROC AUC Score:", mac_roc_auc)
    print("Micro Precision:", mic_precision)
    print("Micro Recall:", mic_recall)
    print("Micro F1 score:", mic_f1)
    print("Micro ROC AUC Score:", mic_roc_auc)
    print("Hamming Loss:", hamming)
    print("Average time taken: %f" % fitting_time)
    print("==========================================")

    return ml


def cv_ml(model, X, Y, num_split):
    print("Random Splitting of %s folds" % num_split)
    from sklearn.model_selection import ShuffleSplit

    shuffle = ShuffleSplit(n_splits=num_split, test_size=0.2, random_state=7)
    accuracy, fitting_time = 0.0, 0.0
    mac_precision, mac_recall, mac_f1, mac_roc_auc = 0.0, 0.0, 0.0, 0.0
    mic_precision, mic_recall, mic_f1, mic_roc_auc = 0.0, 0.0, 0.0, 0.0
    hamming = 0.0

    for fold, (train_idx, val_idx) in enumerate(shuffle.split(X)):
        x_train, x_val, y_train, y_val = X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]
        ml = model

        print("=== Split %s ===" % (fold + 1))
        start = time.perf_counter()

        ml = ml.fit(x_train, y_train)

        end = time.perf_counter()

        pred_train_proba = ml.predict_proba(x_train)
        pred_val_proba = ml.predict_proba(x_val)

        if len(pred_train_proba) == 3 and len(pred_val_proba) == 3:
            pred_train_proba = np.stack([pred_train_proba[0][:, 1], pred_train_proba[1][:, 1], pred_train_proba[2][:, 1]], axis=1)
            pred_val_proba = np.stack([pred_val_proba[0][:, 1], pred_val_proba[1][:, 1], pred_val_proba[2][:, 1]], axis=1)

        pred_train = (pred_train_proba == pred_train_proba.max(axis=1, keepdims=1)).astype(float)
        pred_val = (pred_val_proba == pred_val_proba.max(axis=1, keepdims=1)).astype(float)

        acc_, mac_p_, mac_r_, mac_f1_, mac_auc_, mic_p_, mic_r_, mic_f1_, mic_auc_, hamming_ = metrics.performance_metrics(y_val, pred_val, y_train=y_train, pred_train=pred_train)

        accuracy += float(acc_) / num_split

        mac_precision += float(mac_p_) / num_split
        mac_recall += float(mac_r_) / num_split
        mac_f1 += float(mac_f1_) / num_split
        mac_roc_auc += float(mac_auc_) / num_split

        mic_precision += float(mic_p_) / num_split
        mic_recall += float(mic_r_) / num_split
        mic_f1 += float(mic_f1_) / num_split
        mic_roc_auc += float(mic_auc_) / num_split

        hamming += float(hamming_) / num_split

        fitting_time += (end - start) / num_split
        print("Fitting time", (end - start), "\n")

    print("===== Average results over %s splits =====" % num_split)
    print("Accuracy : %f" % accuracy)
    print("Macro Precision:", mac_precision)
    print("Macro Recall:", mac_recall)
    print("Macro F1 score:", mac_f1)
    print("Macro ROC AUC Score:", mac_roc_auc)
    print("Micro Precision:", mic_precision)
    print("Micro Recall:", mic_recall)
    print("Micro F1 score:", mic_f1)
    print("Micro ROC AUC Score:", mic_roc_auc)
    print("Hamming Loss:", hamming)
    print("Average time taken: %f" % fitting_time)
    print("==========================================")

    return ml


def save_model(model, filename, neural_network):
    """
    Save model to disk
    :param model: model to be saved
    :param filename: (string) filename to save the model to
    :param neural_network: (boolean) whether the model is a neural network model (keras) or conventional machine learning model (scikit-learn).
    :return: None
    """
    from functions import general_functions as general
    general.check_path_exists(filename)

    print("\nSaving model to '%s' ..." % filename)
    if neural_network:
        model.save(filename)
    else:
        #    import pickle
        #    pickle.dump(model, open(filename, 'wb'))
        from sklearn.externals import joblib
        joblib.dump(model, filename)
    print("Saving model done.")


def load_model(filepath, neural_network):
    """
    Load model from disk
    :param filepath: (string) file path to load model from
    :param neural_network: (boolean) whether the model is a neural network model (keras) or conventional machine learning model (scikit-learn).
    :return: Loaded model
    """
    if neural_network:
        from keras.models import load_model
        model = load_model(filepath)
    else:
        # import pickle
        # model = pickle.load(open(filepath, 'rb'))
        from sklearn.externals import joblib
        model = joblib.load(filepath)
    print("Loading model done.")
    return model


def plot_roc_curve(fpr, tpr, aucs, tprs, image_name):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import auc
    from functions import general_functions as general

    general.check_path_exists(image_name)
    plt.switch_backend('agg')

    i = 0
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    for fpr_, tpr_, roc_auc in zip(fpr, tpr, aucs):
        i += 1
        plt.plot(fpr_, tpr_, lw=1, alpha=0.3)  # ,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_fpr = np.linspace(0, 1, 100)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of %s' % image_name[image_name.rfind('/') + 1:image_name.rfind("dataset") - 1])
    plt.legend(loc="lower right")
    plt.savefig(image_name)


def nn_training_history(model, X, Y, epochs, batch_size, random_state=0):
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from sklearn.metrics import accuracy_score

    start = time.perf_counter()

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    train_acc, train_loss, val_acc, val_loss = [], [], [], []
    for epoch in range(epochs):
        x_train, y_train = shuffle(x_train, y_train, random_state=epoch)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=batch_size, verbose=0)

        pred_train = model.predict(x_train)
        pred_val = model.predict(x_val)

        pred_train = (pred_train == pred_train.max(axis=1, keepdims=1)).astype(float)
        pred_val = (pred_val == pred_val.max(axis=1, keepdims=1)).astype(float)

        train_acc.append(accuracy_score(y_train, pred_train))
        train_loss.extend(history.history['loss'])
        val_acc.append(accuracy_score(y_val, pred_val))
        val_loss.extend(history.history['val_loss'])

    print("Train accuracy per epoch:", train_acc)
    print("Train loss per epoch:", train_loss)
    print("Val accuracy per epoch:", val_acc)
    print("Val loss per epoch:", val_loss)

    # history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    # print("Train accuracy per epoch:", history.history['acc'])
    # print("Train loss per epoch:", history.history['loss'])
    # print("Val accuracy per epoch:", history.history['val_acc'])
    # print("Val loss per epoch:", history.history['val_loss'])

    end = time.perf_counter()
    print("Fitting time: ", end - start)

    return model
