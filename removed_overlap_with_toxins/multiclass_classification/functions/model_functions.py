def build_simplenn_model(layers_dim=[2048, 128, 8, 1], activation=['relu', 'softmax', 'sigmoid'],
                         loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
    if len(layers_dim) - 1 != len(activation):
        raise Exception("Number of layers and activation functions must match!")

    import keras
    model = keras.models.Sequential()
    for i in range(len(activation)):
        model.add(keras.layers.Dense(layers_dim[i + 1], input_dim=layers_dim[i],
                                     activation=activation[i]))

    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.summary()

    return model


def define_model(method, argv=0):
    """
    Function to define model.
    :param method: (string) model that you want defined.
    :param argv: (list) hyperparameters of model (optional).
    :return: model
    """
    if method == 'simplenn':

        print("\n=================================")
        print("Simple Neural Network Model")
        print("=================================\n")

        return

    elif method == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier

        if argv:
            n_estimators = int(argv[0])
            max_depth = int(argv[1])
        else:
            n_estimators = 200
            max_depth = 1

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=0,
                                       n_jobs=-1)

        print("\n=================================")
        print("Random Forest Classifier Model")
        print("=================================\n")

        print("Num estimators: %s, max depth: %s \n" % (n_estimators, max_depth))

    elif method == 'gradientboosting':
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.multiclass import OneVsRestClassifier

        if argv:
            n_estimators = int(argv[0])
            learning_rate = float(argv[1])
            max_depth = int(argv[2])
        else:
            n_estimators = 200
            learning_rate = 1.0
            max_depth = 1

        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           max_depth=max_depth, random_state=0)
        model = OneVsRestClassifier(model)

        print("\n=================================")
        print("Gradient Boosting Classifier Model")
        print("=================================\n")

        print("Num estimators: %s, max depth: %s, learning_rate: %s \n" % (n_estimators, max_depth,
                                                                           learning_rate))
    elif method == 'knearest':
        from sklearn.neighbors import KNeighborsClassifier

        if argv:
            n_neighbors = int(argv[0])
            weights = str(argv[1])
            algorithm = str(argv[2])
        else:
            n_neighbors = 3
            weights = 'distance'
            algorithm = 'brute'

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

        print("\n=================================")
        print("K Nearest Classifier Model")
        print("=================================\n")

        print(
            "Num neighbors: %s, weights: %s, algorithm: %s \n" % (n_neighbors, weights, algorithm))

    elif method == 'svm':
        from sklearn.svm import SVC

        if argv:
            kernel = str(argv[0])
            gamma = float(argv[1])
        else:
            kernel = 'rbf'
            gamma = 0.1

        model = SVC(kernel=kernel, gamma=gamma)

        print("\n=================================")
        print("SVM Model")
        print("=================================\n")

        print("Kernel: %s, gamma: %s \n" % (kernel, gamma))

    else:
        raise Exception("Method '%s' does not exist." % method)

    return model


def train_model(model, num_split, seed, X, Y, image_name, neural_network_epochs=0):
    """
    Train input model and obtain averaged results.
    :param model: Input model for training.
    :param num_split: (int) Number of splits for averaging of performance metrics.
    :param seed: (int) Seed for random state.
    :param X: (numpy array) Training input.
    :param Y: (numpy array) Class label.
    :param neural_network_epochs: (int) whether the model is a neural network model (keras) or
    conventional machine learning model (scikit-learn) & at the same time used to specify number
    of epochs.
    :return: Trained model
    """
    from sklearn.model_selection import ShuffleSplit
    import numpy as np
    import time

    shuffle = ShuffleSplit(n_splits=num_split, random_state=seed, test_size=0.2)
    accuracy, fitting_time = 0.0, 0.0
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    i = 0

    for train_idx, test_idx in shuffle.split(X):
        i += 1
        print ("== Split %s ==" %i)
        start = time.perf_counter()

        x_train, x_test, y_train, y_test = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

        ml = model

        if neural_network_epochs:
            ml.fit(x_train, y_train, epochs=neural_network_epochs, batch_size=32, verbose=0)
        else:
            ml.fit(x_train, y_train)

        pred_train = ml.predict(x_train)
        pred_test = ml.predict(x_test)

        pred_train = np.around(pred_train)
        pred_test = np.around(pred_test)

        end = time.perf_counter()

        acc_, prec_, rec_, f1_ = performance_metrics(y_train, y_test, pred_train, pred_test)

        accuracy += float(acc_) / num_split
        precision += float(prec_) / num_split
        recall += float(rec_) / num_split
        f1 += float(f1_) / num_split

        fitting_time += (end - start) / num_split
        print ("Fitting time", (end-start), "\n")

    # Plotting ROC curve
    plot_roc_curve(y_test, pred_test, image_name)

    print ("===== Average results over %s splits =====" % num_split)
    print ("Accuracy : %f" % accuracy)
    print ("Precision:", precision)
    print ("Recall:", recall)
    print ("F1 score:", f1)
    print ("Average time taken: %f" % fitting_time)
    print ("==========================================")

    return ml


def plot_roc_curve(y_test, pred_test, image_name):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    label = {0: "herbicide", 1: "insecticide", 2: "nematicide", 3: "fungicide", 4:"toxin"}
    for i in range(y_test.shape[1]):
        plt.plot(fpr[i], tpr[i],
                 lw=lw, label='ROC curve of %s (area = %0.2f)' % (label[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of %s' % image_name[image_name.rfind('/')+1:image_name.rfind("dataset")-1])
    plt.legend(loc="lower right")
    plt.savefig(image_name)
    plt.show()


def save_model(model, filename, neural_network):
    """
    Save model to disk
    :param model: model to be saved
    :param filename: (string) filename to save the model to
    :param neural_network: (boolean) whether the model is a neural network model (keras) or conventional machine learning model (scikit-learn).
    :return: None
    """
    import os
    if not os.path.isdir(filename[:filename.rfind('/')]):
        print ("Creating directory '%s'" % filename[:filename.rfind('/')])
        os.mkdir(filename[:filename.rfind('/')])

    print ("\nSaving model to '%s' ..." % filename)
    if neural_network:
        model.save(filename)
    else:
        import pickle
        pickle.dump(model, open(filename, 'wb'))
    print ("Saving model done.")


def load_model(filepath, neural_network):
    """
    Load model from disk
    :param filepath: (string) file path to load model from
    :param neural_network: (boolean) whether the model is a neural network model (keras) or
    conventional machine learning model (scikit-learn).
    :return: Loaded model
    """
    print ("Loading model from %s ..." % filepath)
    if neural_network:
        from keras.models import load_model
        model = load_model(filepath)
    else:
        import pickle
        model = pickle.load(open(filepath, 'rb'))
    print ("Loading model done.")
    return model


def performance_metrics(y_train, y_test, pred_train, pred_test):
    """
    Evaluate performance of model using various binary classification metrics.
    :param y_train: True training labels.
    :param y_test: True testing labels.
    :param pred_train: Predicted training labels.
    :param pred_test: Predicted testing labels.
    :return: Score of true negative, false positive, false negative, true positive and accuracy of model prediction.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        average_precision_score, roc_auc_score

    print ("Training set report:")
    print ("True class count: ", y_train.sum(axis=0))
    print ("Predicted class count: ", pred_train.sum(axis=0))
    print ("Accuracy score:", accuracy_score(y_train, pred_train))
    print ("Precision score:", precision_score(y_train, pred_train, average='macro'))
    print ("Recall score:", recall_score(y_train, pred_train, average='macro'))
    print ("Average precision score:", average_precision_score(y_train, pred_train,
                                                               average='macro'))
    print ("ROC AUC score:", roc_auc_score(y_train, pred_train, average='macro'))
    print ("F1 score:", f1_score(y_train, pred_train, average='macro'))
    print ()
    print ("Testing set report:")
    print("True class count: ", y_test.sum(axis=0))
    print("Predicted class count: ", pred_test.sum(axis=0))
    print("Accuracy score:", accuracy_score(y_test, pred_test))
    print("Precision score:", precision_score(y_test, pred_test, average='macro'))
    print("Recall score:", recall_score(y_test, pred_test, average='macro'))
    print("Average precision score:", average_precision_score(y_test, pred_test,
                                                              average='macro'))
    print("ROC AUC score:", roc_auc_score(y_test, pred_test, average='macro'))
    print("F1 score:", f1_score(y_test, pred_test, average='macro'))
    print()

    accuracy = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test, average='macro')
    recall = recall_score(y_test, pred_test, average='macro')
    f1 = f1_score(y_test, pred_test, average='macro')

    return accuracy, precision, recall, f1


def plot_nn_loss_against_epoch(X, Y, layers_dim, activation, epochs, image_name):
    import matplotlib.pyplot as plt
    import numpy as np

    model = build_simplenn_model(layers_dim=layers_dim, activation=activation)

    H = model.fit(X, Y, epochs=epochs, batch_size=16, verbose=0, validation_split=0.2, shuffle=True)

    plt.switch_backend("agg")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
    plt.title("Loss and Accuracy against Epochs")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="best")
    plt.savefig(image_name)

    min_loss_epoch = np.argmin(H.history["val_loss"])
    max_acc_epoch = np.argmax(H.history["val_acc"])

    return max(min_loss_epoch, max_acc_epoch)
