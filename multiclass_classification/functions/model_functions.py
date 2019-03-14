def build_simplenn_model(layers_dim=[2048, 128, 8, 1], activation=['relu', 'softmax', 'sigmoid']):
    if len(layers_dim) - 1 != len(activation):
        raise Exception("Number of layers and activation functions must match!")

    import keras
    model = keras.models.Sequential()
    for i in range(len(activation)):
        model.add(keras.layers.Dense(layers_dim[i + 1], input_dim=layers_dim[i],
                                     activation=activation[i]))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_convolutional_model(feature_length):
    """
    Building convolutional network model
    :param feature_length: (int) length of input feature
    :return: keras model
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

    if feature_length < 100:
        kernel_size = 3
    else:
        kernel_size = 100

    model = Sequential()
    model.add(Conv1D(8, kernel_size, activation='relu', input_shape=(feature_length, 1)))
    # model.add(Conv1D(64, kernel_size, activation='relu'))
    model.add(MaxPooling1D(1))
    # model.add(Conv1D(128, 1, activation='relu'))
    # model.add(Conv1D(8, 1, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def define_model(method, argv=0, *kwargs):
    """
    Function to define model.
    :param method: (string) model that you want defined.
    :param argv: (list) hyperparameters of model (optional).
    :return: model
    """
    if method == 'simplenn':
        if argv:
            layers_dim = argv
            activation = kwargs[0]
        else:
            layers_dim = [15, 8, 4, 1]
            activation = ['relu', 'softmax', 'sigmoid']

        model = build_simplenn_model(layers_dim=layers_dim, activation=activation)
        print("\n=================================")
        print("Simple Neural Network Model")
        print("=================================\n")

        model.summary()

    elif method == "convnn":
        if argv:
            feature_length = int(argv)
        else:
            feature_length = 2048

        model = build_convolutional_model(feature_length)

        print("\n=================================")
        print("Convolutional Neural Network Model")
        print("=================================\n")

        model.summary()

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


def train_model(model, num_split, seed, X, Y, neural_network=0):
    """
    Train input model and obtain averaged results.
    :param model: Input model for training.
    :param num_split: (int) Number of splits for averaging of performance metrics.
    :param seed: (int) Seed for random state.
    :param X: (numpy array) Training input.
    :param Y: (numpy array) Class label.
    :param neural_network: (bool) whether the model is a neural network model (keras) or conventional machine learning model (scikit-learn).
    :return: Trained model
    """
    from sklearn.model_selection import ShuffleSplit
    import numpy as np
    import time

    shuffle = ShuffleSplit(n_splits=num_split, random_state=seed, test_size=0.2)
    accuracy, fitting_time = 0.0, 0.0
    tn, fp, fn, tp = 0.0, 0.0, 0.0, 0.0
    i = 0

    for train_idx, test_idx in shuffle.split(X):
        i += 1
        print ("== Split %s ==" %i)
        start = time.perf_counter()

        x_train, x_test, y_train, y_test = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

        ml = model

        if neural_network:
            if neural_network == 'convnn':
                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
            ml.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        else:
            ml.fit(x_train, y_train)

        pred_train = ml.predict(x_train)
        pred_test = ml.predict(x_test)

        pred_train = np.around(np.ndarray.flatten(pred_train))
        pred_test = np.around(np.ndarray.flatten(pred_test))

        end = time.perf_counter()

        tn_, fp_, fn_, tp_, acc_ = performance_metrics(y_train, y_test, pred_train, pred_test)

        tn += float(tn_) / num_split
        fp += float(fp_) / num_split
        fn += float(fn_) / num_split
        tp += float(tp_) / num_split

        accuracy += float(acc_) / num_split
        fitting_time += (end - start) / num_split
        print ("Fitting time", (end-start), "\n")

    print ("===== Average results over %s splits =====" % num_split)
    print ("Accuracy : %f" % accuracy)
    print ("TN, FP, FN, TP:", tn, fp, fn, tp)
    print ("Average time taken: %f" % fitting_time)
    print ("==========================================")

    return ml


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
    :param neural_network: (boolean) whether the model is a neural network model (keras) or conventional machine learning model (scikit-learn).
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
    from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score,\
        recall_score, f1_score

    print ("Training set report:")
    print ("Agro vs non-agro: %s vs %s" %(list(y_train).count(1), list(y_train).count(0)))
    tn_, fp_, fn_, tp_ = confusion_matrix(y_train, pred_train).ravel()
    print ("TN, FP, FN, TP: %s, %s, %s, %s" %(tn_, fp_, fn_, tp_))
    print ("Accuracy score:", accuracy_score(y_train, pred_train))
    print ("AUC score:", roc_auc_score(y_train, pred_train))
    print ("Precision score:", precision_score(y_train, pred_train))
    print ("Recall score:", recall_score(y_train, pred_train))
    print ("F1 score:", f1_score(y_train, pred_train))
    print ()
    print ("Testing set report:")
    print ("Agro vs non-agro: %s vs %s" %(list(y_test).count(1), list(y_test).count(0)))
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, pred_test).ravel()
    print ("TN, FP, FN, TP: %s, %s, %s, %s" %(tn_, fp_, fn_, tp_))
    acc = accuracy_score(y_test, pred_test)
    print ("Accuracy score:", acc)
    print ("AUC score:", roc_auc_score(y_test, pred_test))
    print ("Precision score:", precision_score(y_test, pred_test))
    print ("Recall score:", recall_score(y_test, pred_test))
    print ("F1 score:", f1_score(y_test, pred_test))
    print ()

    N_test = len(y_test)
    tn = float(tn_) / N_test 
    fp = float(fp_) / N_test 
    fn = float(fn_) / N_test 
    tp = float(tp_) / N_test 

    return tn, fp, fn, tp, acc

