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


def train_model(model, num_split, seed, X, Y, image_name, neural_network_epochs=0):
    """
    Train input model and obtain averaged results.
    :param model: Input model for training.
    :param num_split: (int) Number of splits for averaging of performance metrics.
    :param seed: (int) Seed for random state.
    :param X: (numpy array) Training input.
    :param Y: (numpy array) Class label.
    :param neural_network_epochs: (bool) whether the model is a neural network model (keras) or
    conventional machine learning model (scikit-learn) & at the same time specify number of epochs.
    :return: Trained model
    """
    from sklearn.model_selection import ShuffleSplit
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    import numpy as np
    import time

    shuffle = ShuffleSplit(n_splits=num_split, random_state=seed, test_size=0.2)
    accuracy, fitting_time = 0.0, 0.0
    tn, fp, fn, tp = 0.0, 0.0, 0.0, 0.0
    tpr, fpr, tprs, aucs = [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0

    for train_idx, test_idx in shuffle.split(X):
        i += 1
        print ("== Split %s ==" %i)
        start = time.perf_counter()

        x_train, x_test, y_train, y_test = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

        ml = model

        if neural_network_epochs:
            ml.fit(x_train, y_train, epochs=neural_network_epochs, batch_size=16, verbose=0)
        else:
            ml.fit(x_train, y_train)

        pred_train = ml.predict(x_train)
        pred_test = ml.predict(x_test)

        pred_train = np.around(np.ndarray.flatten(pred_train))
        pred_test = np.around(np.ndarray.flatten(pred_test))

        end = time.perf_counter()

        print (type(y_train), type(pred_train))
        tn_, fp_, fn_, tp_, acc_ = performance_metrics(y_train, y_test, pred_train, pred_test)

        tn += float(tn_) / num_split
        fp += float(fp_) / num_split
        fn += float(fn_) / num_split
        tp += float(tp_) / num_split

        accuracy += float(acc_) / num_split
        fitting_time += (end - start) / num_split
        print ("Fitting time", (end-start), "\n")

        # Plotting ROC curve
        fpr_, tpr_, thresholds = roc_curve(y_test, pred_test)
        tprs.append(interp(mean_fpr, fpr_, tpr_))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr_, tpr_)
        aucs.append(roc_auc)
        fpr.append(fpr_)
        tpr.append(tpr_)

    plot_roc_curve(fpr, tpr, aucs, tprs, image_name)

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
    from functions import general_functions as general
    general.check_path_exists(filename)

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


def plot_roc_curve(fpr, tpr, aucs, tprs, image_name):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import auc
    from functions import general_functions as general

    general.check_path_exists(image_name)

    i = 0
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    for fpr_, tpr_, roc_auc in zip(fpr, tpr, aucs):
        i += 1
        plt.plot(fpr_, tpr_, lw=1, alpha=0.3) #,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

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
    plt.title('ROC of %s' % image_name[image_name.rfind('/')+1:image_name.rfind("dataset")-1])
    plt.legend(loc="lower right")
    plt.savefig(image_name)
    # plt.show()


def plot_nn_loss_against_epoch(X, Y, layers_dim, activation, epochs, image_name,
                               loss='binary_crossentropy', optimizer='adam'):
    import matplotlib.pyplot as plt
    import numpy as np
    from functions import general_functions as general

    general.check_path_exists(image_name)

    model = build_simplenn_model(layers_dim=layers_dim, activation=activation, loss=loss,
                                 optimizer=optimizer)

    print()
    print("Number of epochs:", epochs)
    print("Loss function:", loss)
    print("Optimizer function:", optimizer)
    print()

    H = model.fit(X, Y, epochs=epochs, batch_size=16, verbose=0, validation_split=0.2, shuffle=True)

    training_loss = H.history['loss']
    validation_loss = H.history['val_loss']
    training_acc = H.history['acc']
    validation_acc = H.history['val_acc']

    plt.switch_backend('agg')
    plt.figure()
    plt.plot(np.arange(0, epochs), training_loss, marker='o', label="train_loss")
    plt.plot(np.arange(0, epochs), validation_loss, marker='o',label="val_loss")
    plt.plot(np.arange(0, epochs), training_acc, marker='o',label="train_acc")
    plt.plot(np.arange(0, epochs), validation_acc, marker='o',label="val_acc")
    # plt.title("Loss and Accuracy against Epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc="best")
    plt.savefig(image_name)

    return training_acc, training_loss, validation_acc, validation_loss





