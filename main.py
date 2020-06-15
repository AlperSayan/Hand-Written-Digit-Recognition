from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

# I could have used sklearns GridSearchCv method for optimal parameter tuning but it took too much time to compute
# duration > 60 minutes for just knn, therefore I opted to use manuel parameter tuning,
# accuracies might not be the best, but
# they are close to optimum
# This algorithm with the sample size in the code takes about 5 minutes to complete
# with lower sample size it might not work because not all classes are obtained

# The result of each clustering algorithm is on the Sciview, with default train and test values confusion matrix and
# accuracies first. And then the other 2 are with tuned parameters.


def report_accuracy(result, labels):
    sum = 0
    for i in range(len(result)):
        if labels[i] == result[i]:
            sum += 1
    return (sum / len(result)) * 100


def main():

    # load and split dataset, change the numbers if less completion time is desired. Will change results
    mints_dataset_file_names = ["train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                "t10k-labels.idx1-ubyte"]

    MINST_dict = load_minst(mints_dataset_file_names)

    # for timing constraints sample size is low. I used higher values for the report. (10000=train, 1000=test)
    train_size = 1000
    test_size = 100
    train_images = np.copy(MINST_dict['train_images'][:train_size])
    train_labels = np.copy(MINST_dict['train_labels'][:train_size])
    test_images = np.copy(MINST_dict['test_images'][:test_size])
    test_labels = MINST_dict['test_labels'][:test_size]

    train_images_1d = np.reshape(train_images, (train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
    test_images_1d = np.reshape(test_images, (test_images.shape[0], test_images.shape[1] * test_images.shape[2]))

    # comment part where you don`t want to test if you want to test separately
    KNN_train_and_tests(train_images_1d, test_images_1d, train_labels, test_labels)
    GNB_train_and_tests(train_images_1d, test_images_1d, train_labels, test_labels)
    CLF_train_and_tests(train_images_1d, test_images_1d, train_labels, test_labels)


def CLF_train_and_tests(train_images_1d, test_images_1d, train_labels, test_labels):

    # parameters to be tested, not tested all ranges because of performance reasons
    criterion_list = ["gini", "entropy"]
    splitter_list = ["best", "random"]
    max_depth_list = [5, 10, 15, 30, 50, 100]
    max_depth_list.reverse()
    min_samples_split_list = [2, 4, 6, 10]
    min_samples_split_list.reverse()
    min_samples_leaf_list = [1, 3, 5, 10]
    min_samples_leaf_list.reverse()

    best_params_train = []
    best_params_test = []
    best_acc_train = 0
    best_acc_test = 0
    best_param_results_train = []
    best_param_results_test = []
    acc_list_train = []
    acc_list_test = []

    # default accuracy and results of the train and test datasets
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_images_1d, train_labels)
    results = clf.predict(train_images_1d)
    conf_matrix = compute_confusion_matrix(train_labels, results)
    draw_conf_matrix(conf_matrix.astype('uint8'),
                     param=["default"]
                     , acc=report_accuracy(results, train_labels), train_or_test="train"
                     , classifier="clf")

    results = clf.predict(test_images_1d)
    conf_matrix = compute_confusion_matrix(test_labels, results)
    draw_conf_matrix(conf_matrix.astype('uint8'),
                     param=["default"]
                     , acc=report_accuracy(results, test_labels), train_or_test="test"
                     , classifier="clf")

    # code i used for parameter tuning. Automatic tuning takes too much time
    for criterion in criterion_list:
        for splitter in splitter_list:
            for max_depth in max_depth_list:
                for min_samples_split in min_samples_split_list:
                    for min_samples_leaf in min_samples_leaf_list:

                        clf = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth
                                                          , min_samples_split=min_samples_split
                                                          , min_samples_leaf=min_samples_leaf)
                        clf.fit(train_images_1d, train_labels)
                        results = clf.predict(train_images_1d)

                        if best_acc_train < report_accuracy(results, train_labels):
                            best_acc_train = report_accuracy(results, train_labels)
                            best_param_results_train = results
                            best_params_train.append(criterion)
                            best_params_train.append(splitter)
                            best_params_train.append(max_depth)
                            best_params_train.append(min_samples_split)
                            best_params_train.append(min_samples_leaf)
                            acc_list_train.append(best_acc_train)

                        results = clf.predict(test_images_1d)

                        if best_acc_test < report_accuracy(results, test_labels):
                            best_acc_test = report_accuracy(results, test_labels)
                            best_param_results_test = results
                            best_params_test.append(criterion)
                            best_params_test.append(splitter)
                            best_params_test.append(max_depth)
                            best_params_test.append(min_samples_split)
                            best_params_test.append(min_samples_leaf)
                            acc_list_test.append(best_acc_test)

    best_params_train = np.array(best_params_train)
    best_params_test = np.array(best_params_test)

    # best results achieved with parameter tuning
    conf_matrix = compute_confusion_matrix(train_labels, best_param_results_train)
    draw_conf_matrix(conf_matrix.astype('uint8'),
                     param=best_params_train[-5:]
                     , acc=best_acc_train, train_or_test="train"
                     , classifier="clf")

    conf_matrix = compute_confusion_matrix(test_labels, best_param_results_test)
    draw_conf_matrix(conf_matrix.astype('uint8'),
                     param=best_params_test[-5:]
                     , acc=best_acc_test, train_or_test="test"
                     , classifier="clf")

    # plot changing accuracy with better parameter tuning
    line_plot_CLF(acc_list_train, range(len(acc_list_train)), train_or_test="train")
    line_plot_CLF(acc_list_test, range(len(acc_list_test)), train_or_test="test")


def line_plot_CLF(accuracy, len, train_or_test):
    plt.plot(len, accuracy, linestyle='--', marker='o')
    plt.xlabel("iteration")
    plt.xticks(len)
    plt.ylabel("accuracy")
    plt.title("best accuracy increase each iteration [" + train_or_test + "_set]")
    plt.show()


def GNB_train_and_tests(train_images_1d, test_images_1d, train_labels, test_labels):

    # tested smoothing values
    var_values = [0.001, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    acc_list_train = []
    acc_list_test = []
    best_acc_train = 0
    best_smooth_train = 0
    best_acc_test = 0
    best_smooth_test = 0
    best_smooth_results_train = []
    best_smooth_results_test = []

    # default gnb test, results, accuracy
    gnb = GaussianNB()
    gnb.fit(train_images_1d, train_labels)
    results = gnb.predict(train_images_1d)
    conf_matrix = compute_confusion_matrix(train_labels, results)
    draw_conf_matrix(conf_matrix.astype('uint'), param="default", acc=report_accuracy(results, train_labels)
                     , train_or_test="train"
                     , classifier="gnb")

    results = gnb.predict(test_images_1d)
    conf_matrix = compute_confusion_matrix(test_labels, results)
    draw_conf_matrix(conf_matrix.astype('uint'), param="default", acc=report_accuracy(results, test_labels)
                     , train_or_test="test"
                     , classifier="gnb")

    # best accuracy result with tuned parameters, results are on plot
    for var_value in var_values:
        gnb = GaussianNB(var_smoothing=var_value)
        gnb.fit(train_images_1d, train_labels)

        results = gnb.predict(train_images_1d)
        acc_list_train.append(float("{:.1f}".format(report_accuracy(results, train_labels))))

        if best_acc_train < report_accuracy(results, train_labels):
            best_smooth_train = var_value
            best_acc_train = report_accuracy(results, train_labels)
            best_smooth_results_train = results

        results = gnb.predict(test_images_1d)
        acc_list_test.append(float("{:.1f}".format(report_accuracy(results, test_labels))))

        if best_acc_test < report_accuracy(results, test_labels):
            best_smooth_test = var_value
            best_acc_test = report_accuracy(results, test_labels)
            best_smooth_results_test = results


    # best achieved accuracies and its confusion matrix
    conf_matrix = compute_confusion_matrix(train_labels, best_smooth_results_train)
    draw_conf_matrix(conf_matrix.astype('uint8'), param='{:.3f}'.format(best_smooth_train),
                     acc=best_acc_train, train_or_test="train", classifier="gnb")

    conf_matrix = compute_confusion_matrix(test_labels, best_smooth_results_test)
    draw_conf_matrix(conf_matrix.astype('uint8'), param='{:.3f}'.format(best_smooth_test),
                     acc=best_acc_test, train_or_test="test", classifier="gnb")

    # plot changing accuracy with changing smoothing values
    line_plot_GNB(acc_list_train, var_values, train_or_test="train")
    line_plot_GNB(acc_list_test, var_values, train_or_test="test")


def line_plot_GNB(accuracy, kvalues, train_or_test):
    plt.plot(kvalues, accuracy, linestyle='--', marker='o')
    plt.xlabel("smoothing value")
    plt.xticks(kvalues)
    plt.ylabel("accuracy")
    plt.title("accuracy with changing smoothing values [" + train_or_test + "_set]")
    plt.show()


def KNN_train_and_tests(train_images_1d, test_images_1d, train_labels, test_labels):

    global KNN_results_train, KNN_results_test
    acc_list_train = []
    kvals = range(1, 51, 5)
    weights_list = ['uniform', 'distance']
    p_list = [1, 2]
    acc_list_test = []

    best_acc_train = 0
    best_kval_train = 0
    best_acc_test = 0
    best_kval_test = 0
    best_weight_train = 0
    best_weight_test = 0
    best_p_train = 0
    best_p_test = 0

    best_knn_results_train = []
    best_knn_results_test = []

    # train and test results with default knn parameters accuracy is reported in confusion matrix
    KNN = neighbors.KNeighborsClassifier()
    KNN.fit(train_images_1d, train_labels)
    KNN_results = np.array(KNN.predict(train_images_1d)).astype('uint8')
    confusion_matrix = compute_confusion_matrix(train_labels.astype('uint8'), KNN_results)
    draw_conf_matrix(confusion_matrix.astype('uint8'), ["default"],
                    report_accuracy(KNN_results, train_labels), train_or_test="train", classifier="knn")

    KNN_results = np.array(KNN.predict(test_images_1d)).astype('uint8')
    confusion_matrix = compute_confusion_matrix(test_labels.astype('uint8'), KNN_results)
    draw_conf_matrix(confusion_matrix.astype('uint8'), ["default"],
                    report_accuracy(KNN_results, test_labels), train_or_test="test", classifier="knn")

    # confusion matrix and accuracy with changing k values, weight method, and distance calculation
    # only the best confusion matrix is computed/plotted as a result
    for kval in kvals:
        for weight in weights_list:
            for p in p_list:

                KNN = neighbors.KNeighborsClassifier(kval, weights=weight, p=p)
                KNN.fit(train_images_1d, train_labels)

                KNN_results_train = np.array(KNN.predict(train_images_1d)).astype('uint8')


                if best_acc_train < report_accuracy(KNN_results_train, train_labels):
                    best_kval_train = kval
                    best_acc_train = report_accuracy(KNN_results_train, train_labels)
                    best_knn_results_train = KNN_results_train
                    best_weight_train = weight
                    best_p_train = p


                KNN_results_test = np.array(KNN.predict(test_images_1d)).astype('uint8')

                if best_acc_test < report_accuracy(KNN_results_test, test_labels):
                    best_kval_test = kval
                    best_acc_test = report_accuracy(KNN_results_test, test_labels)
                    best_knn_results_test = KNN_results_test
                    best_weight_test = weight
                    best_p_test = p

        acc_list_train.append(report_accuracy(KNN_results_train, train_labels))
        acc_list_test.append(report_accuracy(KNN_results_test, test_labels))

    # plot the confusion matrix with best results
    confusion_matrix = compute_confusion_matrix(train_labels.astype('uint8'), best_knn_results_train)
    draw_conf_matrix(confusion_matrix.astype('uint8'), param=[best_kval_train, best_weight_train, best_p_train],
                    acc=best_acc_train, train_or_test="train", classifier="knn")

    confusion_matrix = compute_confusion_matrix(test_labels.astype('uint8'), best_knn_results_test)
    draw_conf_matrix(confusion_matrix.astype('uint8'), param=[best_kval_test, best_weight_test, best_p_test],
                    acc=best_acc_test, train_or_test="test", classifier="knn")

    # plot accuracy with changing k values
    line_plot_knn(acc_list_train, kvals, train_or_test="train")
    line_plot_knn(acc_list_test, kvals, train_or_test="test")


def line_plot_knn(accuracy, kvalues, train_or_test):
    plt.plot(kvalues, accuracy, linestyle='--', marker='o')
    plt.xlabel("k value")
    plt.xticks(kvalues)
    plt.ylabel("accuracy")
    plt.title("accuracy with changing k values [" + train_or_test + "_set]")
    plt.show()


def compute_confusion_matrix(true, pred):
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


def draw_conf_matrix(conf_matrix, param, acc, train_or_test, classifier):
   classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

   fig, ax = plt.subplots()
   im = ax.imshow(conf_matrix)

   ax.set_xticks(np.arange(len(classes)))
   ax.set_yticks(np.arange(len(classes)))
   # ... and label them with the respective list entries
   ax.set_xticklabels(classes)
   ax.set_yticklabels(classes)

   plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

   for i in range(len(classes)):
       for j in range(len(classes)):
           text = ax.text(j, i, conf_matrix[i, j],
                          ha="center", va="center", color="w")

   if(classifier == "knn"):
       if(param[0] == "default"):
           ax.set_title("Confusion matrix of the MNIST dataset\n with default parameters\n Accuracy=%" + str(int(acc))
                        + " [" + train_or_test + "_set]")

       else:
           ax.set_title(
               "Confusion matrix of the MNIST dataset\n with k=" + str(param[0]) + " weight=" + str(param[1])
               + " p=" + str(param[2])
               + "\n accuracy=%" + str(int(acc))
               + " [" + train_or_test + "_set]")

   if(classifier == "gnb"):
       if param == "default":
           ax.set_title(
               "Confusion matrix of the MNIST dataset\n with default params" + " accuracy=%" + str(int(acc))
               + " [" + train_or_test + "_set]")
       else:
           ax.set_title(
               "Confusion matrix of the MNIST dataset\n with smoothing=" + str(param) + " accuracy=%" + str(int(acc))
               + " [" + train_or_test + "_set]")

   if(classifier == "clf"):
       if param[0] == "default":
           ax.set_title(
               "Confusion matrix of the MNIST dataset\n with default parameters"
               + " accuracy=%" + str(int(acc))
               + " [" + train_or_test + "_set]")
       else:
           ax.set_title(
               "criterion=" + str(param[0]) + " splitter=" + str(param[1]) + "\nmax depth=" + str(param[2])
               + " min samples split=" + str(param[3]) + " min samples leaf=" + str(param[3])
               + "\n accuracy=%" + str(int(acc))
               + " [" + train_or_test + "_set]")


   plt.ylabel("True label")
   plt.xlabel("Predicted label")
   fig.tight_layout()
   plt.show()


def get_int(byte):
    return int.from_bytes(byte, "big")


def load_minst(minst_dataset):
    data_dict = {}

    for file_name in minst_dataset:
        if file_name.endswith('ubyte'):
            with open(file_name, 'rb') as f:
                data = f.read()
                type = get_int(data[:4])
                length = get_int(data[4:8])
                if (type == 2051):
                    category = 'images'
                    num_rows = get_int(data[8:12])
                    num_cols = get_int(data[12:16])
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
                    parsed = parsed.reshape(length, num_rows, num_cols)
                elif (type == 2049):
                    category = 'labels'
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                    parsed = parsed.reshape(length)
                if (length == 10000):
                    set = 'test'
                elif (length == 60000):
                    set = 'train'
                data_dict[set + '_' + category] = parsed

    return data_dict


if __name__ == '__main__':
    main()
