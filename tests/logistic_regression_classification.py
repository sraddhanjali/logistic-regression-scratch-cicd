import time
import pandas as pd
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


def custom_softmax(unsoftmaxed_data, along_axis=0):
    """Computes the softmax function to data along a particular axis (only for rows, columns).
    :param unsoftmaxed_data: data to be applied softmax to. (ndarray)
    :param along_axis: axis along which to apply softmax;
           along_axis = 0 applies the function column wise (default value).
           along_axis = 1 applies the function row wise.
    :return: softmaxed data (ndarray).
    """

    def softmax_func(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    return np.apply_along_axis(softmax_func, along_axis, unsoftmaxed_data)


def create_powers_desc(descriptors, complexities):
    """Adds powers of descriptors using values in polys.
    :param descriptors: a list of descriptors.
    :param complexities: a list of complexities.
    :return: iterable for powers.
    """
    for degree in complexities[1:]:
        for d in descriptors:
            r = np.power(d, degree)
            yield r.T


def create_combination_desc(descriptors, complexities, d_shape):
    """Build combinations of descriptors.
    :param descriptors: a list of descriptors.
    :param complexities: a list of polynomial complexities.
    :param d_shape: number of instances in data.
    :return: iterable for combination of descriptors.
    """
    for degree in complexities:
        indices = [i for i in range(len(descriptors))]
        iterab = combinations(indices, degree)
        for it in iterab:
            mult = np.ones(d_shape)
            for i in it:
                mult *= descriptors[i]
            yield mult.T


def create_phi_matrix(descriptors, d_shape, complexities):
    """Build a phi data from descriptors.
    :param descriptors: a list of descriptors.
    :param d_shape: number of instances in data.
    :param complexities: a list of polynomial complexities.
    :return: matrix of features built from desc as a list.
    """
    one = np.ones(d_shape)
    comb_arr = []
    com_pow = []
    for val in create_combination_desc(descriptors, complexities, d_shape):
        comb_arr.append(val)
    for val in create_powers_desc(descriptors, complexities):
        com_pow.append(val)
    return np.hstack([one, descriptors, np.hstack(comb_arr), np.hstack(com_pow)])
    # return np.matrix(np.vstack(phi).T)


def compute_target_vector(y_original, n_class):
    """Convert a 1D class labels to numerical encodings.
    :param y_original: 1D class labels.
    :param n_class: number of classes.
    :return: numerical encodings (ndarray).
    """
    if n_class == 2:
        y_encoded = np.zeros((y_original.shape[0], 2))
        for i in range(y_original.shape[0]):
            if y_original[i] == 1:
                y_encoded[i, 1] = 1
            elif y_original[i] == 0:
                y_encoded[i, 0] = 1
    else:
        labelencoder_y = LabelBinarizer()
        y_encoded = labelencoder_y.fit_transform(y_original)
    return y_encoded


def gradient_mat(phi_matrix, old_weight, n_class, target_vec, reg_param_lambda):
    """Computes the cost function and the gradient matrix.
    :param phi_matrix: matrix containing features built from original descriptors from data.
    :param old_weight: old weight matrix.
    :param n_class: number of classes.
    :param target_vec: target encoded values.
    :param reg_param_lambda: regularization parameter.
    :return: error gradient matrix and cost Function (float).
    """
    y = phi_matrix * old_weight
    n_samples = y.shape[0]
    y = custom_softmax(y, along_axis=1)
    b = (1 / n_samples) * reg_param_lambda * np.sum(np.square(old_weight))
    tar_vector_sh = (target_vec.reshape((n_samples * n_class), 1)).T
    y_sh = y.reshape((n_samples * n_class), 1)
    cost_val = -(tar_vector_sh * np.log(y_sh)) / n_samples + b
    diff = y - target_vec
    grad_matrix = ((phi_matrix.T * diff) + reg_param_lambda * old_weight) / n_samples
    return grad_matrix, cost_val


def grad_descent(
    phi_matrix,
    n_class,
    curr_weight,
    l_rate,
    max_iteration,
    target_vec,
    reg_param_lambda,
):
    """Optimize weights using gradient descent algorithm with stopping criterions.
    :param phi_matrix: matrix containing features built from original descriptors from data.
    :param n_class: number of classes.
    :param curr_weight: current weight matrix.
    :param l_rate: learning rate.
    :param max_iteration: maximum iterations to go for while updating weights.
    :param target_vec: target encoded values.
    :param reg_param_lambda: regularization parameter.
    :return: new weights, cost list, total iterations, gradient matrix history.
    """
    old_weight = np.zeros((phi_matrix.shape[1], n_class))
    iters = 0
    cost_history = []
    cost_iterations = []
    grad_matrix_history = []
    while not np.allclose(old_weight, curr_weight) and iters < max_iteration:
        old_weight = curr_weight
        grad_matrix, cost_val = gradient_mat(
            phi_matrix, old_weight, n_class, target_vec, reg_param_lambda
        )
        curr_weight = curr_weight - l_rate * grad_matrix
        cost_history.append(cost_val)
        grad_matrix_history.append(grad_matrix)
        cost_iterations.append(iters)
        iters = iters + 1
    return curr_weight, cost_history, cost_iterations, grad_matrix_history


def standardize_data(org_data):
    """Use mean and variance of data to standardize the data.
    :param org_data: data (1D) array.
    :return: standardized array, with its mean and variance.
    """
    mean_ = np.mean(org_data)
    var_ = np.std(org_data)
    stand_data = (org_data - mean_) / var_
    return stand_data, mean_, var_


def mapping_three_prob_to_class(y_logistic):
    """
    :param y_logistic: encoded labels.
    :return: list of class labels obtained from encoded labels.
    """
    return [np.argmax(row) for row in y_logistic]


def accuracy(y_pred, y_original):
    """Compute accuracy comparing matches between predicted and original class labels.
    :param y_pred: predicted class labels (list).
    :param y_original: original class labels (list).
    :return: accuracy percentage (float).
    """
    y_pred = np.asarray(y_pred)
    y_original = np.asarray(y_original)
    count = np.sum(y_pred == y_original)
    acc = (count / len(y_pred)) * 100
    return acc


def descriptor_plot(d1, d2, y, title):
    """Plot scatter matrix of two descriptors.
    :param d1: first descriptor (list).
    :param d2: second descriptor (list.
    :param y: class labels (list).
    :param title: title of the plot.
    :return: None
    """
    colors = ["green", "blue", "red"]
    plt.scatter(d1, d2, c=y, cmap=mc.ListedColormap(colors), marker="*", s=30)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.title(title)
    plt.show()


def make_synthetic_data(n_samples, n_classes, n_features=2):
    """Make synthetic data to work for validation of implementation.
    :param n_samples: number of samples in data.
    :param n_classes: number of classes.
    :param n_features: number of features built from descriptors.
    :return: data (ndarray).
    """
    return make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=1.6,
        random_state=50,
    )


def split_data(all_data, labels, test_size=0.10):
    return train_test_split(all_data, labels, test_size=test_size, random_state=42)


def preprocess_data(descriptors, mean_vars):
    descrips = []
    if not mean_vars:
        for i, d in enumerate(descriptors):
            d_norm, mean, var = standardize_data(d)
            mean_vars[i] = [mean, var]
            descrips.append(d_norm)
    else:
        for i, d in enumerate(descriptors):
            d_norm = (d - mean_vars[i][0]) / mean_vars[i][1]
            descrips.append(d_norm)
    d_shape = descrips[0].shape[0]
    return descrips, d_shape, mean_vars


def plot_cost_function_convergence(cost_history, cost_iteration):
    for i in range(len(cost_iteration)):
        cost_history[i] = cost_history[i].tolist()
    cost_history = [i[0] for i in cost_history]
    plt.plot(cost_iteration, cost_history)
    plt.title("Cost Function")
    plt.xlabel("Iterations")
    plt.ylabel("J_cost")
    plt.show()


def plot_data():
    # plotting for test data
    #  descriptor_plot(d1_test1, d2_test2, y_test, 'actual-test-data') #Actual test plots
    #  descriptor_plot(d1_test1, d2_test2, y_map_test, 'predicted-test-data')#predicted test plots
    #
    # #plotting for train data
    #  descriptor_plot(d1_train, d2_train, y_train, 'actual-train-data') #Actual train plots
    #  descriptor_plot(d1_train, d2_train, y_map_train, 'predicted-train-data')#predicted train plots
    pass


def main_run_for_any_data(
    data,
    train_index,
    test_index,
    n_class,
    complexities,
    reg_param_lambda,
    max_iteration,
    l_rate,
):
    X_train = []
    X_test = []
    for i in range(data[0].shape[1]):
        X_train.append(data[0][:, i][train_index])
        X_test.append(data[0][:, i][test_index])

    y_train, y_test = data[1][train_index], data[1][test_index]
    desc, data_shape, mean_var_list = preprocess_data(X_train, {})
    desc_test, data_test_shape, mean_var_list = preprocess_data(X_test, mean_var_list)
    phi_fin_train = create_phi_matrix(desc, data_shape, complexities)
    phi_fin_test = create_phi_matrix(desc_test, data_test_shape, complexities)

    tar_vector = compute_target_vector(y_train, n_class)
    cur_w = np.random.random_sample((phi_fin_train.shape[1], n_class))

    # start time for training
    t0 = time.time()
    fin_w, cost_history, cost_iterations, g = grad_descent(
        phi_fin_train,
        n_class,
        cur_w,
        l_rate,
        max_iteration,
        tar_vector,
        reg_param_lambda,
    )
    t1 = time.time()
    training_time = t1 - t0
    # end time for training

    # plot_cost_function_convergence(cost_history, cost_iterations)
    # start of testing
    t00 = time.time()
    y_logistic_train = (fin_w.T * phi_fin_train.T).T
    print(fin_w.shape, phi_fin_train.shape, y_logistic_train.shape)
    print("_______________________________-------")
    t11 = time.time()
    testing_time = t11 - t00
    # end of testing
    y_logistic_train = custom_softmax(y_logistic_train, along_axis=1)
    y_map_train = mapping_three_prob_to_class(y_logistic_train)

    # start time testing
    y_logistic_test = (fin_w.T * phi_fin_test.T).T
    # end time testing
    y_logistic_test = custom_softmax(y_logistic_test, along_axis=1)
    y_map_test = mapping_three_prob_to_class(y_logistic_test)

    # plot_confusion_matrix(y_test, y_map_test)
    acc_train = accuracy(y_map_train, y_train)
    print("Training accuracy = {0}".format(acc_train))
    acc_test = accuracy(y_map_test, y_test)
    print("Testing accuracy = {0}".format(acc_test))
    print("Training time = {0}".format(training_time))
    print("Testing time = {0}".format(testing_time))
    return acc_train, acc_test, training_time, testing_time


def average_accuracies(accuracy_dict):
    accuracies = list(accuracy_dict.values())
    n_accuracies = len(accuracies)
    return np.sum(accuracies) / n_accuracies


def averaging_times(time_dict):
    time_values = list(time_dict.values())
    n_time_values = len(time_values)
    return np.sum(time_values) / n_time_values


def run_kfold(
    data, folds, complexities, reg_param_lambda, max_iteration, l_rate, n_class
):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    kf.get_n_splits(data[0])
    training_acc = {}
    testing_acc = {}
    training_time = {}
    testing_time = {}
    fold_n = 0
    for train_ind, test_ind in kf.split(data[0][:, 0]):
        fold_n += 1
        print("--------Running K fold for fold={0}.......".format(fold_n))
        a_train, a_test, train_time, test_time = main_run_for_any_data(
            data,
            train_ind,
            test_ind,
            n_class,
            complexities,
            reg_param_lambda,
            max_iteration,
            l_rate,
        )
        training_acc[fold_n] = a_train
        testing_acc[fold_n] = a_test
        training_time[fold_n] = train_time
        testing_time[fold_n] = test_time

    print("------Training Accuracy----")
    print(training_acc)
    print("------Testing Accuracy----")
    print(testing_acc)
    print("-------Avg Training Accuracy----")
    avg_train_acc = average_accuracies(training_acc)
    print(avg_train_acc)
    print("-------Avg Testing Accuracy----")
    avg_test_acc = average_accuracies(testing_acc)
    print(avg_test_acc)
    print("------Training Time----")
    print(training_time)
    print("------Testing Time----")
    print(testing_time)
    print("-------Avg Training Time----")
    avg_train_time = averaging_times(training_time)
    print(avg_train_time)
    print("-------Avg Testing Time----")
    avg_test_time = averaging_times(testing_time)
    print(avg_test_time)
    return avg_train_acc, avg_test_acc, avg_train_time, avg_test_time


# def plot_confusion_matrix(y_test, y_map_test):
#
#     cm = confusion_matrix(y_test, y_map_test)
#
#     ax = plt.subplot()
#     sns.heatmap(cm, annot=True, ax=ax, cmap="YlGnBu");  # annot=True to annotate cells
#
#     # labels, title and ticks
#     ax.set_xlabel('Predicted labels');
#     ax.set_ylabel('True labels');
#     ax.set_title('Confusion Matrix');
#     ax.xaxis.set_ticklabels(['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']);
#     ax.yaxis.set_ticklabels(['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']);
#     plt.show()


def read_main_data(file_name):
    data = pd.read_csv(
        file_name,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
    )
    data = shuffle(data)
    labels = data["class"]
    descriptors = data.drop(columns=["class"])
    lb = LabelEncoder()
    labels_numerical = lb.fit_transform(labels.values)
    return descriptors.values, labels_numerical, lb


if __name__ == "__main__":

    np.random.seed(100)
    n = 1000
    k_class = 3
    m1 = 3
    rate1 = 0.05
    max_iters = 500
    lamda1 = 0.00001
    fold = 2
    iris_filename = "iris_data.csv"
    print(
        "--The code runs for synthetic dataset first and then for iris dataset next--"
    )
    print("-----------For m={0} ------".format(m1))
    print("-----------For rate={0}-----".format(rate1))
    print("-----------For lambda={0}----".format(lamda1))
    polys = [i for i in range(1, m1 + 1)]
    print("-----------FOR SYNTHETIC DATASET---------")
    synth_data = make_synthetic_data(n, k_class)
    strainacc, stestacc, straintime, stesttime = run_kfold(
        synth_data, fold, polys, lamda1, max_iters, rate1, k_class
    )
    all_data = read_main_data(iris_filename)
    main_data = all_data[:2]
    # for converting to categorical later
    lb = all_data[2:]
    print("------------------------------------")
    print("------------------------------------")
    print("-----------FOR IRIS DATASET---------")
    m2 = 4
    rate2 = 0.5
    lamda2 = 0.00001
    polys = [i for i in range(1, m2 + 1)]
    print("-----------For m={0} ------".format(m2))
    print("-----------For rate={0}-----".format(rate2))
    print("-----------For lambda={0}----".format(lamda2))
    itrainacc, itestacc, itraintime, itesttime = run_kfold(
        main_data, fold, polys, lamda2, max_iters, rate2, k_class
    )
    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")
    print("---------------SUMMARY---------------------")
    print("---------------------------------------------------------------------------")
    print("----------------------------FOR SYNTHETIC DATASET-------------------------")
    print("---------------------------------------------------------------------------")
    print("-------Avg Training Accuracy----")
    print(strainacc)
    print("-------Avg Testing Accuracy-----")
    print(stestacc)
    print("-------Avg Training Time--------")
    print(straintime)
    print("-------Avg Testing Time---------")
    print(stesttime)
    print("---------------------------------------------------------------------------")
    print(
        "--------------------------FOR IRIS DATASET----------------------------------------"
    )
    print("---------------------------------------------------------------------------")
    print("-------Avg Training Accuracy----")
    print(itrainacc)
    print("-------Avg Testing Accuracy-----")
    print(itestacc)
    print("-------Avg Training Time--------")
    print(itraintime)
    print("-------Avg Testing Time---------")
    print(itesttime)
