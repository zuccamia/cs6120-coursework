import pickle

import nltk
import numpy as np

from utils import extract_features

nltk.download("twitter_samples")
nltk.download("stopwords")

################################################################################
# @title Q1.1 Sigmoid Inference
################################################################################


def sigmoid(x):
    """
    This function implements the logistic regression algorithm.

    Args:
      x: A numpy array of shape (d,) or (N, d) representing
        the input data.

    Returns:
      y: A numpy array of shape (l,) or (N, l) representing
        the output data.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_test():
    x = np.array([1, 2, 3])
    y = sigmoid(x)
    assert np.allclose(y, np.array([0.73105858, 0.88079708, 0.95257413]))
    print("sigmoid: \033[1;32mtests OK.\033[0m")


################################################################################
# @title Q1.2 One Layer Inference
################################################################################


def inference_layer(X, W, b):
    """
    Implements the forward propagation for the logistic regression model.

    Args:
      X: The input data, of shape (number of features, number of examples).
      W: Weights, a numpy array of shape (number of features, 1).
      b: Bias, a scalar.

    Returns:
      y: The output of shape l or l x N
    """
    z = W @ X + b[:, None]
    return sigmoid(z)


def inference_layer_test():
    X = np.array([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]])
    W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b = np.array([1, 2, 3])
    y = inference_layer(X, W, b)
    assert np.allclose(
        y,
        np.array(
            [
                [
                    0.86989153,
                    0.90024951,
                    0.92414182,
                    0.94267582,
                    0.95689275,
                    0.96770454,
                ],
                [0.98015969, 0.9900482, 0.9950332, 0.99752738, 0.9987706, 0.99938912],
                [0.99726804, 0.99908895, 0.99969655, 0.99989897, 0.99996637, 0.9999888],
            ]
        ),
    )
    print("inference_layer: \033[1;32mtests OK.\033[0m")


################################################################################
# @title Q1.3 Two Layer Inference
################################################################################


def inference_2layers(X, W1, W2, b1, b2):
    """
    Implements the forward propagation of a two layer neural network
    model.

    - Let d be the number of features (dimensionality) of the input.
    - Let N be the number of data samples in your batch
    - Let H be the number of hidden units

    Args:
    X: The input data, with shape either d or d x N
    W1: Weights for the first weight matrix: shape = H x d
    w2: Weights for the second matrix: shape = 1 x H
    b1: Bias value for first layer
    b2: Bias value for second layer

    Returns:
    y: Singular output data (1, 0), with shape either 1 or N.
    """
    a1 = inference_layer(X, W1, b1)
    return inference_layer(a1, W2, b2)


def inference_2layers_test():
    X = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    W2 = np.array([[0.7, 0.8, 0.9]])
    b1 = np.array([1, 2, 3])
    b2 = np.array([4])
    yhat = inference_2layers(X, W1, W2, b1, b2)
    assert np.allclose(
        yhat, np.array([[0.99814977, 0.99820579, 0.99824346, 0.99826983]])
    )
    print("inference_2layers: \033[1;32mtests OK.\033[0m")


################################################################################
# @title Q1.4 BCE Loss
################################################################################


def bce_forward(yhat, y):
    """
    This function calculates the gradient of the logistic regression cost
    function with respect to the weights and biases.

    Args:
      yhat: A numpy array of shape (N,) representing the predicted outputs
      y: A numpy array of shape (N,) representing the target labels.

    Returns:
      loss_value
    """
    derived = y * (np.log(yhat)) + (1 - y) * (np.log(1 - yhat))
    return -np.mean(derived)


def bce_forward_test():
    yhat = np.array([0.5, 0.5, 0.5])
    y = np.array([0, 1, 0])
    loss_value = bce_forward(yhat, y)
    assert np.allclose(loss_value, 0.6931471805599453)
    print("bce_forward: \033[1;32mtests OK.\033[0m")


################################################################################
# @title Q3.1 Gradients
################################################################################


def gradients(X, y, W1, W2, b1, b2):
    """
    Calculate the gradients of the cost functions with respect to W1, W2, b1, and b2.

    Args:
      W1: Weight np.array of shape (h, d), first layer
      b1: Bias np.array of shape (h,), first layer
      W1: Weight np.array of shape (1, h), second layer
      b1: Bias value, second layer
      X: Input data of shape (d, N) representing the input data.
      y: Target / output labels of shape (N,)

    Returns:
      dL/dW1: Derivative of cost fxn w/r/t W1
      dL/db1: Derivative of cost fxn w/r/t b1
      dL/dW2: Derivative of cost fxn w/r/t W2
      dL/db2: Derivative of cost fxn w/r/t b2
      L: Loss value
    """
    # forward
    z1 = W1 @ X + b1[:, None]
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + b2[:, None]
    a2 = sigmoid(z2)
    loss = bce_forward(a2, y)

    # backward
    # gradient of loss L with respect to A2
    dz2 = a2 - y
    # gradient of L with respect to W2
    dl_dw2 = dz2 @ a1.T
    # gradient of L with respect to b2
    dl_db2 = np.sum(dz2)
    # gradient of L with respect to A1
    da1 = W2.T @ dz2
    # gradient of L with respect to Z1
    dz1 = da1 * a1 * (1 - a1)
    # gradient of L with respect to W1
    dl_dw1 = dz1 @ X.T
    # gradient of L with respect to b1
    dl_db1 = np.sum(dz1, axis=1)

    return dl_dw1, dl_db1, dl_dw2, dl_db2, loss


def gradients_test():

    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1, 0])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])
    W2 = np.array([[0.7, 0.8, 0.9]])
    b2 = np.array([0.3])
    dL_dW1, dL_db1, dL_dW2, dL_db2, L = gradients(X, y, W1, W2, b1, b2)

    assert dL_dW1.shape == (3, 2), (
        "dL_dW1 should have shape (3, 2). It is currently %s" % str(dL_dW1.shape)
    )
    assert dL_db1.shape == (3,), (
        "dL_db1 should have shape (3,). It is currently %s" % str(dL_db1.shape)
    )
    assert dL_db1.shape == (3,), (
        "dL_db1 should have shape (3,). It is currently %s" % str(dL_db1.shape)
    )
    assert dL_dW2.shape == (1, 3), (
        "dL_dW2 should have shape (1, 3). It is currently %s" % str(dL_dW2.shape)
    )
    assert dL_db2.shape == () or dL_db2.shape == (1,), (
        "dL_db2 should have shape (). It is currently %s" % str(dL_db2.shape)
    )

    expected_dL_dW1 = np.array(
        [[0.38040779, 1.00394945], [0.12798198, 0.39510952], [0.04042118, 0.14198144]]
    )
    assert np.allclose(dL_dW1, expected_dL_dW1), (
        "dL_dW1 is \n" + str(dL_dW1) + " but should be \n" + str(expected_dL_dW1)
    )

    expected_dL_db1 = np.array([0.20784722, 0.08904251, 0.03385342])
    assert np.allclose(dL_db1, expected_dL_db1), (
        "dL_db1 is \n" + str(dL_db1) + " but should be \n" + str(expected_dL_db1)
    )

    expected_dL_dW2 = np.array([[1.38163852, 1.64474415, 1.72847074]])
    assert np.allclose(dL_dW2, expected_dL_dW2), (
        "dL_dW2 is \n" + str(dL_dW2) + " but should be \n" + str(expected_dL_dW2)
    )

    expected_dL_db2 = 1.767495823193705
    expected_dL_db2_array = np.array([1.767495823193705])
    assert np.allclose(dL_db2, expected_dL_db2) or np.allclose(
        dL_db2, expected_dL_db2_array
    ), "dL_db2 is \n" + str(dL_db2) + " but should be \n" + str(expected_dL_db2)

    expected_L = 1.7287272636229514
    assert np.allclose(L, expected_L), (
        "L is \n" + str(L) + " but should be \n" + str(expected_L)
    )

    print("gradients: \033[1;32mtests OK.\033[0m")


################################################################################
# @title Q4 Parameter Updates
################################################################################


def update_params(batchx, batchy, W1, b1, W2, b2, lr=0.01):
    """
    Updates your parameters
        batchx: Mini-batch of features
        batchy: Corresponding mini-batch of labels
        W1: Starting value of W1
        b1: Starting value of b1
        W2: Starting value of W2
        b2: Starting value of b2
        lr: Learning parameter, default to 0.01

    Returns
        W1: New value of W1
        b1: New value of b1
        W2: New value of W2
        b2: New value of b2
        L: Loss value
    """
    dl_dw1, dl_db1, dl_dw2, dl_db2, loss = gradients(batchx, batchy, W1, W2, b1, b2)
    W1 = W1 - lr * dl_dw1
    b1 = b1 - lr * dl_db1
    W2 = W2 - lr * dl_dw2
    b2 = b2 - lr * dl_db2

    return W1, b1, W2, b2, loss


def update_params_test():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1, 0])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])
    W2 = np.array([[0.7, 0.8, 0.9]])
    b2 = np.array([0.3])
    W1, b1, W2, b2, L = update_params(X, y, W1, b1, W2, b2)

    assert W1.shape == (3, 2), "W1 should have shape (3, 2). It is currently %s" % str(
        W1.shape
    )
    assert b1.shape == (3,), "b1 should have shape (3,). It is currently %s" % str(
        b1.shape
    )
    assert W2.shape == (1, 3), "W2 should have shape (1, 3). It is currently %s" % str(
        W2.shape
    )
    assert b2.shape == () or b2.shape == (1,), (
        "b2 should have shape (). It is currently %s" % str(b2.shape)
    )

    expected_W1 = np.array(
        [[0.09619592, 0.18996051], [0.29872018, 0.3960489], [0.49959579, 0.59858019]]
    )
    assert np.allclose(W1, expected_W1), (
        "new W1 is \n" + str(W1) + " but should be \n" + str(expected_W1)
    )

    expected_b1 = np.array([0.09792153, 0.19910957, 0.29966147])
    assert np.allclose(b1, expected_b1), (
        "new b1 is \n" + str(b1) + " but should be \n" + str(expected_b1)
    )

    expected_W2 = np.array([[0.68618361, 0.78355256, 0.88271529]])
    assert np.allclose(W2, expected_W2), (
        "W2 is \n" + str(W2) + " but should be \n" + str(expected_W2)
    )

    expected_b2 = 0.28232504
    expected_b2_array = np.array([0.28232504])
    assert np.allclose(b2, expected_b2) or np.allclose(b2, expected_b2_array), (
        "b2 is \n" + str(b2) + " but should be \n" + str(expected_b2)
    )

    expected_L = 1.7287272636229514
    expected_L_array = np.array([1.7287272636229514])
    assert np.allclose(L, expected_L) or np.allclose(L, expected_L_array), (
        "L is \n" + str(L) + " but should be \n" + str(expected_L)
    )

    print("update_params: \033[1;32mtests OK.\033[0m")


################################################################################
# @title Q5 Training the Neural Network
################################################################################


def train_nn(
    filename,
    hidden_layer_size,
    iters=1e6,
    lr=0.01,
    batch=32,
    W1=None,
    b1=None,
    W2=None,
    b2=None,
    Ls=None,
):
    """
    Reads in data (e.g., "twitter_data.pkl"), initiates parameters, extracts
    features (make sure you've imported "utils.py"), and returns the parameters

    Args:
        filename: Name of the file to read
        hidden_layer_size: Number of hidden units
        iters: Number of iterations to train for
        lr: Learning rate
        W1: Initial weight matrix for the first layer   (optional)
        b1: Initial bias vector for the first layer     (optional)
        W2: Initial weight matrix for the second layer  (optional)
        b2: Initial bias vector for the second layer    (optional)
        Ls: List to store loss values (optional)

    Returns:
        W1, b1, W2, b2, Ls
    """
    X_train, y_train, X_test, y_test = load_data(filename)
    X_train = X_train.T
    d, N = X_train.shape
    y_train = y_train.flatten()

    # init parameters if None
    INIT_W_SCALE = 0.01
    if W1 is None:
        W1 = np.random.randn(hidden_layer_size, d) * INIT_W_SCALE
    if b1 is None:
        b1 = np.zeros(hidden_layer_size)
    if W2 is None:
        W2 = np.random.randn(1, hidden_layer_size) * INIT_W_SCALE
    if b2 is None:
        b2 = np.zeros(1)
    if Ls is None:
        Ls = []

    for i in range(int(iters)):
        split = np.random.choice(N, batch)
        batchx = X_train[:, split]
        batchy = y_train[split]
        W1, b1, W2, b2, loss = update_params(batchx, batchy, W1, b1, W2, b2, lr)
        Ls.append(loss)
        # for observing training progress
        if i % 1000 == 0:
            print(f"iter {i:>6} | loss {loss:.4f}")

    return W1, b1, W2, b2, Ls


def train_nn_test():

    with open("mock_data.pkl", "wb") as f:
        mock_data = {
            "train_x": ["this is a positive tweet", "this is a negative tweet"],
            "train_y": np.array([1, 0]),
            "test_x": ["another positive one", "and a negative one"],
            "test_y": np.array([1, 0]),
            "freqs": {
                ("this", 1): 2,
                ("is", 1): 2,
                ("a", 1): 2,
                ("positive", 1): 2,
                ("tweet", 1): 2,
                ("this", 0): 2,
                ("is", 0): 2,
                ("a", 0): 2,
                ("negative", 0): 2,
                ("tweet", 0): 2,
                ("another", 1): 1,
                ("positive", 1): 1,
                ("one", 1): 1,
                ("and", 0): 1,
                ("a", 0): 1,
                ("negative", 0): 1,
                ("one", 0): 1,
            },
        }
        pickle.dump(mock_data, f)

    W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    b1 = np.array([0.1, 0.2])
    W2 = np.array([[0.7, 0.8]])
    b2 = np.array([0.3])

    W1, b1, W2, b2, Ls = train_nn(
        "mock_data.pkl", 2, iters=8, lr=0.01, batch=2, W1=W1, b1=b1, W2=W2, b2=b2
    )

    assert W1.shape == (2, 3), "W1 should have shape (2, 3). It is currently %s" % str(
        W1.shape
    )
    assert b1.shape == (2, 1) or b1.shape == (2,), (
        "b1 should have shape (2,). It is currently %s" % str(b1.shape)
    )
    assert W2.shape == (1, 2), "W2 should have shape (1, 2). It is currently %s" % str(
        W2.shape
    )
    assert b2.shape == () or b2.shape == (1,), (
        "b2 should have shape (). It is currently %s" % str(b2.shape)
    )

    print("train_nn: \033[1;32mtests OK.\033[0m Please thoroughly test convergence")


def predict_nn(data, W1, b1, W2, b2):
    return inference_2layers(data.T, W1, W2, b1, b2)


################################################################################
# @title Load data and run tests. Then train the model.
################################################################################


def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        if "train_x" in data and "test_x" in data:
            train_x = data["train_x"]
            train_y = data["train_y"]
            test_x = data["test_x"]
            test_y = data["test_y"]
            freqs = data["freqs"]
        elif "X" in data:
            train_x = data["X"]
            train_y = data["y"]
            test_x = []
            test_y = []
            freqs = data["freqs"]
    X_train = np.zeros((len(train_x), 3))
    X_test = np.zeros((len(test_x), 3))
    for i in range(len(train_x)):
        X_train[i] = extract_features(train_x[i], freqs)
    for i in range(len(test_x)):
        X_test[i] = extract_features(test_x[i], freqs)
    y_train = train_y
    y_test = test_y

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    sigmoid_test()
    inference_layer_test()
    inference_2layers_test()
    bce_forward_test()
    gradients_test()
    update_params_test()
    train_nn_test()

    W1, b1, W2, b2, Ls = train_nn("twitter_data.pkl", 6, iters=1e5, lr=1e-5, batch=32)
    np.savez("model_params.npz", W1=W1, b1=b1, W2=W2, b2=b2, Ls=Ls)
