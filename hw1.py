###### Your ID ######
# ID1: 208127423
# ID2: 212081129
#####################

# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform Standardization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    x_mean = np.mean(X, axis=0)
    x_sigma = np.std(X, axis=0)
    y_mean = np.mean(y)
    y_sigma = np.std(y)

    X = (X - x_mean) / x_sigma
    y = (y - y_mean) / y_sigma
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # get num of rows to know how many ones are needed

    number_of_rows = X.shape[0]
    ones = np.ones((number_of_rows, 1))
    # concat the two matrices placing ones to the left of all data of X
    X = np.hstack((ones, X))

    return X


def compute_loss(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """
    
    predictions = np.matmul(X, theta)  # Matrix multiplication to get predictions
    J = np.sum((predictions - y) ** 2) / (2 * X.shape[0])  # Compute loss

    return J


def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using
    the training set. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    n = X.shape[0]
    #prediction is the multiplication of the theta by the values of X
    for i in range(num_iters):
        predictions = X@theta 
        #errors is difference from real values
        errors = predictions-y
        gradient = (np.transpose(X)@errors)/n
        theta -= eta * gradient
        loss = compute_loss(X,y,theta)
        J_history.append(loss)

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################
    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y

    return pinv_theta


def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop
    the learning process once the improvement of the loss value is smaller
    than epsilon. This function is very similar to the gradient descent
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent with stop condition optimization algorithm.  #
    ###########################################################################
    for i in range(max_iter):
        delta_loss = (1/X.shape[0]) * X.T @ (X @ theta - y)
        loss = compute_loss(X, y, theta)
        J_history.append(loss)
        theta = theta - eta*delta_loss
        if i > 2 and abs(J_history[i] - J_history[i-1]) < epsilon:
            break
        #print(J_history)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using
    the training dataset. Maintain a python dictionary with eta as the
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """

    etas = [
        0.00001,
        0.00003,
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1,
        2,
        3,
    ]
    eta_dict = {}  # {eta_value: validation_loss}
    for eta_value in etas:
        input_theta = np.zeros(X_train.shape[1])
        theta, _ = gradient_descent_stop_condition(X_train,y_train, input_theta, eta_value, iterations)
        loss = compute_loss(X_val,y_val, theta) 
        eta_dict[eta_value] = loss

    return eta_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to
    select the most relevant features for a predictive model. The objective
    of this algorithm is to improve the model's performance by identifying
    and using only the most relevant features, potentially reducing overfitting,
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    rows, num_features = X_train.shape
    all_features_indices = set(range(num_features))
    min_loss = float("inf") 
    target_num_features = min(5,num_features)
    while len(selected_features) < target_num_features:
        current_best_index = -1
        remaining_features_indices = all_features_indices - set(selected_features)
        if not remaining_features_indices:
            print("no more features to check")
            break
        for feature_index in remaining_features_indices:
            current_selection_indicies = selected_features + [feature_index]
            X_train_subset = X_train[:, current_selection_indicies]
            X_val_subset = X_val[:, current_selection_indicies]
            input_theta = np.zeros(X_train_subset.shape[1])
            theta, _ = gradient_descent_stop_condition(X_train_subset,y_train, input_theta, best_eta, iterations)
            feature_loss = compute_loss(X_val_subset,y_val,theta)
            if feature_loss<min_loss:
                min_loss = feature_loss
                current_best_index = feature_index
        if current_best_index!=-1:
            selected_features.append(current_best_index)
            print(f"Iteration {len(selected_features)}: Added feature {current_best_index}, Validation Loss: {min_loss:.4f}")
        else:
             print("No feature improved performance. Stopping selection.")
             break 
        
    print(f"\nSelected top {len(selected_features)} feature indices: {selected_features}")
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################

    for col in df_poly.columns:
        #vectorized squaring of the cols
        df_poly[f"{col}^2"] = df_poly[col] ** 2
    
    # Add interaction terms between all pairs of columns
    columns = df_poly.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            df_poly[f"{col1}*{col2}"] = df_poly[col1] * df_poly[col2]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
