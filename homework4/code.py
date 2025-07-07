import numpy as np
import csv
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression



class ANNClassification:

    # implement me okay
    # for the regularization we need to modify only the update step, and cost funciton 
    def __init__(self, units , lambda_ = 0, lr = 0.2, epochs = 10000, activations = None):

        np.random.seed(42)
        self.weights_= []
        self.biases = []
        self.hidden_layers = units

        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.activations = activations or ['sigmoid'] * len(units)
        self.losses = []
        self.zsval = []
        self.asval = []



    def forward(self, X, task='classification'):
        a, zsval, asval = forward(X, self.weights_, self.biases, self.activations, task)
        self.zsval = zsval
        self.asval = asval
        return a

 
    def compute_log_loss(self, Y, A_last): 

        m = Y.shape[1] # number of samples
        eps = 1e-9
        cost = - np.sum(Y * np.log(A_last + eps)) / m
        reg = (self.lambda_ / (2 * m)) * sum(np.sum(W**2) for W in self.weights_)
        self.losses.append(cost + reg)
        # print(f"Current log loss: {cost}")
        return cost + reg
    
    def backpropagation(self, Y): 

        self.grads_w, self.grads_b = backpropagation(self.weights_, self.biases, self.L, self.activations, self.asval, Y)

    def update_params(self, lr): 

        update_params(self, lr)

    def fit(self, X, y):

       return fit(self, X, y)

    def predict(self, X):

        return predict(X, self.weights_, self.biases, self.activations)
 
    def check_gradient(self, X, y, layer_index=0):

        check_gradient(self, X, y, layer_index=0)

  

class ANNRegression:
    # implement me too, please alright
    # what is different in regression config : loss funciton mse, output is 1d no one hot encoded, 
    # and output layer consists of 1 neuron 
    # no need for softmax, last activation fucntion is liner

   def __init__(self, units, lambda_=0, lr=0.2, epochs=20000, activations=None):
        
        np.random.seed(42)
        self.weights_ = []
        self.biases = []
        self.hidden_layers = units
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.activations = activations or ['sigmoid'] * len(units)
        self.losses = []
        self.zsval = []
        self.asval = []

   def forward(self, X, task='regression'):
        a, zsval, asval = forward(X, self.weights_, self.biases, self.activations, task)
        self.zsval = zsval
        self.asval = asval
        return a

   def compute_mse_loss(self, Y, A_last):
        m = Y.shape[1]
        cost = np.sum((Y - A_last)**2) / (2 * m)
        reg = (self.lambda_ / (2 * m)) * sum(np.sum(W**2) for W in self.weights_)
        self.losses.append(cost + reg)
        return cost + reg

   def backpropagation(self, Y):
        self.grads_w, self.grads_b = backpropagation(self.weights_, self.biases, self.L, self.activations, self.asval, Y)

   def update_params(self, lr):
        update_params(self, lr)

   def fit(self, X, y):
        return fit(self, X, y, task='regression')

   def predict(self, X):
        return predict(X, self.weights_, self.biases, self.activations, task='regression')

   def weights(self):
        matrices = []
        for W, b in zip(self.weights_, self.biases):
            wb_mat = np.hstack([b, W])
            matrices.append(wb_mat.T)
        return matrices

# SHARED CODE 

def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a): 
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
    return (a > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(a, alpha=0.01):
    return np.where(a > 0, 1, alpha)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1 - a**2

def forward(X, weights_, biases, activations, task='classification'):

    zsval = []
    asval = [X]
    a = X
    for i in range(len(weights_)):
        z = weights_[i] @ a + biases[i]
        if i == len(weights_) - 1:
            if task == 'classification':
                a = softmax(z)
            else:
                a = z  # linear for regression
        else:
            if activations[i] == 'relu':
                a = relu(z)
            elif activations[i] == 'leaky_relu':
                a = leaky_relu(z)
            elif activations[i] == 'tanh':
                a = tanh(z)
            else:
                a = sigmoid(z)

        zsval.append(z)
        asval.append(a)

    return a, zsval, asval

def backpropagation(weights_, biases, L, activations, asval, Y): 

        m = Y.shape[1]
        grads_w = [np.zeros_like(W) for W in weights_]
        grads_b = [np.zeros_like(b) for b in biases]

        A_last = asval[-1]
        dZ = (A_last - Y) / m
        A_prev = asval[-2]
        grads_w[-1] = dZ @ A_prev.T
        grads_b[-1] = np.sum(dZ, axis=1, keepdims=True)


        for l in range(L-2, -1, -1):
            W_next = weights_[l+1]
            dA = W_next.T @ dZ

            A_curr = asval[l+1]

            if activations[l] == 'relu':
                    derivative = relu_derivative(A_curr)

            elif activations[l] == 'leaky_relu': 
                    derivative = leaky_relu_derivative(A_curr)

            elif activations[l] == 'tanh':
                    derivative = tanh_derivative(A_curr)
            else: 
                    derivative = sigmoid_derivative(A_curr)

            dZ = dA * derivative
            A_prev = asval[l]
            grads_w[l] = dZ @ A_prev.T
            grads_b[l] = np.sum(dZ, axis=1, keepdims=True)

        return grads_w, grads_b


def predict(X,  weights_, biases, activations, task='classification'):
    X = X.T
    A_last, _, _ = forward(X, weights_, biases, activations, task)
    if task == 'regression':
        return A_last.flatten()
    else:
        return A_last.T

def fit(self, X, y, task='classification'):

    X = X.T
    self.m = len(y)
    input_layer = X.shape[0]
    output_layer = np.max(y) + 1 if task == 'classification' else 1
    self.layer_sizes = [input_layer] + self.hidden_layers + [output_layer]
    self.L = len(self.layer_sizes) - 1

    if task == 'classification':
        Y = np.zeros((output_layer, y.shape[0]))
        Y[y, np.arange(y.shape[0])] = 1
    else:
        Y = y.reshape(1, -1)

    self.weights_ = []
    self.biases = []
    for i in range(self.L):
        w = np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * 1
        b = np.zeros((self.layer_sizes[i+1], 1))
        self.weights_.append(w)
        self.biases.append(b)

    for epoch in range(self.epochs):

        A_last = self.forward(X, task)

        if task == 'classification':
            loss = self.compute_log_loss(Y, A_last)

        else:
            loss = self.compute_mse_loss(Y, A_last)

        if loss < 1e-8:
            break
        
        self.backpropagation(Y)
        self.update_params(self.lr)

    return self

def update_params(self, lr):

    for l in range(self.L):
        self.weights_[l] -= lr * (self.grads_w[l] + (self.lambda_ / self.m) * self.weights_[l])
        self.biases[l] -= lr * self.grads_b[l]

def check_gradient(self, X, y, layer_index=0):
        
        epsilon = 1e-5

        x = X[0].reshape(-1, 1) # use only the first sample
        y_class = y[0]

        output_layer = np.max(y) + 1
        Y = np.zeros((output_layer, 1))
        Y[y_class, 0] = 1

        self.forward(x)
        self.backpropagation(Y)
        analytical_grad = self.grads_w[layer_index]

        W = self.weights_[layer_index].copy()
        num_grad = np.zeros_like(W)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                original = W[i, j]

                W[i, j] = original + epsilon
                self.weights_[layer_index] = W
                loss_plus = self.compute_log_loss(Y, self.forward(x))

                W[i, j] = original - epsilon
                self.weights_[layer_index] = W
                loss_minus = self.compute_log_loss(Y, self.forward(x))

                W[i, j] = original  # reset
                num_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        self.weights_[layer_index] = W  

        diff = np.linalg.norm(num_grad - analytical_grad) / (np.linalg.norm(num_grad + analytical_grad) + 1e-10)
        if diff < 1e-5:
                print(f"Relative error layer {layer_index}: {diff:.10f}")

# data reading

def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def doughnut():
    legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y


def squares():
    legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
    return X, y


def fit_data(): 

    X_squares, y_squares = squares()
    X_doughnut, y_doughnut = doughnut()

    fitter_squares = ANNClassification(units=[5], lambda_= 0, lr = 0.1, epochs=50000)
    model_squares = fitter_squares.fit(X_squares, y_squares)
    y_pred_squares = model_squares.predict(X_squares)
    y_pred_squares = np.argmax(y_pred_squares, axis=1)
    predictions = (y_pred_squares > 0.5).astype(int)
    accuracy = (predictions == y_squares).mean()
    print(f"Accuracy on squares data: {accuracy}")

    model_donut = ANNClassification(units=[5], lr=0.1, epochs=50000)
    model_donut.fit(X_doughnut, y_doughnut)

    # Train model on squares.tab
    model_squares = ANNClassification(units=[5], lr=0.1, epochs=50000)
    model_squares.fit(X_squares, y_squares)

    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model_donut.losses, label='Doughnut', color='royalblue')
    plt.title('Doughnut data')
    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel('Log Loss', fontsize = 14)
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(model_squares.losses, label='Squares', color='crimson')
    plt.title('Square data')
    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel('Log Loss', fontsize = 14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("perfect_fit.pdf")
    plt.show()

def compare_models(task = 'classification', units = [5], activation = 'tanh',  lr = 0.1, epochs = 5000, name = "plot.pdf" ):
 
    
    if task == 'classification': 

        X, y = doughnut()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        my_model = ANNClassification(units=units, lr=0.1, epochs=8000, activations=['tanh'])
        my_model.fit(X_scaled, y)
        my_preds = np.argmax(my_model.predict(X_scaled), axis=1)

        sklearn_model = MLPClassifier(hidden_layer_sizes=units, activation='tanh', solver='sgd', learning_rate_init=0.1, max_iter=8000, random_state=42, early_stopping = False)
        sklearn_model.fit(X_scaled, y)

        sklearn_preds = sklearn_model.predict(X_scaled)
        my_accuracy = np.mean(my_preds == y)
        sklearn_accuracy = np.mean(sklearn_preds == y)

        print(f"Custom ANN Classifier - Accuracy: {my_accuracy:.4f}, final log-loss: {my_model.losses[-1]:.4f}")
        print(f"Sklearn MLP Classifier - Accuracy: {sklearn_accuracy:.4f}, final log-loss: {sklearn_model.loss_:.4f}")
    
    else: 

        # generate simple synthetic dataset 2 features
        X, y = make_regression(n_samples=200, n_features=2, noise=5.0, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        my_model = ANNRegression(units=[5], lr=0.001, epochs=8000, activations=[activation])
        my_model.fit(X_scaled, y)
        my_preds = my_model.predict(X_scaled)
        my_mse = np.mean((y - my_preds) ** 2)

        sklearn_model = MLPRegressor(hidden_layer_sizes=(5,), activation=activation, solver='sgd', learning_rate_init=0.001, max_iter=8000, random_state=42, early_stopping=False)
        sklearn_model.fit(X_scaled, y)
        sklearn_preds = sklearn_model.predict(X_scaled)
        sklearn_mse = np.mean((y - sklearn_preds) ** 2)

        print(f"Custom ANN Regression final mse: {my_mse:.4f}")
        print(f"Sklearn MLP Regression  final mse: {sklearn_mse:.4f}")

    plot_training_loss(my_model, sklearn_model, name, task=task)
   

def plot_training_loss(my_model, sklearn_model, name = "plot.pdf" , task = 'classification'):

    if task == 'classification': 
        label_mlp = 'Sklearn MLPClassifier Loss'
        loss = 'Log-loss'
    else: 
        label_mlp = 'Sklearn MLPRegression Loss'
        loss = 'MSE'

    plt.figure(figsize=(8,5))
    plt.plot(my_model.losses, label='Custom ANN Loss', linewidth=2)
    plt.plot(sklearn_model.loss_curve_, label=label_mlp, linewidth=2, linestyle='--')

    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel(loss, fontsize = 14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()

def different_activation_functions():
    
    X_doughnut, y_doughnut = doughnut()
    fitter_doughnut = ANNClassification(units=[5, 5, 5], lambda_= 0, lr = 0.1, epochs=50000, activations = ['relu', 'tanh', 'sigmoid'])
    model_doughnut = fitter_doughnut.fit(X_doughnut, y_doughnut)
    y_pred_doughnut = model_doughnut.predict(X_doughnut)
    y_pred_doughnut = np.argmax(y_pred_doughnut, axis=1)
    predictions = (y_pred_doughnut > 0.5).astype(int)
    accuracy = (predictions == y_doughnut).mean()
    print(f"Accuracy on doughnut data: {accuracy}")

def test_regularization():
    X_doughnut, y_doughnut = doughnut()

    model_no_reg = ANNClassification(units=[5], lambda_=0.0, lr=0.1, epochs=5000, activations=['tanh'])
    model_no_reg.fit(X_doughnut, y_doughnut)
    preds_no_reg = np.argmax(model_no_reg.predict(X_doughnut), axis=1)
    acc_no_reg = (preds_no_reg == y_doughnut).mean()

    model_with_reg = ANNClassification(units=[5], lambda_=1.0, lr=0.1, epochs=5000, activations=['tanh'])
    model_with_reg.fit(X_doughnut, y_doughnut)
    preds_with_reg = np.argmax(model_with_reg.predict(X_doughnut), axis=1)
    acc_with_reg = (preds_with_reg == y_doughnut).mean()

    print(f"Accuracy without regularization: {acc_no_reg:.4f}")
    print(f"Accuracy with regularization: {acc_with_reg:.4f}")

    
def compare_gradients():

    X_doughnut, y_doughnut = doughnut()
    fitter_doughnut = ANNClassification(units=[5], lambda_= 0, lr = 0.1, epochs=50000)
    model_doughnut = fitter_doughnut.fit(X_doughnut, y_doughnut)
    y_pred_doughnut = model_doughnut.predict(X_doughnut)
    y_pred_doughnut = np.argmax(y_pred_doughnut, axis=1)
    predictions = (y_pred_doughnut > 0.5).astype(int)
    accuracy = (predictions == y_doughnut).mean()
    print(f"Accuracy on doughnut data: {accuracy}")
    
    model_doughnut.fit(X_doughnut, y_doughnut)
    model_doughnut.check_gradient(X_doughnut, y_doughnut, layer_index=0)


if __name__ == "__main__":


    # PART 1
    # COMPARE GRADIENTS
    compare_gradients()

    # fit on sqaures and doughnut
    fit_data()

    # PART2 
    # ADD SUPPORT FOR REGULARIZATION
    test_regularization()

    # DIFFERENT ACTIVATION FUCNTIONS
    different_activation_functions()

    # COMPARE SOLUTIONS
    compare_models(task='regression', name = "regression_tanh.pdf")
    compare_models(task='classification', name = "classifcation_tanh.pdf")
    compare_models(task='regression', activation = 'relu', name = "regression_relu.pdf")
    compare_models(task='classification', activation = 'relu' , name = "classifcation_relu.pdf")