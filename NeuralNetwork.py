#Code source adapted from
#http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

class NeuralNetwork():

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_examples = len(X) # training set size
        self.nn_input_dim = 2 # input layer dimensionality
        self.nn_output_dim = 2 # output layer dimensionality

        # Gradient descent parameters (I picked these by hand)
        self.epsilon = 0.01 # learning rate for gradient descent
        self.reg_lambda = 0.01 # regularization strength

    # Helper function to evaluate the total loss on the dataset
    def calculate_loss(self, model):
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        #W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Forward propagation to calculate our predictions
        z1 = self.X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), self.y])
        data_loss = np.sum(corect_logprobs)

        # Add regularization term to loss (optional)
        data_loss += self.reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        return 1./self.num_examples * data_loss

    # Helper function to predict an output (0 or 1)
    def predict(self, model, x):
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        #W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return np.argmax(probs, axis=1)

    # This function learns parameters for the neural network and returns the model.
    # - nn_hdim: Number of nodes in the hidden layer
    # - num_passes: Number of passes through the training data for gradient descent
    # - print_loss: If True, print the loss every 1000 iterations
    def build_model(self, nn_hdim, num_passes=20000, print_loss=False):
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, self.nn_output_dim))

        # This is what we return at the end
        model = {}
    
        # Gradient descent. For each batch...
        for i in range(0, num_passes):

            # Forward propagation
            z1 = self.X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(self.num_examples), self.y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(self.X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2
        
            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" %(i, self.calculate_loss(model)))
    
        return model

    #Plot decision boundary
    def plot_decision_boundary(self, pred_func):
        #Set min and max values and give it some padding
        x_min = self.X[:, 0].min() - .5
        x_max = self.X[:, 0].max() + .5
        y_min = self.X[:, 1].min() - .5
        y_max = self.X[:, 1].max() + .5

        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.Spectral)
