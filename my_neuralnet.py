import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # I set the number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialization of weights
        self.weights_input_to_hidden = np.random.normal(
            0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(
            0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))

        self.lr = learning_rate

        # I set the sigmoid function using a lambda expression
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

        # I can also define sigmoid function with a method
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        self.activation_function_by_method = sigmoid

    def train(self, features, targets):
        '''
        Train the network on batch of features and targets.

        Arguments
        ---------

        features: 2D array, each row is one data record, each column is a feature
        targets: 1D array of target values
        '''

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # Call to forward function
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            # Call to backpropagation function
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        # Call to a method who update weights along the training
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        '''
        Funtion that implements the forward algorithm

        Arguments
        ---------

        X: features batch
        '''
        ### Forward pass ###

        # signals into hidden layer, hidden_inputs.shape: (2,)
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)

        # signals from hidden layer, hidden_outputs.shape: (2,)
        hidden_outputs = self.activation_function(hidden_inputs)

        # signals into final output layer, final_inputs.shape: (1,)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        # signals from final output layer
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        '''
        Implement backpropagation

        Arguments
        ---------

        final_outputs: output from forward pass
        y: target (i. e. label) batch
        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers
        '''
        ### Backward pass ###

        # Output layer error is the difference between desired target and actual output.
        error = y - final_outputs

        # Backpropagated error term from output
        output_error_term = error * final_outputs * (1 - final_outputs)

        # Hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, delta_weights_h_o.T)
        print("Shape of output_error_term: {}".format(output_error_term.shape))
        print("Shape of delta_weights_h_o: {}".format(delta_weights_h_o.shape))

        # Backpropagated error term from hidden layer
        hidden_error_term = hidden_error * \
            hidden_outputs * (1 - hidden_outputs)

        # Weight step (input to hidden)
        print("Shape of hidden_error_term: {}".format(hidden_error_term.shape))
        print("Shape of delta_weights_i_h: {}".format(delta_weights_i_h.shape))
        print("Shape of X: {}".format(X.shape))
        delta_weights_i_h += hidden_error_term * X[:, None]

        # Weight step (hidden to output)
        print("Shape of delta_weights_h_o: {}".format(delta_weights_h_o.shape))
        print("Shape of hidden_outputs: {}".format(hidden_outputs.shape))
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        '''
        Update weights on gradient descent step

        Arguments
        ---------

        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers
        n_records: number of records
        '''

        # Update of hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records

        # Update of input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        '''
        Run a forward pass through the network with input features

        Arguments
        ---------

        features: 1D array of feature values
        '''

        ###Forward pass implementation ###

        # signals into hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)

        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        #  signals from final output layer
        final_outputs = final_inputs

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1