import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1 / (1 + np.exp(-input))  # TODO!
        return output

    def sigmoid_derivative(self, x):
        output = x * (1 - x)
        return output
    
    def softmax(self, x):
        if len(x.shape) == 1:
        # Reshape 1D array to 2D array with one column
            x = x.reshape(1, -1)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Suared error to calculate loss error
    def loss_func(self,y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def cross_entropy(y, p):
        """
        Calculates the cross-entropy loss between the predicted probability distribution p and the true probability distribution y.
        Args:           y (numpy.ndarray): A one-hot encoded vector of shape (n_samples, n_classes).
                        p (numpy.ndarray): A probability distribution of shape (n_samples, n_classes).
        """
        n_samples = y.shape[0]
        loss = -np.sum(y * np.log(p)) / n_samples
        return loss

    def _compute_output_weights(self, output_Error, hidden_layer_outputs, deriv_ouput):
        # Gradient weight H5_O7_W[0][0]
        dW_0_0 = output_Error * deriv_ouput[0] * hidden_layer_outputs[0]
        # Gradient weight H6_O7_W[1][0]
        dw_1_0 = output_Error * deriv_ouput[0] * hidden_layer_outputs[1]
        # Gradient weight H5_O8_W[0][1]
        dw_0_1 = output_Error * deriv_ouput[1] * hidden_layer_outputs[0]
        # Gradient weight H6_O8_W[1][1]
        dw_1_1 = output_Error * deriv_ouput[1] * hidden_layer_outputs[1]
        # Gradient weight H5_O9_W[0][2]
        dw_0_2 = output_Error * deriv_ouput[2] * hidden_layer_outputs[0]
        # Gradient weight H5_O9_W[1][2]
        dw_1_2 = output_Error * deriv_ouput[2] * hidden_layer_outputs[1]
        # First way to calculate weights
        #delta_H6_output_weights_ = np.dot(output_delta, np.array(hidden_layer_outputs[0]))
        #delta_H6_output_weights = np.dot(output_delta, np.array(hidden_layer_outputs[1]))
        return np.array([[dW_0_0, dw_0_1, dw_0_2], [dw_1_0, dw_1_1, dw_1_2]])
        
    def _compute_hidden_weights(self, hidden_error, hidden_derivative, inputs):
        # Gradient weight I1_H5_W[0][0]
        dw_0_0 = hidden_error[0] * hidden_derivative[0] * inputs[0]
        # Gradient weight I2_H5_W[1][0]
        dw_1_0 = hidden_error[0] * hidden_derivative[0] * inputs[1]
        # Gradient weight I3_H5_W[2][0]
        dw_2_0 = hidden_error[0] * hidden_derivative[0] * inputs[2] 
        # Gradient weight I3_H5_W[3][0]
        dw_3_0 = hidden_error[0] * hidden_derivative[0] * inputs[3] 
        # Gradient weight I1_H6_W[0][1]
        dw_0_1 = hidden_error[1] * hidden_derivative[1] * inputs[0]     
        # Gradient weight I2_H6_W[1][1]
        dw_1_1 = hidden_error[1] * hidden_derivative[1] * inputs[1]
        # Gradient weight I3_H6_W[2][1]
        dw_2_1 = hidden_error[1] * hidden_derivative[1] * inputs[2]
        # Gradient weight I3_H6_W[3][1]
        dw_3_1 = hidden_error[1] * hidden_derivative[1] * inputs[3] 
        return np.array([[dw_0_0, dw_0_1],[dw_1_0, dw_1_1],[dw_2_0, dw_2_1],[dw_3_0, dw_3_1]])
    
    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = np.dot(inputs, self.hidden_layer_weights[:][:,i])
            output = self.sigmoid(weighted_sum)   # Pass (weighted_sum) to activation function to get the output
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        temp_output = []
        for i in range(self.num_outputs):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = np.dot(hidden_layer_outputs, self.output_layer_weights[:][:,i])
            temp_output.append(weighted_sum)
        soft_output = self.softmax(np.array(temp_output))
        #output_layer_outputs.append(soft_output)

        return hidden_layer_outputs, output_layer_outputs[0][0]
        
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        # Compute gradients of loss with respect to hidden layer weights
        '''dloss_dhidden_output = np.dot(dloss_doutput * doutput_dhidden_output, self.output_layer_weights.T)
        dhidden_output_dhidden_input = self.sigmoid_derivative(hidden_input)
        dhidden_input_dW_hidden = X.T
        dloss_dW_hidden = np.dot(dhidden_input_dW_hidden, dloss_dhidden_output * dhidden_output_dhidden_input)'''

        # Compute error and gradient for Output layer -------------------------------
        output_layer_betas = np.zeros(self.num_outputs)
        # TODO! Calculate output layer betas.
        output_Error = self.loss_func(desired_outputs, output_layer_outputs)
        output_deriv = self.sigmoid_derivative(np.array(output_layer_outputs))
        output_delta = output_Error * output_deriv  # Calculate gradient with respect Output layer Actication (O7, O8, O9)
        output_layer_betas = output_delta
        print('OL betas: ', output_layer_betas)

        # Compute error and gradient for Hidden layer -------------------------------
        hidden_layer_betas = np.zeros(self.num_hidden)
        # TODO! Calculate hidden layer betas.
        hidden_error = np.dot(output_delta, self.output_layer_weights.T)
        hidden_deriv = self.sigmoid_derivative(np.array(hidden_layer_outputs))
        hidden_delta = hidden_error * hidden_deriv  # Calculate gradient with respect Input Activation (H5,H6)
        hidden_layer_betas = hidden_delta
        print('HL betas: ', hidden_layer_betas)

        # Clculate gradient weights for ouput layer -------------------------------
        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # TODO! Calculate output layer weight changes.
        # Calculate gradient with respect Output layer weights
        delta_output_layer_weights = self._compute_output_weights(output_Error, hidden_layer_outputs, output_deriv)

        # Calculate gradient weights for Hidden layer ----------------------------------
        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # TODO! Calculate hidden layer weight changes.
        # Calculate gradient with respect Hiddenn layer weights
        delta_hidden_layer_weights = self._compute_hidden_weights(hidden_error, hidden_deriv, inputs)

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        # TODO! Update the weights.
        # Calculate the weights and bias 
        output_weights_update = self.learning_rate * delta_output_layer_weights
        hidden_weights_update = self.learning_rate * delta_hidden_layer_weights

        # Update the weights and bias 
        self.hidden_layer_weights += hidden_weights_update
        self.output_layer_weights += output_weights_update

        print('Placeholder')

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            #for i, instance in enumerate(instances):
            for idx in range(len(instances)):
                instance = instances[idx]
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs)
                predicted_class = None  # TODO!
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            print('Hidden layer weights \n', self.hidden_layer_weights)
            print('Output layer weights  \n', self.output_layer_weights)

            # TODO: Print accuracy achieved over this epoch
            acc = None
            print('acc = ', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            #print(output_layer_outputs)
            # Convert the probabilities to a binary target class
            predicted_class = np.argmax(output_layer_outputs)  # TODO! Should be 0, 1, or 2.
            predictions.append(predicted_class)
        return predictions
