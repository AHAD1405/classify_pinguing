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

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = np.dot(inputs, self.hidden_layer_weights[:][:,i])
            output = self.sigmoid(weighted_sum)   # Pass (weighted_sum) to activation function to get the output
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = np.dot(hidden_layer_outputs, self.output_layer_weights[:][:,i])
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):
        
        ''' Method 2 '''
        '''output_error = actual_outputs - desired_outputs
        output_delta = output_error * self.sigmoid_derivative(output_error)
        hidden_error2 = np.dot(output_delta, self.hidden_layer_weights.T)
        hidden_delta2 = hidden_error2 *  self.sigmoid_derivative(hidden_layer2)  
        hidden_error1 = np.dot(hidden_delta2, self.hidden_layer_weights.T)
        hidden_delta1 = hidden_error1 *  self.sigmoid_derivative(hidden_layer_outputs) 
        
        # Update weights and biases
        output_weights -= self.learning_rate * np.dot(hidden_error2.T, output_delta)
        output_bias -= self.learning_rate * np.sum(output_delta, axis=0)
        hidden_weights2 -= self.learning_rate * np.dot(hidden_layer1.T, hidden_delta2)
        hidden_bias2 -= self.learning_rate * np.sum(hidden_delta2, axis=0)
        hidden_weights1 -= self.learning_rate * np.dot(inputs.T, hidden_delta1)
        hidden_bias1 -= self.learning_rate * np.sum(hidden_delta1, axis=0)'''

        ''' Method 1'''
        # Compute gradients of loss with respect to output layer weights. In other word, Compute loss derivative with respect to output layer
        '''dloss_doutput = actual_outputs - desired_outputs

        doutput_dhidden_output = self.sigmoid_derivative(dloss_doutput)
        dhidden_output_dW_output = hidden_layer_outputs.T
        dloss_dW_output = np.dot(dhidden_output_dW_output, dloss_doutput * doutput_dhidden_output)

        # Compute gradients of loss with respect to hidden layer weights
        dloss_dhidden_output = np.dot(dloss_doutput * doutput_dhidden_output, self.output_layer_weights.T)
        dhidden_output_dhidden_input = self.sigmoid_derivative(hidden_input)
        dhidden_input_dW_hidden = X.T
        dloss_dW_hidden = np.dot(dhidden_input_dW_hidden, dloss_dhidden_output * dhidden_output_dhidden_input)'''

        output_layer_betas = np.zeros(self.num_outputs)
        # TODO! Calculate output layer betas.
        print('OL betas: ', output_layer_betas)

        hidden_layer_betas = np.zeros(self.num_hidden)
        # TODO! Calculate hidden layer betas.
        print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # TODO! Calculate output layer weight changes.

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # TODO! Calculate hidden layer weight changes.

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        # TODO! Update the weights.
        update_output_weights = self.learning_rate * delta_output_layer_weights
        update_hidden_weights = self.learning_rate * delta_hidden_layer_weights
        print('Placeholder')

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            #for i, instance in enumerate(instances):
            for idx in range(len(instances)):
                instance = instances[idx]
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[idx])
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
            predicted_class = None  # TODO! Should be 0, 1, or 2.
            predictions.append(predicted_class)
        return predictions