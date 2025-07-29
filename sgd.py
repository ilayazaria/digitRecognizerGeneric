from NeuralNetwork import NeuralNetwork
from utils import ReLU, derivative_ReLU, softmax, no_activation, list_divide_by


class StochasticGradientDescent:

    def __init__(self, number_of_inputs, inputs, expected_values, mini_batch_size):
        self.mini_batch_size = mini_batch_size
        self.neural_network = NeuralNetwork(learning_rate=0.01, hidden_layer_activation=ReLU,
                                            hidden_layer_activation_derivative=derivative_ReLU,
                                            output_layer_activation=no_activation,
                                            layers_size=[number_of_inputs, 28, 10], mini_batch_size=self.mini_batch_size)
        self.inputs = [inputs[i:i + self.mini_batch_size] for i in range(0, len(inputs), self.mini_batch_size)]
        self.expected_values = [expected_values[i:i + self.mini_batch_size] for i in
                                range(0, len(expected_values), self.mini_batch_size)]

    def build_expected_values_array(self, expected_number):
        exp_values = [0] * len(self.neural_network.layers[-1])
        exp_values[expected_number] = 1
        return exp_values

    def output_layer_loss_calc(self, nets, expected_values):
        success = 0
        failures = 0
        output_loss = [0] * 10
        for sample, expected_value in zip(nets, expected_values):
            output_layer_values = [output_neuron[0] for output_neuron in sample[-1]]
            softmax_values = softmax(output_layer_values)
            if softmax_values.index(max(softmax_values)) == expected_value:
                success += 1
            else:
                failures += 1
            output_neurons_error = [o - e for o, e in
                                    zip(softmax_values, self.build_expected_values_array(expected_value))]
            for i, output_neuron_error in enumerate(output_neurons_error):
                output_loss[i] += output_neuron_error
        return success, output_loss

    @staticmethod
    def transform_mini_batch_tensorflow(mini_batch):
        return [list_divide_by(inputs.flatten(), 500) for inputs in mini_batch]

    def train(self):
        i = 0
        number_of_success = 0
        for mini_batch_inputs, mini_batch_expected_values in zip(self.inputs, self.expected_values):
            mini_batch_inputs = self.transform_mini_batch_tensorflow(mini_batch_inputs)
            nets = self.neural_network.forward_prop(mini_batch_inputs)
            curr_success, output_loss = self.output_layer_loss_calc(nets, mini_batch_expected_values)
            number_of_success += curr_success
            self.neural_network.back_prop(nets, output_loss)
            i += 1
            if i == 15:
                percent = number_of_success / (15 * self.mini_batch_size)
                print(percent)
                number_of_success = 0
                i = 0
