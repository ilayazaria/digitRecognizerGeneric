from neural_network import NeuralNetwork
from utils import ReLU, derivative_ReLU, softmax, no_activation, list_divide_by


class StochasticGradientDescent:

    def __init__(self, inputs, expected_values):
        self.neural_network = NeuralNetwork(learning_rate=0.01, hidden_layer_activation=ReLU,
                                            hidden_layer_activation_derivative=derivative_ReLU,
                                            output_layer_activation=no_activation,
                                            layers_size=[len(inputs), 28, 10], batch_size=64)
        self.inputs = inputs
        self.expected_values = expected_values

    def build_expected_values_array(self, expected_number):
        exp_values = [0] * len(self.neural_network.layers[-1])
        exp_values[expected_number] = 1
        return exp_values

    def output_layer_loss_calc(self, softmax_values, expected_number):
        return [o - e for o, e in
                zip(softmax_values, self.build_expected_values_array(expected_number))]

    def train(self):
        i = 1
        success = 0
        failure = 0
        episode = []
        for instance, expected_value in zip(self.inputs, self.expected_values):
            pixels = list_divide_by(instance.flatten(), 100)
            output_layer_values = self.neural_network.forward_prop(pixels)
            softmax_values = softmax(output_layer_values)
            output_layer_loss = self.output_layer_loss_calc(softmax_values, expected_value)
            self.neural_network.back_prop(pixels, output_layer_loss)
            if softmax_values.index(max(softmax_values)) == expected_value:
                success += 1
            else:
                failure += 1
            if i % 2000 == 0:
                percentage = success / (success + failure) * 100
                print(str(percentage) + "%")
                success = 0
                failure = 0
            i += 1
