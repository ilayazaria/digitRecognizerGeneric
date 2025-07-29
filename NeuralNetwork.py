from typing import List

from Neuron import Neuron
from utils import get_value_from_neuron_list


class NeuralNetwork:

    def __init__(self, learning_rate, hidden_layer_activation, output_layer_activation,
                 hidden_layer_activation_derivative, layers_size, mini_batch_size):
        self.input_layer_size = layers_size[0]
        self.layers = self.build_nn(layers_size, hidden_layer_activation, output_layer_activation)
        self.learning_rate = learning_rate
        self.epsilon = 1
        self.hidden_layer_activation_derivative = hidden_layer_activation_derivative
        self.mini_batch_size = mini_batch_size

    def build_nn(self, layers_size, hidden_layer_activation, output_layer_activation):
        layers = []
        for layer in layers_size[1: -1]:
            layers.append([Neuron(self.input_layer_size, hidden_layer_activation) for _ in range(layer)])
        layers.append([Neuron(self.input_layer_size, output_layer_activation) for _ in range(layers_size[-1])])
        return layers

    def forward_prop(self, mini_batch):
        mini_batch_net = []
        for inputs in mini_batch:
            sample_net = [inputs]
            previous_layer_values = inputs
            for layer in self.layers:
                layer_net = []
                for neuron in layer:
                    weighted_sum = neuron.calculate_weighted_sum(previous_layer_values)
                    activation = neuron.activate()
                    layer_net.append((weighted_sum, activation))
                sample_net.append(layer_net)
                previous_layer_values = [values[1] for values in layer_net]
            mini_batch_net.append(sample_net)
        return mini_batch_net

    def weight_gradient_calc(self, input_layer, output_layer: List[Neuron], loss):
        for i, output_layer_neuron in enumerate(output_layer):
            for j, input_layer_neuron_value in enumerate(input_layer):
                output_layer_neuron.in_weigths[j] -= (self.learning_rate * input_layer_neuron_value * loss[
                    i]) / self.mini_batch_size

    def bias_gradient_calc(self, output_layer: List[Neuron], loss):
        for i, output_layer_neuron in enumerate(output_layer):
            output_layer_neuron.bias -= (self.learning_rate * loss[i]) / self.mini_batch_size

    def neuron_loss_calc(self, input_layer, output_layer, output_layer_loss):
        input_layer_loss = [0] * len(input_layer)
        for i, output_layer_neuron in enumerate(output_layer):
            for j, input_layer_neuron in enumerate(input_layer):
                input_layer_loss[j] += output_layer_loss[i] * output_layer_neuron.in_weigths[j]
        for i, loss in enumerate(input_layer_loss):
            input_layer_loss[i] = loss * self.hidden_layer_activation_derivative(input_layer[i].value)
        return input_layer_loss

    def layers_loss_calc(self, output_layer_loss):
        layers_loss = [output_layer_loss]
        current_loss = output_layer_loss
        for input_layer, output_layer in zip(reversed(self.layers[:-1]), reversed(self.layers)):
            current_loss = self.neuron_loss_calc(input_layer, output_layer, current_loss)
            layers_loss.append(current_loss)
        return layers_loss

    def back_prop(self, mini_batch, output_layer_loss):
        for net in mini_batch:
            layers_loss = self.layers_loss_calc(output_layer_loss)
            for input_layer, output_layer, layer_loss in zip(reversed(self.layers[:-1]), reversed(self.layers),
                                                             layers_loss):
                self.weight_gradient_calc(get_value_from_neuron_list(input_layer), output_layer, layer_loss)
                self.bias_gradient_calc(output_layer, layer_loss)
            self.weight_gradient_calc(net[0], self.layers[0], layers_loss[-1])
            self.bias_gradient_calc(self.layers[0], layers_loss[-1])
