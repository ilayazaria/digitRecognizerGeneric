from typing import List

from neuron import Neuron
from utils import get_value_from_neuron_list


class NeuralNetwork:

    def __init__(self, learning_rate, hidden_layer_activation, hidden_layer_activation_derivative,
                 output_layer_activation, layers_size, batch_size):
        self.input_layer_size = None
        self.layers = self.build_nn(layers_size, hidden_layer_activation, output_layer_activation)
        self.learning_rate = learning_rate
        self.epsilon_greedy_probabilities = [0.9, 0.1]
        self.hidden_layer_activation_derivative = hidden_layer_activation_derivative
        self.batch_size = batch_size
        self.counter = 1

    def build_nn(self, layers_size, hidden_layer_activation, output_layer_activation):
        if len(layers_size) < 2:
            raise RuntimeError("not enough layers")
        self.input_layer_size = layers_size[0]
        layers = []
        for layer in layers_size[1: -1]:
            layers.append([Neuron(self.input_layer_size, hidden_layer_activation) for _ in range(layer)])
        layers.append([Neuron(self.input_layer_size, output_layer_activation) for _ in range(layers_size[-1])])
        return layers

    def forward_prop(self, inputs: List):
        for neuron in self.layers[0]:
            neuron.calculate_weighted_sum(inputs)
            neuron.activate()
        previous_layer_values = get_value_from_neuron_list(self.layers[0])
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.calculate_weighted_sum(previous_layer_values)
                neuron.activate()
            previous_layer_values = get_value_from_neuron_list(layer)
        return previous_layer_values

    def weight_gradient_calc(self, input_layer, output_layer: List[Neuron], loss, batch_size_reached: bool):
        for i, output_layer_neuron in enumerate(output_layer):
            for j, input_layer_neuron_value in enumerate(input_layer):
                output_layer_neuron.current_weight_gradient[j] += self.learning_rate * input_layer_neuron_value * loss[
                    i]
                if batch_size_reached:
                    output_layer_neuron.in_weigths[j] -= output_layer_neuron.current_weight_gradient[
                                                             j] / self.batch_size
                    output_layer_neuron.current_weight_gradient[j] = 0

    def bias_gradient_calc(self, output_layer: List[Neuron], loss, batch_size_reached: bool):
        for i, output_layer_neuron in enumerate(output_layer):
            output_layer_neuron.current_bias_gradient += self.learning_rate * loss[i]
            if batch_size_reached:
                output_layer_neuron.bias -= (output_layer_neuron.current_bias_gradient / self.batch_size)
                output_layer_neuron.current_bias_gradient = 0

    def neuron_loss_calc(self, input_layer, output_layer, output_layer_loss):
        input_layer_loss = [0] * len(input_layer)
        for i, output_layer_neuron in enumerate(output_layer):
            for j, input_layer_neuron in enumerate(input_layer):
                input_layer_loss[j] += output_layer_loss[i] * output_layer_neuron.in_weigths[j]
        for i, loss in enumerate(input_layer_loss):
            input_layer_loss[i] = loss * self.hidden_layer_activation_derivative(input_layer[i].weighted_sum)
        return input_layer_loss

    def layers_loss_calc(self, output_layer_loss):
        layers_loss = [output_layer_loss]
        current_loss = output_layer_loss
        for input_layer, output_layer in zip(reversed(self.layers[:-1]), reversed(self.layers)):
            current_loss = self.neuron_loss_calc(input_layer, output_layer, current_loss)
            layers_loss.append(current_loss)
        return layers_loss

    def back_prop(self, inputs, output_layer_loss):
        layers_loss = self.layers_loss_calc(output_layer_loss)
        batch_size_reached = self.batch_size == self.counter
        for input_layer, output_layer, layer_loss in zip(reversed(self.layers[:-1]), reversed(self.layers),
                                                         layers_loss):
            self.weight_gradient_calc(get_value_from_neuron_list(input_layer), output_layer, layer_loss,
                                      batch_size_reached)
            self.bias_gradient_calc(output_layer, layer_loss, batch_size_reached)
        self.weight_gradient_calc(inputs, self.layers[0], layers_loss[-1], batch_size_reached)
        self.bias_gradient_calc(self.layers[0], layers_loss[-1], batch_size_reached)
        self.counter += 1
        if batch_size_reached:
            self.counter = 1
