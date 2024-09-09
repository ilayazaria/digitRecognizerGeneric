import random


class Neuron:

    def __init__(self, number_of_previous_neurons, activation_function):
        self.value = 0
        self.weighted_sum = 0
        self.bias = random.uniform(-0.5, 0.5)
        self.activation_function = activation_function
        self.in_weigths = [random.uniform(-0.1, 0.1) for _ in range(number_of_previous_neurons)]
        self.current_bias_gradient = 0
        self.current_weight_gradient = [0 for _ in range(number_of_previous_neurons)]

    def calculate_weighted_sum(self, inputs):
        self.weighted_sum = self.bias
        for input_neuron, weight in zip(inputs, self.in_weigths):
            self.weighted_sum += input_neuron * weight
        return self.weighted_sum

    def activate(self):
        self.value = self.activation_function(self.weighted_sum)
        return self.value
