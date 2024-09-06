import math
from typing import List

from neuron import Neuron


def no_activation(z):
    return z


def ReLU(Z):
    return max(0, Z)


def softmax(inputs):
    temp = [math.exp(v) for v in inputs]
    total = sum(temp)
    return [t / total for t in temp]


def derivative_ReLU(activation):
    return activation > 0


def get_value_from_neuron_list(neuron_list: List[Neuron]):
    return [neuron.value for neuron in neuron_list]


def get_bias_from_neuron_list(neuron_list: List[Neuron]):
    return [neuron.bias for neuron in neuron_list]


def get_weight_from_neuron_list(neuron_list: List[Neuron]):
    return [neuron.in_weigths for neuron in neuron_list]


def list_divide_by(num_lst, number):
    return [value / number for value in num_lst]
