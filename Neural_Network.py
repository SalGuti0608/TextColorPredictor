import random
import time
import math

color_input = [[235, 20, 20],  # Red
               [49, 64, 164],  # Dark Blue
               [84, 207, 37],  # Light Green
               [247, 53, 220],  # Pink
               [189, 17, 248],  # Purple
               [247, 132, 2],   # Orange
               [244, 251, 11],  # Yellow
               [15, 139, 248],  # Light Blue
               [15, 248, 170],  # Aqua
               [248, 199, 19],  # Orange/Yellow
               [181, 248, 15],  # Neon Green
               [137, 161, 99],  # Moss Green
               [99, 136, 161],  # Lake
               [111, 50, 132],  # Dark Purple
               [225, 42, 81],  # Fuchsia
               [0, 0, 0],  # Black
               [255, 255, 255]]  # White

# White Black
output_correction = [[1, 0], # Red
                     [1, 0],  # Dark Blue
                     [0, 1],  # Light Green
                     [0, 1],  # Pink
                     [1, 0],  # Purple
                     [0, 1],   # Orange
                     [0, 1],  # Yellow
                     [0, 1],  # Light Blue
                     [0, 1],  # Aqua
                     [0, 1],  # Orange/Yellow
                     [0, 1],  # Neon Green
                     [0, 1],  # Moss Green
                     [1, 0],  # Lake
                     [1, 0],  # Dark Purple
                     [1, 0],  # Fuchsia
                     [1, 0],  # Black
                     [0, 1]]  # White


class Neuron:

    def __init__(self, num):
        self.weight_list = []
        self.value_list = []
        self.num_inputs = num
        self.sum = 0  # Value
        self.output = 0

        for x in range(0, self.num_inputs - 1):
            self.weight_list.append(random.uniform(-0.1, 0.1))
            self.value_list.append(0)

        # add the bias value and weight
        self.weight_list.append(random.uniform(-0.1, 0.1))
        self.value_list.append(1)

    def calculate(self):
        self.sum = 0
        for x in range(0, self.num_inputs):
            self.sum += self.weight_list[x] * self.value_list[x]
        self.output = sigmoid(self.sum)


def sigmoid(x):
    return 1 / (1 + math.e ** (-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# initialize the layers.
input_layer = [Neuron(1) for i in range(0, 3)]
hidden_layer1 = [Neuron(4) for x in range(0, 64)]
hidden_layer2 = [Neuron(65) for y in range(0, 32)]
output_layer = [Neuron(33) for z in range(0, 2)]

# initialize the error matrices
input_error = [0 for i in range(len(input_layer))]
hidden1_error = [0 for i in range(len(hidden_layer1))]
hidden2_error = [0 for i in range(len(hidden_layer2))]
output_error = [0 for i in range(len(output_layer))]


error_matrix = [input_error, hidden1_error, hidden2_error, output_error]
network = [input_layer, hidden_layer1, hidden_layer2, output_layer]


def back_propagation(learning_rate, case):

    for x in range(0, len(output_layer)):
        output_error[x] = sigmoid_prime(output_layer[x].sum) * (output_correction[case][x] - output_layer[x].output)

    for x in range(0, len(hidden_layer2)):
        temp = 0
        for y in range(0, len(output_layer)):
            temp += output_layer[y].weight_list[x] * output_error[y]
        hidden2_error[x] = sigmoid_prime(hidden_layer2[x].sum) * temp

    for x in range(0, len(hidden_layer1)):
        temp = 0
        for y in range(0, len(hidden_layer2)):
            temp += hidden_layer2[y].weight_list[x] * hidden2_error[y]
        hidden1_error[x] = sigmoid_prime(hidden_layer1[x].sum) * temp

    for x in range(0, len(output_layer)):  # Cambias los weights en output_layer
        for y in range(0, len(hidden_layer2) + 1):
            output_layer[x].weight_list[y] = output_layer[x].weight_list[y] + (
                learning_rate * output_layer[x].value_list[y] * output_error[x])

    for x in range(0, len(hidden_layer2)):  # Cambias weights en hidden_layer2
        for y in range(0, len(hidden_layer1) + 1):
            hidden_layer2[x].weight_list[y] = hidden_layer2[x].weight_list[y] + (
                learning_rate * hidden_layer2[x].value_list[y] * hidden2_error[x])

    for x in range(0, len(hidden_layer1)):  # Cambias weights en hidden_layer1
        for y in range(0, len(input_layer) + 1):
            hidden_layer1[x].weight_list[y] = hidden_layer1[x].weight_list[y] + (
                learning_rate * hidden_layer1[x].value_list[y] * hidden1_error[x])


def forward_propagation(case):

    # Inputs for layer 1
    for x in range(0, len(hidden_layer1)):
        for y in range(0, len(input_layer)):
            hidden_layer1[x].value_list[y] = color_input[case][y]

    # Calculate output for hidden layer 1
    for x in range(0, len(hidden_layer1)):
        hidden_layer1[x].calculate()

    # Inputs for layer 2
    for x in range(0, len(hidden_layer2)):
        for y in range(0, len(hidden_layer1)):
            hidden_layer2[x].value_list[y] = hidden_layer1[y].output

    for x in range(0, len(hidden_layer2)):
        hidden_layer2[x].calculate()

    for x in range(0, len(output_layer)):
        for y in range(0, len(hidden_layer2)):
            output_layer[x].value_list[y] = hidden_layer2[y].output

    for x in range(0, len(output_layer)):
        output_layer[x].calculate()


def print_layer(layer):
    for x in range(0, len(layer)):
        print("Node " + str(x) + "'s output was: " + str(layer[x].output))


def write_weights_to_file():  # run after a successful train() is called
    fw = open("file_weights.txt", "w+")

    for x in range(0, len(network)):
        for y in range(0, len(network[x])):
            for z in range (0, len(network[x][y].weight_list)):
                fw.write(str(network[x][y].weight_list[z]) + "\n")
    fw.close()


def read_weights_from_file():  # run before run() is called
    count = 0
    fr = open("file_weights.txt", "r")
    temp = fr.read().splitlines()

    for x in range(0, len(network)):
        for y in range(0, len(network[x])):
            for z in range(0, len(network[x][y].weight_list)):
                network[x][y].weight_list[z] = float(temp[count])
                count += 1

    # print(temp)

    fr.close()


def train():

    iterations = 0
    learning_rate = 0.05
    while iterations < 1000:
        case = 0
        while case < 15:
            forward_propagation(case)
            back_propagation(learning_rate, case)

            #print(str(output_layer[0].output) + " " + str(case))
            #print(str(output_layer[1].output) + " " + str(case))
            #print()

            write_weights_to_file()

            case += 1
        iterations += 1


def run():

    #read_weights_from_file()

    train()

    input = [76, 41, 243]

    for x in range(0, 15):
        color_input[x] = input

    forward_propagation(0)
    print_layer(output_layer)
    print()

run()