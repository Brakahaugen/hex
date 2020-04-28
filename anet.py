import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

import time
import pathlib
import random

class ANET:
    def __init__(self, 
                layers: list = [256,256], 
                activations = [None, torch.nn.ReLU()], #torch.nn.Sigmoid() torch.nn.Linear() torch.nn.Tanh()
                optimizer = "Adagrad",
                size: int = 6, lr = 0.05, 
                epsilon = 1):

        self.epsilon = epsilon #Should be decreased for each backward.
        self.size = size

        self.activations = activations
        self.model = self.init_nn(layers, size)

        if optimizer == "Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        

        self.loss = nn.BCELoss()

        
        self.epochs = 1

        self.global_step = 0
        self.losses = {}
        self.best_index_losses = {}
        self.plot_every_x_step = 100

        self.amount_of_training_cases = 0
        self.print_model()

    def init_nn(self, layers: list, size: int):
        """
            Creates the model based on the layers. Always apply relu in hidden layers and softmax at end.
        """

        modules = []
        modules.append(torch.nn.Linear(size**2 + 1, layers[0]))
        if self.activations[0] != None:
            modules.append(self.activations[0])

        for i in range(len(layers)-1):
            modules.append(torch.nn.Linear(layers[i], layers[i+1]))
            if len(self.activations) > (i+1) and self.activations[i+1] != None:
                modules.append(self.activations[i + 1])


        modules.append(torch.nn.Linear(layers[-1], size**2))
        modules.append(torch.nn.Softmax(-1))
        

        model = torch.nn.Sequential(*modules)
        # for layer in model:
        #     if type(layer) == nn.Linear:
        #         layer.weight.data.fill_(1/size**2)
        #         print(layer.weight.data)
        return model

    def forward(self, x):
        """
            Forward x through the network
            return a tensor which should represent the probability distribution over each possible child state
        """

        if type(x) == str:
            x = self.pre_process_state(x[0:])
        elif type(x[0]) == str:
            for i in range(len(x)):
                l = list(x[i])
                x[i] = ''.join(l[0:])
            x = self.pre_process_state(x, batch = True)
        forwarded_x = self.model.forward(x)

        return forwarded_x

    def backward(self, x_batch, y_batch):
        """
            Does the backward function on a batch from the Replay Buffer.
            args:
                x_batch: list strings where each string is a state-representation:
                y_batch: list of lists of floats, where each list is a distribution over the given state.
        """

        #Preprocess by making the lists into tensors.
        for epoch in range(self.epochs):

            y_batch = torch.FloatTensor(y_batch)

            output = self.forward(x_batch.copy())
            
            self.optimizer.zero_grad()
            calculated_loss = self.loss(output, y_batch)
            calculated_loss.backward()
            self.optimizer.step()
            print(calculated_loss)
            

        # print(calculate_best_index_losses(output.argmax(1), y_batch.argmax(1)))
        self.losses[self.global_step] = calculated_loss
        self.best_index_losses[self.global_step] = calculate_best_index_losses(output.argmax(1), y_batch.argmax(1))
        self.global_step += 1
        self.amount_of_training_cases += len(x_batch)
        
        if self.global_step % self.plot_every_x_step == 0:
            x_after_train = self.forward(x_batch.copy())
            print("Saving figure")
            for i in range(len(x_batch)):
                print(output[i])
                print(y_batch[i])
                print(x_after_train[i])
                print()

            create_plot(self.losses, self.best_index_losses)


    def pre_process_state(self, state: list or str, batch: bool = False):
        """
            Changes the state or state batch to a tensor
        """

        if batch:
            tensor_state = torch.zeros([len(state),len(state[0])])
            for i in range(len(state)):
                for j in range(len(state[0])):
                    if int(state[i][j]) == 2:
                        tensor_state[i,j] = -1
                    else:
                        tensor_state[i,j] = int(state[i][j])
        else:
            tensor_state = torch.zeros([len(state)])
            for i in range(len(state)):
                if int(state[i]) == 2:
                    tensor_state[i] = -1
                else:
                    tensor_state[i] = int(state[i])
        return tensor_state


    # unrelated help functions

    def print_model(self):
        for layer in self.model:
            print(layer)        

    def save_model(self, filepostfix: str):
        torch.save(self.model.state_dict(), 'models/checkpoint' + filepostfix + '.pth.tar')
        

    def load_model(self, PATH: str = 'models/checkpoint.pth.tar'):
        """
            Loads the model weights from file into the model
        """
        if 'models/' not in PATH:
            PATH = 'models/' + PATH
        try:
            self.model.load_state_dict(torch.load(PATH))
            self.model.eval()
        except:
            print(PATH)
            raise AttributeError


    def print_state(self, state):
        state_array = list(state[1:])
        for row in range(self.size):
            print(" ".join(str(state_array[row*self.size + i])+ ' ' +' - '[i < self.size-1] for i in range(0,self.size)))
            if row < self.size - 1:
                print(" ".join("| " +  ' / '[i < self.size-1] for i in range(0,self.size)))


    def print_state_array(self, state_array):
        for i in range(self.size):
                print(state_array[i*self.size:(i+1)*self.size])
        print()


def calculate_best_index_losses(pred, target):
    hits = 0
    misses = 0
    for i in range(pred.shape[0]):
        if pred[i] == target[i]:
            hits += 1
        else:    
            misses += 1
    return hits/(hits + misses)

def create_plot(loss_dict: dict, accuracy_dict):
    sum_down_factor = 5
    x_dicts = []
    y_dicts = []

    dicts = [loss_dict, accuracy_dict]
    for loss_dict in dicts:
        x_list = []
        y_list = []
        lists = sorted(loss_dict.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        step = 0
        y_sum = 0
        for x_, y_ in zip(x,y):
            y_sum += y_
            if x_ % sum_down_factor == 0:
                x_list.append(step)
                y_list.append(y_sum/sum_down_factor)
                step += 1
                y_sum = 0
        x_dicts.append(x_list)
        y_dicts.append(y_list)
        
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(x_dicts[0], y_dicts[0], 'ko-')
    plt.title('A tale of 2 subplots')
    plt.ylabel('Damped oscillation')

    plt.subplot(1, 2, 2)
    plt.plot(x_dicts[1], y_dicts[1], 'r.-')
    plt.xlabel('time (s)')
    plt.ylabel('Undamped')


    # plt.plot(x_list, y_list)

    plt.ylim((0, 1))   # set the ylim to bottom, top
    # plt.show()
    plt.savefig('plots/plot_' + str(time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime())) + '.png')   # save the figure to file


if __name__ == "__main__":
    net = ANET(size = 6)
