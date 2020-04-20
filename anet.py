import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

import pathlib

class ANET:
    def __init__(self, layers: list = [32, 64, 128, 256, 128, 64, 32], size: int = 6, lr = 0.2, epsilon = 1, epsilon_decay_rate = 0.99):

        self.epsilon = epsilon #Should be decreased for each backward.
        self.epsilon_decay_rate = epsilon_decay_rate
        self.size = size

        self.model = self.init_nn(layers, size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.loss = nn.BCELoss()

        self.global_step = 0
        self.losses = {}
        self.best_index_losses = {}
        self.amount_of_training_cases = 0
        self.plot_every_x_step = 100
        self.print_model()

    def init_nn(self, layers: list, size: int):
        modules = []
        modules.append(torch.nn.Linear(size**2 + 1, layers[0]))
        # modules.append(torch.nn.ReLU())

        for i in range(len(layers)-1):
            modules.append(torch.nn.Linear(layers[i], layers[i+1]))
            modules.append(torch.nn.ReLU())


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
            x = self.pre_process_state(x)
            return self.model.forward(x)#.tolist()
        elif type(x[0]) == str:
            x = self.pre_process_state(x, batch = True)
            return self.model.forward(x)

    def backward(self, x_batch, y_batch):
        """
            Does the backward function on a batch from the Replay Buffer.
            args:
                x_batch: list strings where each string is a state-representation:
                y_batch: list of lists of floats, where each list is a distribution over the given state.
        """
        # self.epsilon *= self.epsilon_decay_rate

        #Preprocess by making the lists into tensors.
        
        y_batch = torch.FloatTensor(y_batch)
                
        output = self.forward(x_batch)
        # print("follws:")
        # print(x_batch)
        # print(self.pre_process_state(x_batch[0]))
        # print(self.pre_process_state(x_batch, batch = True))
        # print(output)

        # print("output followed by target")
        
        # for i in range(len(x_batch)):
        self.optimizer.zero_grad()
        calculated_loss = self.loss(output, y_batch)
        calculated_loss.backward()
        self.optimizer.step()


        self.losses[self.global_step] = calculated_loss
        self.best_index_losses[self.global_step] = calculate_best_index_losses(output.argmax(1), y_batch.argmax(1))
        self.global_step += 1
        self.amount_of_training_cases += len(x_batch)
        if self.global_step % self.plot_every_x_step == 0:
            print("Saving figure")
            for i in range(len(x_batch)):
                print(output[i])
                print(y_batch[i])
                print()
            create_plot(self.losses, str(self.global_step) + "_loss")
            create_plot(self.best_index_losses, str(self.global_step) + "_index_accuracy")
            
            print(self.amount_of_training_cases)
            print(calculated_loss)


        # print("\nimprovement for a single case")
        # print("output")
        # print(output[0])
        # print("target")
        # print(y_batch[0])
        # print("after backward")
        # print(self.forward(x_batch[0]))
        
        


    def pre_process_state(self, state: list or str, batch: bool = False):
        if batch:
            tensor_state = torch.zeros([len(state),len(state[0])])
            for i in range(len(state)):
                for j in range(len(state[0])):
                    if int(state[i][j]) == 2:
                        tensor_state[i,j] = 2
                    else:
                        tensor_state[i,j] = int(state[i][j])
        else:
            tensor_state = torch.zeros([len(state)])
            for i in range(len(state)):
                if int(state[i]) == 2:
                    tensor_state[i] = 2
                else:
                    tensor_state[i] = int(state[i])
        return tensor_state
    
    def print_model(self):
        for layer in self.model:
            print(layer)
            


    # unrelated help functions
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

def create_plot(loss_dict: dict, name: str):
    lists = sorted(loss_dict.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples

    plt.clf()
    plt.plot(x, y)
    if y[-1] < 1:
        plt.ylim((0, 1))   # set the ylim to bottom, top
    plt.savefig('plots/plot_' + str(name) + '.png')   # save the figure to file


if __name__ == "__main__":
    net = ANET(size = 3)
    net.load_model('models/checkpoint95.pth.tar')
    print(net.model)
    # for layer in net.model:
    #     print(layer.weight.shape)
    #     print(layer.weight.data)
        

    state = "1122100000"
    # for i in range(10):
    import pdb; pdb.set_trace()
    distro = net.forward(state)
    print(distro)
    print(distro.index(max(distro)))
    print(max(distro))
    # print(len(distro))
    net.print_state(state)

        # state_array = list(state)
        # state_array[distro.index(max(distro))] = state[0]
        # # state_array[0] = state[0]
        
        # state = ''.join(state_array)

    # net.save_model()

# def test_code():
    #Load ned
    #test different cases where you know what is the best answer...
    


            