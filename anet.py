import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

import time
import pathlib

class ANET:
    def __init__(self, player = 1, layers: list = [256,256], size: int = 6, lr = 0.02, epsilon = 1, epsilon_decay_rate = 0.99):

        self.epsilon = epsilon #Should be decreased for each backward.
        self.epsilon_decay_rate = epsilon_decay_rate
        self.size = size

        self.model = self.init_nn(layers, size)
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        self.loss = nn.BCELoss()

        self.global_step = 0
        self.losses = {}
        self.best_index_losses = {}
        self.amount_of_training_cases = 0
        self.plot_every_x_step = 100
        self.print_model()
        self.player = player
        self.epochs = 5

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
        # self.epsilon *= self.epsilon_decay_rate

        #Preprocess by making the lists into tensors.
        for epoch in range(self.epochs):
            y_batch = torch.FloatTensor(y_batch)
                    
            output = self.forward(x_batch.copy())
            
            # print("follws:")


            # print("output followed by target")
            
            # for i in range(len(x_batch)):
            self.optimizer.zero_grad()
            calculated_loss = self.loss(output, y_batch)
            calculated_loss.backward()
            self.optimizer.step()
            # print(calculated_loss)
            # for i in range(len(x_batch)):
            #     print(y_batch[i])
            #     print(output[i])
            #     print(self.forward(x_batch[i]))
        

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
            create_plot(self.losses, self.best_index_losses, str(self.player) + "_" + str(id(self)))
            
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
        all_moves_like_player_one = False

        if all_moves_like_player_one:
            if batch:
                transformed_state = []
                for s in state:
                    new_state = transform_like_one(s)
                    transformed_state.append(new_state)
            else: 
                transformed_state = transform_like_one(state) 

            state = transformed_state
            return torch.Tensor(state)

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


def transform_like_one(s: str):
    """
        takes a state string or a batch of state string, and 
        transforms them to be like the player 1 one would see this state.
    """
    new_state = []
    if (s[0] == "1"):
        for i in range(1, len(s)):
            if s[i] == "2":
                new_state.append(-1)
            else:
                new_state.append(int(s[i]))
    elif (s[0] == "2"):
        for i in range(1, len(s)):
            if s[i] == "2":
                new_state.append(1)
            elif s[i] == "1":
                new_state.append(-1)
            else:
                new_state.append(0)

        #Now rotate list 90degrees:
        rotated_state = []
        size = int(len(s)**(0.5))
        for i in reversed(range(size)):
            for j in (range(size)):
                rotated_state.append(new_state[i+j*size])
        new_state = rotated_state
        # print("old state")
        # print(list(s))
        # print("new state")
        # print(new_state)
    return new_state

def calculate_best_index_losses(pred, target):
    hits = 0
    misses = 0
    for i in range(pred.shape[0]):
        if pred[i] == target[i]:
            hits += 1
        else:    
            misses += 1
    return hits/(hits + misses)

def create_plot(loss_dict: dict, accuracy_dict, name: str):
    sum_down_factor = 10
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
    plt.savefig('plots/plot_' + str(time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime())) + '.png')   # save the figure to file


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
    


            