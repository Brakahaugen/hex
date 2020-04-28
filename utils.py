import numpy as np
import random

def bias_example(distro: list, factor = 2):

    sum = 0
    best_index = max_index(distro)

    for i in range(len(distro)):
        distro[i] = 0
    distro[best_index] = 1

    return distro


def rotate_state(state: str, D: list, size: int, n: int):
    """
        NOT FINISHED
        Rotate the state and the list so that the representation will be equal to one of the other player.
    """
    if not (n % 2):
        print("invalid input")
        return
    print("before")
    print(state)
    print(D)
    player = state[0]
    #reshape to 2d list
    flipped_state = np.reshape(list(state[1:]), (size,size))
    flipped_distro = np.reshape(D, (size,size))
    print("2d:")
    print(flipped_state)
    print(flipped_distro)
    #flip both n times
    for i in range(n):
        flipped_state = np.rot90(flipped_state)
        flipped_distro = np.rot90(flipped_distro)
    print("flipped:")
    print(flipped_state)
    print(flipped_distro)
    # flatten states
    state = flipped_state.flatten().tolist()
    D = flipped_distro.flatten().tolist()
    print("flattened")
    print(state)
    print(D)
    state.insert(0, player)
    for i in range(len(state)):
        if state[i] is not "0":
            state[i] = str((int(state[i]) % 2) + 1)
    return ''.join(state), D



def max_index(D: list):
    """
    returns max index of list, random choice if several.
    """
    
    max = -999
    max_list = []
    for i in range(len(D)):
        if D[i] > max:
            max = D[i]
            max_list = []
        if D[i] == max:
            max_list.append(i)
    return (random.choice(max_list))