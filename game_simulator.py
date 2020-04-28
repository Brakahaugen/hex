from anet import ANET
from mc_tree import Node, MC_Tree
from state_manager import StateManager
from tqdm import tqdm
import time
import random
from matplotlib import pyplot as plt
from topp import Topp
from utils import bias_example

from multiprocessing import Process, Queue
from visualization2 import GameDisplay

class GameSimulator:

    def __init__(self):
        ""
    
    def run_algo(self,
        anet: ANET, 
        state_manager: StateManager, 
        I: int = 24, 
        num_actual_games: int = 200, 
        number_search_games: int = 200,
        V = False,
        batch_size: int = 128,
        heuristic_rollout = False,
        bias = True,
        reverse_cases = True):
        """
            Implementation of the pseudo code given in the paper.
        """

        self.state_manager = state_manager

        global_RBUF = []
        #1. save interval for ANET
        anet.save_model(str(-1))

        #pretrain
        # with open("RBUF.txt") as file_in:
        #     for line in file_in:
        #         if line[0] == "[":
        #             try:
        #                 line = eval(line)
        #                 anet.backward([state for state, D in line], [D for state, D in line])
        #             except:
        #                 pass
        # anet.save_model(str("super_boy"))
        # return

        for i in range(I):
            

            anet.epsilon = 0.2 - 0.2*(i/I)**0.5
            print("epsilon = " + str(anet.epsilon))
            print("We are now at epoch: " + str(i))
      

            #2. Clear Replay buffer
            RBUF = [[],[]]

            #3. Randomly init the weights and biases of anet
            # already_done

            #4
            for g_a in tqdm(range(num_actual_games)):

                #(a) Initialize the actual game board (Ba) to an empty board.
                b_a = state_manager.create_initial_state()

                #(b) sinit ← starting board state
                s_init = b_a
                
                # (c) Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit
                tree = MC_Tree(Node(s_init, state_manager, None))

                #Print in verbose mode
                if V: self.create_output(tree.root, True)

                search_timer = 0 #adding search timer for dynamish search games
                while not state_manager.is_terminal_state(tree.root.state):
                    search_discount = (1 - (search_timer/state_manager.size**2)) #Based on search timer

                    #• Initialize Monte Carlo game board (Bmc) to same state as root.
                    b_mc = tree.root.state
                    
                    #Uses a dynamically decrease of number of search games given epsilon and after each search.
                    for g_s in (range(int(number_search_games*search_discount))):
                        #Use tree policy P to search from root to a leaf (L) of MCT. Update Bmc with each move.
                        leaf_node = tree.select()
                        
                        reward, child_node = leaf_node.rollout(anet, heuristic = heuristic_rollout)
                        
                        child_node.backpropagate(reward)


                    #• D = distribution of visit counts in MCT along all arcs emanating from root.
                    D, actions = tree.root.get_distribution(heuristic = heuristic_rollout)

                    if bias and sum(D) > 0: #Biasing the distribution Only hard or not.
                        D = bias_example(D, bias)

                    # • Add case (root, D) to RBUF seperate lists for p1 and p2
                    RBUF[int(tree.root.state[0])-1].append([tree.root.state, D])

                    #Append the reverse as well as this is a similiar state...
                    if reverse_cases:
                        RBUF[int(tree.root.state[0])-1].append([tree.root.state[0] + tree.root.state[1:][::-1], list(reversed(D))])

                    #Choose actual move (a*) based on D
                    if (sum(D) <= 0):
                        a = random.choice(tree.root.getLegalActions())
                        s_next = state_manager.simulate_move(b_mc, a)
                        root = Node(s_next, state_manager, a)
                    else:
                        s_next = state_manager.simulate_move(b_mc, actions[D.index(max(D))])
                        root = Node(s_next, state_manager, D.index(max(D)))                        
                    tree.root = root

                    #Print if verbose mode
                    if V: self.create_output(tree.root, False)

                    #Increase the discount since we now will have a shorter tree.
                    search_timer += 1

                #(e) Train ANET on a random minibatch of cases from RBUF

                #Take out only the cases that led to a win. if p1 won; extract RBUFs for player1.
                RBUF = RBUF[int(tree.root.state[0]) % 2]

                try: #Save cases for later.
                    with open("RBUF.txt", "a") as file:
                        file.write(str(f'g={number_search_games},size={state_manager.size},num_cases={num_actual_games}' + '\n'))
                        file.write(str(RBUF) + "\n")
                except:
                    pass

                #Backward pass
                anet.backward([state for state, D in RBUF], [D for state, D in RBUF])
                RBUF = [[],[]]
                

            #1. save interval for ANET
            anet.save_model(str(i) + str("heur5x5_sg=500"))
        
        topp = Topp(models = list(range(i)), G = 1, size = self.state_manager.size, id = "final_tests", V = False)
        topp.play()



    def create_output(self, node: Node, init_mode: bool = False):

        if node.state[0] == '1':
            player = "P1"
        else:
            player = "P2"
        
        if init_mode:
            print("____________S T A R T I N G_________G A M E____________")
            self.state_manager.print_state(node.state)
            print(player + "'s turn:")
            return
        else:
            self.state_manager.print_state(node.state)
            
            if self.state_manager.is_terminal_state(node.state):
                print('P' + str(int(node.state[0])%2+1) + " WON")
            else:
                print(player + "'s turn:")


    

if __name__ == "__main__":
    # topp = Topp(models = list(range(0, 50)), G = 1, size = 6, id = "heur_6x6_super_tesings", V = False)
    # topp.play()
    
    game_size = 5
    g = GameSimulator()
    g.run_algo(anet = ANET(size = game_size), 
                state_manager = StateManager(game_size), 
                V = False,
                I = 50,
                number_search_games=100,
                num_actual_games= 100,
                heuristic_rollout = True,
                bias = True,
                reverse_cases = True)
                