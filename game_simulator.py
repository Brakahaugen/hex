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
        batch_size: int = 128):
        """
            Implementation of the pseudo code given in the paper.
        """
        x = []
        y = []


        self.state_manager = state_manager

        #1. save interval for ANET
        anet.save_model(str(-1) + "3x3")

        for i in (range(I)):
            

            anet.epsilon = 1.0 - 1.0*(i/I)
 
            print("We are now at epoch: " + str(i))
            # print("\ncurrent epsilon: " + str(anet.epsilon))
            # print("\n" + state_manager.create_initial_state())
            # print("\ncurrent starting policy: ")
            # print([ '%.2f' % elem for elem in anet.forward("1102000201")])
            # x.append(i)
            # forward_result = anet.forward("1102000201").tolist()
            # y.append([forward_result.index(max(forward_result))])
            # print("11020")
            # print(anet.forward("11020"))
            # print("Expect index 2")

            # print("21020")
            # print(anet.forward("21020"))
            # print("Expect index 2 or 4")

            # print("20000")
            # print(anet.forward("20000"))
            # print("Expect index 2 or 4")

            # print("10000")
            # print(anet.forward("10000"))
            # print("Expect index 2 or 4")
            

            
            # fig = plt.figure()
            # plt.plot(x,y)
            # name = "plots/learning_rate" + str(i) + ".png"
            # plt.savefig(name)
            # plt.close(fig)


            # if len(anet):
            #     for net in range(len(anet)):
            #         anet[net].save_model("p"+str(anet[net].player)+ "_" + str(i))
            # else:

            #2. Clear Replay buffer
            # if len(anet):
            #     RBUF = [[],[]]
            # else:
            RBUF = [[],[]]
            #3. Randomly init the weights and biases of anet: WHY???
            '...'

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

                search_timer = 0
                while not state_manager.is_terminal_state(tree.root.state):
                    search_discount = (1 - (search_timer/state_manager.size**2))
                    #• Initialize Monte Carlo game board (Bmc) to same state as root.
                    b_mc = tree.root.state
                    
                    #Uses a dynamically decrease of number of search games given epsilon and after each search.
                    for g_s in (range(int(number_search_games*search_discount))):
                        #Use tree policy P to search from root to a leaf (L) of MCT. Update Bmc with each move.
                        leaf_node = tree.select()
                        
                        reward, child_node = leaf_node.rollout(anet, heuristic = True)
                        
                        child_node.backpropagate(reward)


                    #• D = distribution of visit counts in MCT along all arcs emanating from root.
                    D, actions = tree.root.get_distribution(heuristic = True)

                    #biasD
                    bias = 5
                    if bias > 1 and sum(D) > 0:
                        D = bias_example(D, bias)

                    # • Add case (root, D) to RBUF
                    # if len(anet):
                    #     RBUF[int(tree.root.state[0]) % 2].append([tree.root.state, D])
                    # else:
                    """
                        Only train on winning states, not on losing ones.
                    """
                    RBUF[int(tree.root.state[0])-1].append([tree.root.state, D])


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
                """
                    Train on winning states only
                """
                RBUF = RBUF[int(tree.root.state[0]) % 2]
                anet.backward([state for state, D in RBUF], [D for state, D in RBUF])
                RBUF = [[],[]]

            #1. save interval for ANET
            anet.save_model(str(i))
            bottom = 0
            if i > 20:
                bottom = i - 20
            topp = Topp(models = list(range(bottom,i + 1)), G = 1, size = self.state_manager.size, id = "4x4_test", V = False)
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
    game_size = 4
    g = GameSimulator()
    g.run_algo(anet = ANET(size = game_size), 
                state_manager = StateManager(game_size), 
                V = False,
                I = 100,
                number_search_games=10,
                num_actual_games= 100)
    # d = GameDisplay(3, "1000000000")
    # d.run_game()