from state_manager import StateManager
from anet import ANET
from visualization import GameDisplay
from matplotlib import pyplot as plt

class Topp:
    def __init__(self, models: list = [1,25,50,75,99], G: int = 25, size: int = 3, V = False):#, saving, mbs, optimizer, player, screen_size):
        # self.saving = saving
        # self.optimizer = optimizer
        # self.state_manager = state_manager
        self.M = len(models)
        self.G = G
        self.models = models
        # self.player = player
        # self.mbs = mbs
        self.V = V
        self.size = size
        self.init_players()
        # self.screen_size = screen_size
        
        
        
        
    def init_players(self):
        self.players = {}
        self.score = {}
        for i in range(self.M):
            self.players["player " + str(i)] = ANET(size = self.size)
            self.score["player " + str(i)] = 0
            self.players["player " + str(i)].load_model("models/checkpoint" + str(self.models[i]) + ".pth.tar")
        #print(self.players)
        
        
                
        
    def play(self):
        # init game_display
        game = GameDisplay(self.size, AI_players = [], V = self.V)
        for g in range(self.G):
            for i in range(self.M):
                for j in range(i+1, (self.M)):
                    print("\n---------------Changing players---------------")
                    print("i: " +str(i) + ", j:"+str(j))
                    print(self.M)

                    print("Player 1: Model_" + str(self.models[i]) + " VS Player 2: Model_" + str(self.models[j]))
                    
                    #time.sleep(1)
                    # if(player == 1):
                    #     caption = ("Green: M_" +  str(self.models[i]) + " VS Red: M_" +  str(self.models[j]))
                    # else:
                    #     caption = ("Red: M_" + str(self.models[i]) + " VS Green: M_" +  str(self.models[j]))
                    
                    two_games = [
                        [self.players["player " + str(i)], self.players["player " + str(j)]],
                        [self.players["player " + str(j)], self.players["player " + str(i)]]
                    ]
                    for entry in range(2):
                        game.AI_players = two_games[entry]
                        winner = game.run_game()
                        print(winner)
                        if int(winner) == (entry + 1):
                            self.score["player " + str(i)] += 1
                        else:
                            self.score["player " + str(j)] += 1

            
        print("\n-------------------SCORE-------------------")
        print(self.score)

        lists = sorted(self.score.items()) # sorted by key, return a list of tuples

        x, y = zip(*lists) # unpack a list of pairs into two tuples

        plt.plot(x, y)
        plt.show()
        fig = plt.figure()
        plt.plot(x,y)
        name = "plots/topp_results" + str(self.models[-1]) + str(sum(y)) + ".png"
        plt.savefig(name)
        plt.close(fig)
        
        
               

if __name__ == "__main__":
    topp = Topp(models = [0,1,2,3,4,5,6,7,8,9], G = 25, size = 2, V = False)
    topp.play()