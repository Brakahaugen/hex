import math
from BasicClientActorAbs import BasicClientActorAbs
from state_manager import StateManager
import random
from anet import ANET

class BasicClientActor(BasicClientActorAbs):

    def __init__(self, IP_address=None, verbose=True):
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """


        # YOUR CODE HERE
        self.p = state[0]
        print(state)

        state = ''.join(str(s) for s in state)

        #Forward the state
        # print(state)
        print("contestants move: ")
        self.state_manager.print_state(state)
        D = self.anet.forward(state)
        
        # #Invalidate every illegal action
        for i in range(len(D)): 
            if i not in self.state_manager.get_legal_moves(state):
                D[i] = -99999 
        action = D.tolist().index(max(D))
        print("my move ("+str(self.p)+"): ")
        self.state_manager.print_state(self.state_manager.simulate_move(state, action))

        #Index to tuple:
        row = int(action/self.state_manager.size)
        column = action % self.state_manager.size

        # next_move is a tupe of row column where we want to place our stone.
        return (row,column)


    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        #############################

        # Initiating model and game_manager here:
        self.state_manager = StateManager(size = game_params[0])
        self.anet = ANET(size = game_params[0])
        #LOAD YOUR BEST 6x6 MODEL
        self.anet.load_model("models/checkpoint29heur.pth.tar")
        self.wins = 0
        self.total_games = 0

        print("num games to be played:" + str(num_games))

        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        self.wins += int(str(winner) == str(self.p))
        self.total_games += 1

        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))
        print("winrate: " + str(self.wins/self.total_games))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=True)
    bsa.connect_to_server()
