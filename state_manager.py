import time
import copy
import termcolor
import numpy as np

class StateManager:

    def __init__(self, size: int = 5):
        self.size = size
        self.hexNeighbours = [[-1,0],[-1,1],[0,1],[0,-1],[1,-1],[1,0]]
        self.random_player = 1

    # def get_best_action(distro: list = []):
    #     best_action = index(max(distro))

    def create_initial_state(self, p: int = 0, padding = 0):
        """
            Creates a flattened board-string representing an empty board.
            If no player is specified, alternates between player 1 and 2
            Padding applies n number of padding layers to the model, Making the effective board smaller in size.
        """
        if p == 0:
            self.random_player = self.random_player % 2 + 1
            p = self.random_player

        if padding > 0:
            board = ["0"] * self.size**3
            for i in range(padding):
                for j in range(i, self.size):
                    board[i*self.size + j] = "1" #top
                    board[(self.size - i - 1)*self.size + j] = "1" #bottom

                    board[j*self.size + i] = "2" #left
                    board[(j + 1)*self.size - i - 1] = "2" #right
                    
            return str(p) + ''.join(board)

        else:
            return str(p).ljust(self.size**2 + 1, '0')
        
    def create_padded_states(self, state: str):
        """
            Given a winning, padded, board state, find all combinations that would lead to a win 
        """
        winner = self.is_terminal_state(state)
        state_2d = self.state_to_2D(state)

        # if string    
        #     for i in range(self.size):
        #         state_2d[0,i] 
        #         state_2d[self.size,i]
        #         state_2d[i,0]
        #         state_2d[i,self.size]
                

    def flip_state(self, state):
        """
            returns a mirrored repr. of the board
        """
        return state[::-1]
    
    def rotate_state(self, state, k = 1):
        # print(type(state))
        if isinstance(state, str):
            print("haø")
            print("heyo")
            state = self.state_to_2D(state)
        print(state)
        
        return self.flatten_2d(np.rot90(state))

    
    def state_to_2D(self, state: str):
        """
            Takes a string state and returns a 2D numpy matrix representing that state
        """
        state_list = list(state[1:])
        two_dim = np.reshape(state_list, (self.size, self.size))
        print(two_dim)
        return two_dim
    
    def flatten_2d(self, state_2d: np.array):
        """
            takes a 2D numpy matrix and returns a flattened string
        """
        state = state_2d.flatten()
        return ''.join(state.tolist())


    def simulate_move(self, state: str, action: int):
        """
            applies the action on the state by placing the players pin on the specified location.
            args:
            state = board_state in flattened form: "10200020"
            action = location to place pin
        """
        p = state[0]
        state_array = list(state[1:])
        state_array[action] = p
        
        #Change player and return string:
        return (str(int(p)%2 + 1) + ''.join(state_array))

    def get_legal_moves(self, state: str):
        """
            returns a list of legal moves as indexes in the state
        """

        state_array = list(state[1:])
        legal_moves = []
        
        for i in range(len(state_array)):
            if state_array[i] == '0':
                legal_moves.append(i)

        return legal_moves

    def is_terminal_state(self, state: str):
        """
            Scans the given board to see if there is any link from one side to another.
            
            returns 1 if p1 wins, 2 if p2wins, otherwise: 0
        """
        state_array = list(state[1:])

        for i in range(self.size):
            win = self.check_wins([i,0], '1', state_array.copy())
            if win != '0':
                return int(win)

            win = self.check_wins([0,i], '2', state_array.copy())
            if win != '0':
                return int(win)       
        return int(win)

    #HELPER FUNCTION FOR IS_TERMINAL STATE
    def check_wins(self, cell: list, p: str, state_array: list):
        """
            Checks if there is a connected list from the starting cells to the ending cells
        """
        if state_array[cell[0] + cell[1]*self.size] == p:
            state_array[cell[0] + cell[1]*self.size] = '0'
        else:
            return '0'

        for n in self.hexNeighbours:
            if (0 <= cell[0] + n[0] < self.size) and (0 <= cell[1] + n[1] < self.size):
                n_cell = [cell[0] + n[0], cell[1] + n[1]]                
            else:
                continue
            if state_array[n_cell[0] + n_cell[1]*self.size] == p:
                if (p == '1') and (n_cell[1] == self.size - 1):
                    return '1'
                elif (p == '2') and (n_cell[0] == self.size - 1):
                    return '2'
                check_next = self.check_wins(n_cell, p, state_array)
                if check_next != '0':
                    return check_next
        return '0'

        #HELPER FUNCTIONS FOR PRINTING AND DEBUGGING
   
    #HELPER FUNCTIONS FOR PRINTING IN DEBUGGING
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
            



if __name__ == "__main__":
    s = StateManager(3)
    s.create_initial_state(padding=0)
    #Make a visualization given a size and a state.
    # s.print_state(s.create_initial_state(padding=2))
    print()
    # s.print_state(s.create_initial_state(padding=2))
    print()
    str = "1110022220"
    s.print_state(s.rotate_state(str,1))
    print()
    s.print_state(s.rotate_state(str,3))
    
    # print(s.is_terminal_state("12111222121221211"))
    # print(s.check_wins([0,0],'2',list("2111222121221211")))

    
    