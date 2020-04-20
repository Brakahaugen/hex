import time
import copy
import termcolor

class StateManager:

    def __init__(self, size: int = 5):
        self.size = size
        self.hexNeighbours = [[-1,0],[-1,1],[0,1],[0,-1],[1,-1],[1,0]]
        self.random_player = 1

    def create_initial_state(self, p: int = 0):
        """
            Creates a flattened board-string representing an empty board.
            If no player is specified, alternates between player 1 and 2
        """
        if p == 0:
            self.random_player = self.random_player % 2 + 1
            p = self.random_player
        return str(p).ljust(self.size**2 + 1, '0')
        
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
    s = StateManager(4)
    #Make a visualization given a size and a state.
    s.print_state("12111222121221211")
    print(s.is_terminal_state("12111222121221211"))
    # print(s.check_wins([0,0],'2',list("2111222121221211")))
    