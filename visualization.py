import pygame
import time
import random
from state_manager import StateManager
from os import path
from anet import ANET
import threading


# Define Colors 
GREY = (100, 100, 100)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BACKGROUND = (215,215,215)
###############################

## assets folder
img_dir = path.join(path.dirname(__file__), 'assets')

font_name = pygame.font.match_font('arial')


class GameDisplay:
    def __init__(self, size, start_state = None, AI_players: list = None, V = True):
        """
            Takes in board size and optionally start_state
            If it also takes in a list of players it will run a game between the twos
        """
        self.state_manager = StateManager(size) #
        self.SIZE = size
        self.V = V
        self.init_modules()
        #Init start_state
        if start_state == None: 
            self.start_state = '2'.ljust(size**2 + 1, '0')
        self.AI_players = AI_players
        self.player_names = None
        self.player_scores = None


    def run_game(self):
    
        if self.V:
            self.init_modules()
            return game_loop(self.start_state, self.state_manager, self.SIZE, self.GRID, self.WIDTH, self.LINE_WIDTH, self.CIRCLES, self.HOLE_RADIUS, self.FPS, self.players, self.screen, self.clock, self.AI_players, self.V, self.player_names, self.player_scores)
        else:
            return game_loop(self.start_state, self.state_manager, self.SIZE, self.GRID, self.WIDTH, self.LINE_WIDTH, self.CIRCLES, self.HOLE_RADIUS, AI_players = self.AI_players, V = self.V)



    def init_modules(self):
        #Init constants
        self.WIDTH = 700
        self.HEIGHT = 700
        self.LINE_WIDTH = 4
        self.FPS = 1
        self.HOLE_RADIUS = int(self.HEIGHT/ (4*self.SIZE))
        self.GRID = {}
        self.CIRCLES = {}

        #Init modules
        if self.V:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Hex Grid")
            self.clock = pygame.time.Clock()     ## For syncing the FPS
            create_grid(self.SIZE, self.WIDTH, self.HEIGHT, self.GRID)

            #Create player avatars
            self.players = {}
            self.players[1] = pygame.transform.scale(pygame.image.load(path.join(img_dir, 'blue.png')).convert_alpha(), (2*self.HOLE_RADIUS, 2*self.HOLE_RADIUS))
            self.players[2] = pygame.transform.scale(pygame.image.load(path.join(img_dir, 'red.png')).convert_alpha(), (2*self.HOLE_RADIUS, 2*self.HOLE_RADIUS))
            



def create_grid(SIZE, WIDTH, HEIGHT, GRID):
    """
        Inits the circle grid
    """
    for i in range(SIZE):
        for j in range (SIZE):
            circle = (
                int((WIDTH * (SIZE + j - i)) / (SIZE*2 - 1) - WIDTH/(2*(2*SIZE - 1))), 
                int((HEIGHT * (j + i + 1)) / (SIZE*2 - 1) - HEIGHT/(2*(2*SIZE - 1))),
                )
            GRID[i,j] = circle

def draw_pins(state: str, SIZE, CIRCLES, GRID, HOLE_RADIUS, screen, players):
    """
        Draws the pins on the grid matching the state
        Args: str: State: holds info on state of board.
    """
    for i in range(SIZE):
        for j in range(SIZE):
            if state[i*SIZE + j + 1] == "1":
                CIRCLES[i,j] = screen.blit(players[1], (GRID[i,j][0] - HOLE_RADIUS, GRID[i,j][1] - HOLE_RADIUS))
                
            elif state[i*SIZE + j + 1] == "2":
                CIRCLES[i,j] = screen.blit(players[2], (GRID[i,j][0] - HOLE_RADIUS, GRID[i,j][1] - HOLE_RADIUS))
            
            else:
                CIRCLES[i,j] = (pygame.draw.circle(screen, GREY, GRID[i,j], HOLE_RADIUS))

            
def draw_grid(SIZE, GRID, LINE_WIDTH, screen):
    """
        Draws the grid lines
    """
    for i in range(SIZE):
        pygame.draw.line(screen, BLACK, GRID[i,0], GRID[0,i], LINE_WIDTH)
        pygame.draw.line(screen, BLACK, GRID[i,SIZE - 1], GRID[SIZE - 1,i], LINE_WIDTH)
        for j in range(1, SIZE):
            if i == 0 or i == SIZE - 1:
                pygame.draw.line(screen, BLUE, GRID[i,j-1], GRID[i,j], LINE_WIDTH*3)            
                pygame.draw.line(screen, RED, GRID[j-1,i], GRID[j,i], LINE_WIDTH*3)
            else:
                pygame.draw.line(screen, BLACK, GRID[i,j-1], GRID[i,j], LINE_WIDTH)            
                pygame.draw.line(screen, BLACK, GRID[j-1,i], GRID[j,i], LINE_WIDTH)

def draw_text(state, surf, text, SIZE, x, y):
    """
        Draws the given text on the given position.
    """
    COLOR = BLACK
    if state[0] == "1":
        COLOR = BLUE
    elif state[0] == "2":
        COLOR = RED

    font = pygame.font.Font(font_name, SIZE)
    text_surface = font.render(text, True, COLOR)       ## True denotes the font to be anti-aliased 
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)

def win_display(winner: str, screen, WIDTH, HEIGHT, SIZE, clock, FPS):
    screen.fill(BACKGROUND)   
    draw_text(screen, "P L A Y E R " + str(winner) + "   W I N S", 32, WIDTH/2, HEIGHT/2)
    draw_text(screen, "Press any key to restart game", 16, WIDTH/2, HEIGHT/1.5)
    pygame.display.update() 
    global STATE
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            if event.type == pygame.MOUSEBUTTONUP or event.type == pygame.KEYDOWN:
                STATE = "1".ljust(SIZE**2 + 1, '0')
                return game_loop() 
        clock.tick(FPS)

def return_winner(state: str):
    return int(state[0]) % 2 + 1

def render_board(state, SIZE, GRID, WIDTH, LINE_WIDTH, CIRCLES, HOLE_RADIUS, players, screen, player_names, player_scores = []):
    screen.fill(BACKGROUND)   
    draw_grid(SIZE, GRID, LINE_WIDTH, screen) 
    draw_pins(state, SIZE, CIRCLES, GRID, HOLE_RADIUS, screen, players)
    if StateManager(int((len(state) - 1)**(0.5))).is_terminal_state(state) and player_names:
        draw_text("0", screen, player_names[(int(state[0]) % 2)] + " WINS THE GAME", 36, WIDTH/2,WIDTH/2) 
    pygame.display.update()
    # pygame.time.delay(400)
    if player_names:
        draw_text(state, screen, "Player: " + player_names[int(state[0]) - 1], 20, WIDTH - 100,30) 
    if player_scores:
        for i in range(2): 
            draw_text("0", screen, player_names[i] + " score: " + str(player_scores[i]), 20, 100,30 + 30*i) 
    pygame.display.update() 
    return

def mouse_place_pin(state, CIRCLES, SIZE, state_manager):

    #Find the circle that is getting clicked on
    for key in CIRCLES:
        if CIRCLES[key].collidepoint(pygame.mouse.get_pos()):
            key_index = key[0] * SIZE + key[1]
            
            if key_index in state_manager.get_legal_moves(state):
                #Append move to state
                state_list = list(state)
                state_list[key_index + 1] = state_list[0]
                state_list[0] = str(int(state_list[0]) % 2 + 1)
                state = ''.join(state_list)

                # if state_manager.is_terminal_state(state) != 0:
                #     win_display(int(state[0]) % 2 + 1)
                print(state)
                return (state)
    return state



# def game_manager(state, state_manager, SIZE, GRID, WIDTH, LINE_WIDTH, CIRCLES, HOLE_RADIUS, FPS, players, screen, clock, AI_players):
#     game_loop(state, state_manager, SIZE, GRID, WIDTH, LINE_WIDTH, CIRCLES, HOLE_RADIUS, FPS, players, screen, clock, AI_players)
     
     
    #SINGLE GAME LOOP
def game_loop(state, state_manager, SIZE, GRID, WIDTH, LINE_WIDTH, CIRCLES, HOLE_RADIUS, FPS = None, players = None, screen = None, clock = None, AI_players = None, V = True, player_names = [], player_scores = []):
    if V:
        render_board(state, SIZE, GRID, WIDTH, LINE_WIDTH, CIRCLES, HOLE_RADIUS, players, screen, player_names, player_scores)
        # pygame.time.delay(1000)
    while True:
    
        if state_manager.is_terminal_state(state):
            # draw_text(state, screen, "Player: " + str(return_winner(state)) + " WON!", 30, WIDTH - 80,30)  
            # render_board(state, SIZE, GRID, WIDTH, LINE_WIDTH, CIRCLES, HOLE_RADIUS, players, screen)
            # state = "1".ljust(SIZE**2 + 1, "0")
            return return_winner(state)
        if AI_players != None:
            if (len(AI_players) == 1):
                #Player vs. AI Mode. Player will be first.
                if state[0] == "1":
                    #Get Distribution from anet and choose the best move
                    D = AI_players[0].forward(state).tolist()
                    print(state_manager.get_legal_moves(state))

                    print("Before")
                    print(D)
                    
                    #Find best legal move
                    while D.index(max(D)) not in state_manager.get_legal_moves(state):
                        D[D.index(max(D))] = -99999 
                    print("After")
                    print(D)
                    #Do action
                    state = state_manager.simulate_move(state, D.index(max(D)))
                    continue
            else:
                #Get Distribution from anet and choose the best move
                D = AI_players[int(state[0]) - 1].forward(state).tolist()
                print(D)
                while D.index(max(D)) not in state_manager.get_legal_moves(state):
                    D[D.index(max(D))] = -99999 
                state = state_manager.simulate_move(state, D.index(max(D)))

        if V:
            render_board(state, SIZE, GRID, WIDTH, LINE_WIDTH, CIRCLES, HOLE_RADIUS, players, screen, player_names, player_scores)
            # pygame.time.delay(1000)
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                if event.type == pygame.MOUSEBUTTONUP:
                    state = mouse_place_pin(state,CIRCLES, SIZE, state_manager)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        break
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        quit()


def func2():
    i = 1
    print(i)
    while i < 10000000:
        i += 1
        if i % 1000 == 0:
            print(i) 
            time.sleep(1)
def func1():
    game = GameDisplay(3)#, AI_players = AI_players)
    game.run_game()

if __name__ == "__main__":
    AI_players = [ANET(size = 6, player = "AI_1"), ANET(size = 6, player = "AI_2")] #, ANET(size = 3, player = 2)]
    # print(AI_players)
    AI_players[0].load_model('checkpoint29heur.pth.tar')
    # AI_players[0].load_model('checkpoint8.pth.tar')

    AI_players[1].load_model('checkpoint20heur.pth.tar')
    
    # # print(AI_players[0].forward('11200'))
    # # print(AI_players[0].forward('11002'))
    # # print(AI_players[0].forward('10200'))
    # print(AI_players[0].forward("2101020200"))
    # print(AI_players[0].forward("1202010100"))
    

    
    # AI_players[1].load_model('checkpoint40.pth.tar')
    
    game = GameDisplay(6, AI_players = AI_players)
    game.player_names = ["model29_heur", "20heur"]#, "model_6"]
    while True:
        game.run_game() 

    # t1 = threading.Thread(target=func1)
    # t2 = threading.Thread(target=func2)

    # t1.start()
    # t2.start()