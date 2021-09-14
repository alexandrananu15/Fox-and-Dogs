import time
import sys
from typing import Callable, List, Optional, Set, Tuple
from copy import deepcopy as copy

import pygame

GAME_H_OFFSET = 50
RADIUS = 30
PIECE_RADIUS = 30
DIMENSION = 8
DISPLAY = None
PLAYER_0_BACKGROUND_COLOR = (255, 200, 200)
PLAYER_1_BACKGROUND_COLOR = (200, 200, 255)
GAME_SCREEN_X, GAME_SCREEN_Y = 80 * 8, 80 * 8 + 50
BLACK = (0,0,0)
WHITE = (255, 255, 255)
HIGHLIGHT = (200, 200, 200)
DOGS_COLOR = (153, 204, 255)
FOX_COLOR = (255, 153, 255)
BACKGROUND = (230, 255, 215)

pygame.font.init()
FONT = pygame.font.Font('freesansbold.ttf', 32)
FONT_SMALLER = pygame.font.Font('freesansbold.ttf', 28)

# Statistics kept about the game, number of steps or computer.
stats_game_start_time = 0
stats_nodes_generated = []
stats_current_nodes_generated = 0
stats_time = []
stats_moves = [0, 0]

class State:

    def __init__(self, fox_pos: Tuple[int, int], dogs_pos: Set[Tuple[int, int]], turn: int) -> None:
        self.fox_pos = fox_pos
        self.dogs_pos = dogs_pos
        self.turn = turn

    def possible_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        '''
            0 for fox
            1 for dogs
        '''
        ans = []
        if self.turn == 0:
            # for fox
            for move in [(-1, -1),  (1, -1), (1, 1), (-1, 1)]:
                # the fox goes in all directons on diagonals
                new_x, new_y = self.fox_pos[0] + move[0], self.fox_pos[1] + move[1]

                # test if it's not on the dogs or not out of the matrix
                if min(new_x, new_y) < 0 or max(new_x, new_y) > 7 or (new_x, new_y) in self.dogs_pos:
                    continue
                ans.append((self.fox_pos, (new_x, new_y)))
        else:
            # for dogs
            for dog in self.dogs_pos:
                for move in [(1, 1), (-1, 1)]:
                    new_x, new_y = dog[0] + move[0], dog[1] + move[1]
                    if min(new_x, new_y) < 0 or max(new_x, new_y) > 7 or (new_x, new_y) in self.dogs_pos or (new_x, new_y) == self.fox_pos:
                        continue
                    ans.append((dog, (new_x, new_y)))

        return ans

    def estimation(self, idx) -> int:
        '''
            0 - fst estimation: number of tiles until the fox gets to the top
            1 - snd estimation: number of tiles until the fox gets to the top, 
                but according to the dog's position 
            fox is min player
        '''
        if idx == 0:
            return self.fox_pos[1]
        else:
            ans = self.fox_pos[1]
            for dog in self.dogs_pos:
                if dog[1] < self.fox_pos[1]:
                    ans += 1

            return ans      

    def make_new_state(self, coming_from: Tuple[int, int], to: Tuple[int, int]) -> Optional['State']:
        '''
            0 for fox
            1 for dogs
        '''
        # if nu e o mutare valida
        if abs(coming_from[0] - to[0]) != 1 or abs(coming_from[1] - to[1]) != 1:
            return None
        
        # if a iesit din matrice
        if min(to) < 0 or max(to) > 7:
            return None

        # if nu a luat-o in jos
        if self.turn == 1 and to[1] != coming_from[1] + 1:
            return None 

        # daca pozitia pe care mutam este ocupata deja
        if to == self.fox_pos or to in self.dogs_pos:
            return None

        if self.turn == 0:
            # verific ca coming_from e poz vulpii 
            if coming_from != self.fox_pos:
                return None

            new_state = copy(self)
            new_state.fox_pos = to
            new_state.turn = 1
            return new_state

        else:
            # ptr dogs
            if coming_from not in self.dogs_pos:
                return None

            new_state = copy(self)
            new_state.dogs_pos.remove(coming_from)
            new_state.dogs_pos.add(to)
            new_state.turn = 0
            return new_state

    def is_final(self) -> bool:
        '''
            check final state
        '''
        return self.possible_moves() == [] or self.fox_pos[1] == 0

    def get_successors(self) -> List['State']:
        '''
            generate successors
        '''
        ans = []

        for a, b in self.possible_moves():
            x = self.make_new_state(a, b)
            if x is None:
                raise RuntimeError("Nu se poate crea o noua stare!")
            
            ans.append(x)

        return ans

    def show(self):
        '''
            Displays the state in the console. 
        '''
        for idx in range(8):
            for jdx in range(8):
                if (jdx, idx) == self.fox_pos:
                    print(" X ", end='')
                elif (jdx, idx) in self.dogs_pos:
                    print(" 0 ", end='')
                else:
                    print(" . ", end='')
            print("\n")

        print("------------------------")



def min_max(state: State, depth: int, estimate: int) -> int:
    '''
        min_max algo
    '''
    global stats_current_nodes_generated
    stats_current_nodes_generated += 1

    if depth <= 0 or state.is_final():
        return state.estimation(estimate)

    best = 10**9 if state.turn == 0 else -10**9

    for successor in state.get_successors():
        est = min_max(successor, depth - 1, estimate)
    
        if state.turn == 0: 
            best = min(best, est)
        else:
            best = max(best, est)
    
    return best


def alpha_beta(state: State, depth: int, estimate: int, alpha: int, beta: int) -> int:
    '''
        alpha beta algo
    '''
    global stats_current_nodes_generated
    stats_current_nodes_generated += 1

    if depth <= 0 or state.is_final():
        return state.estimation(estimate)

    best = 10**9 if state.turn == 0 else -10**9

    for successor in state.get_successors():
        est = alpha_beta(successor, depth - 1, estimate, alpha, beta)
    
        if state.turn == 0: 
            # nod vulpe
            best = min(best, est)
            beta = min(beta, est)
        else:
            # nod dogs
            best = max(best, est)
            alpha = max(alpha, est)

        if alpha > beta: 
            return best

    return best

def print_title_on_display(content):
    '''
        Prints a given title on the display.
    '''
    text = FONT.render(content, True, (50, 50, 50))
    text_rect = text.get_rect()
    text_rect.center = (GAME_SCREEN_X // 2, GAME_H_OFFSET // 2 + 5)
    DISPLAY.blit(text, text_rect)

def ask_user_questions(question: str, answers: list):
    '''
        Asks the user some questions.
    '''

    DISPLAY.fill(BACKGROUND)
    print_title_on_display(question)
    DISTANCE = 10
    
    L, H = GAME_SCREEN_X, GAME_SCREEN_Y

    X = len(answers)
    Y = len(answers[0])
    H -= GAME_H_OFFSET

    for i in range(X):
        for j in range(Y):
            rect = pygame.Rect(
                int(L / X * i) + 3,
                int(H / Y * j) + GAME_H_OFFSET + 3,
                int(L / X - 6),
                int(H / Y - 6)
            )
            pygame.draw.rect(DISPLAY, (100, 100, 100), rect, width=4, border_radius=6) 

            center_x = int(L / X * i + (L / X / 2))
            center_y = GAME_H_OFFSET + int(H / Y * j + (H / Y / 2))
            text = FONT_SMALLER.render(answers[i][j], True, (20, 20, 20))
            text_rect = text.get_rect()
            text_rect.center = (center_x, center_y)
            DISPLAY.blit(text, text_rect)
            
    pygame.display.flip()

    while True:
        # Loop through the events of the game.
        for event in pygame.event.get():
            # Quit.
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Something was pressed.
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # Check if a move was made, and if yes acknowledge it.
                pos = (pos[0], pos[1] - GAME_H_OFFSET)

                choice = (int(pos[0] / (L / X)), int(pos[1] / (H / Y)))
                return choice

class Player:
    '''
        returns None if Player is Human
        not None for PC
    '''
    def __init__(self):
        self.estimation = None
        self.algo = None
        self.depth = None

    def move(self, state: State) -> State:
        '''
            make move
        '''
        global stats_current_nodes_generated
        stats_current_nodes_generated = 0

        best = 10**9 if state.turn == 0 else -10**9
        best_state = None

        for neighbour in state.get_successors():
            if self.algo == 0:
                est = min_max(neighbour, self.depth, self.estimation)
            else:
                est = alpha_beta(neighbour, self.depth, self.estimation, -10**9, 10**9)

            if state.turn == 0 and est < best:
                best = est
                best_state = neighbour
            elif state.turn == 1 and est > best:
                best = est
                best_state = neighbour
                
        if best_state is None:
            raise RuntimeError("Unable to find next state!")

        stats_nodes_generated.append(stats_current_nodes_generated)
        print(f"Generated {stats_current_nodes_generated} nodes for this step.")
        print(f"Estimation: {best}")


        return best_state

def init_stats():
    '''
        Resets the status.
    '''
    global stats_game_start_time, stats_nodes_generated
    global stats_current_nodes_generated, stats_time
    global stats_moves

    stats_game_start_time = time.time()
    stats_nodes_generated = []
    stats_current_nodes_generated = 0
    stats_time = []
    stats_moves = [0, 0]

def draw_table(state: State, highlighted_piece: Tuple, title: str):
    DISPLAY.fill(WHITE)

    print_title_on_display(title)

    piece_color = WHITE

    L = GAME_SCREEN_X // 8

    for i in range(8):
        for j in range(8):
            color = WHITE if (i + j) % 2 == 0 else BLACK
            color = HIGHLIGHT if (i, j) == highlighted_piece else color
        
            pygame.draw.rect(DISPLAY, color, pygame.Rect(
                                    i * L,
                                    j * L + GAME_H_OFFSET,
                                    L,
                                    L))

    def draw_piece(piece: Tuple, color: Tuple):
        pygame.draw.circle(
                    DISPLAY,
                    color,
                    (piece[0] * L + L // 2, piece[1] * L + L // 2 + GAME_H_OFFSET),
                    RADIUS
                )
    
    draw_piece(state.fox_pos, FOX_COLOR)
    for dog in state.dogs_pos:
        draw_piece(dog, DOGS_COLOR)
    
    pygame.display.flip()

def display_stats(state: State):
    '''
        Prints the statistics in the console.
    '''
    global stats_game_start_time, stats_nodes_generated
    global stats_current_nodes_generated, stats_time
    global stats_moves

    if len(stats_time) == 0:
        return


    names = ["Player 1", "Player 2"]
    status = "finished successfully" if state.is_final() else "was stopped"
    print(f"\n{names[1 - state.turn]} has won!")
    print(f"\nStats for game which {status}:")
    print(f"  * Game duration: {time.time() - stats_game_start_time}")
    print(f"  * Number of moves made by the computer: {len(stats_time)}")
    print(f"  * Total computer thinking time: {sum(stats_time)}")
    print(f"  * Average computer thinking time: {sum(stats_time) / len(stats_time)}")
    print(f"  * Maximal computer thinking time: {max(stats_time)}")
    print(f"  * Minimal computer thinking time: {min(stats_time)}")
    stats_time = sorted(stats_time)
    print(f"  * Median computer thinking time: {stats_time[len(stats_time) // 2]}")
    print(f"  * Total number of visited nodes: {sum(stats_nodes_generated)}")
    print(f"  * Average number of visited nodes: {sum(stats_nodes_generated) / len(stats_nodes_generated)}")
    print(f"  * Maximal number of visited nodes: {max(stats_nodes_generated)}")
    print(f"  * Minimal number of visited nodes: {min(stats_nodes_generated)}")
    stats_nodes_generated = sorted(stats_nodes_generated)
    print(f"  * Median number of visited nodes: {stats_nodes_generated[len(stats_nodes_generated) // 2]}")
    print(f"  * Number of moves:")
    print(f"     * First player: {stats_moves[0]}")
    print(f"     * Second player: {stats_moves[1]}")

def game_loop(players: List[Player], names: List[str]):
    '''
        Game loop. This will run forever until the user closes it.
    '''
    
    global stats_moves

    init_stats()

    # Cream un nou player

    state = State((0, 7), {(1, 0), (3, 0), (5, 0), (7, 0)}, 0)
    highlighted_piece: Tuple = (-1, -1)
    last_move_time = time.time()

    step = True

    # Game loop.
    while True:
        
        # Something moved. Print statistics.
        if step:
            turn = names[state.turn]
        
            if state.is_final():
                winner = names[1 - state.turn]
                draw_table(state, (-1, -1), "Winner is " + winner)
                display_stats(state)

                while True:
                    for event in pygame.event.get():
                        # Quit.
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

            else:
                draw_table(state, highlighted_piece, turn + " should move :)")

            step = False

        # If the player is a bot, then make it play.
        if players[state.turn] != None:
            time_before = time.time()
            new_state = players[state.turn].move(state)
            if new_state is None: 
                raise RuntimeError("Unable to move - PC turn")
            time_after = time.time()
            stats_time.append(time_after - time_before)
            print(f"{names[state.turn]} (PC) moved in {time_after - time_before} sec.")
            state = new_state
            stats_moves[state.turn] += 1
            state.show()
            last_move_time = time.time()
            step = True
            continue

        for event in pygame.event.get():
            # Quit.
            if event.type == pygame.QUIT:
                display_stats(state)
                pygame.quit()
                sys.exit()
            
            # Something was pressed.
            if event.type == pygame.MOUSEBUTTONDOWN:
                if players[state.turn] != None:
                    continue
                pos = pygame.mouse.get_pos()

                # Check if a move was made, and if yes acknowledge it.
                pos = (pos[0], pos[1] - GAME_H_OFFSET)

                if pos[1] < 0:
                    continue

                a, b = pos[0] // (GAME_SCREEN_X // 8), pos[1] // (GAME_SCREEN_X // 8)
                if highlighted_piece == (-1, -1):
                    if state.turn == 0:
                        if (a, b) != state.fox_pos:
                            continue

                    else:
                        if (a, b) not in state.dogs_pos:
                            continue

                    highlighted_piece = (a, b)
                    last_move_time = time.time()
                    step = True

                else:
                    new_state = state.make_new_state(highlighted_piece, (a, b))
                    highlighted_piece = (-1, -1)
                    print(f"{names[state.turn]} attempted a move. Time for the move: {time.time() - last_move_time}")
                    if new_state is not None:
                        state = new_state
                        stats_moves[state.turn] += 1
                        state.show()
                    step = True
        

def main():
    '''
        Entry point of the application.
    '''
    global GAME_SCREEN_Y, GAME_SCREEN_X, DISPLAY
    pygame.display.set_caption("Alexandra Nanu - Vulpea si cainii")

    # Get information.
    DISPLAY = pygame.display.set_mode(size=(GAME_SCREEN_X, GAME_SCREEN_Y))

    players: List[Optional[Player]] = [None, None]


    for name,  id in [("First Player:", 0), ("Second Player:", 1)]:
        idx, idy = ask_user_questions(name, [["Human"],["PC"]] )

        if idx == 1:
            players[id] = Player()
            est, _ = ask_user_questions(
                "Estimation to use:",
                [["Distance to Top"], ["Distance & Dogs"]]
            )

            players[id].estimation = est   

            alg, _ = ask_user_questions(
                "Algorithm to use:",
                [["Min-Max"], ["Alpha-Beta"]]
            )

            players[id].algo = alg

            answers = [["Easy", "Medium"],
                    ["Hard", "Expert"]]

            delta_h = 3 * players[id].algo
            depths = [[2 + delta_h, 3 + delta_h], [4 + delta_h, 5 + delta_h]]
            
            idx, idy = ask_user_questions("Difficulty: ", answers)

            players[id].depth = depths[idx][idy]

    game_loop(players, ["Player 1", "Player 2"])


if __name__ == '__main__':
    main()
