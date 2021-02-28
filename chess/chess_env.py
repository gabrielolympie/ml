import chess
from copy import deepcopy
from chess_utils import *
import chess.svg
from chessboard import display

#display.start('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1') 

class Chess_Env(chess.Board):
    def __init__(self, white = 'human', black = "random", from_list = None, verbose = 1, display = 0):
        """
        In case of from list, provide both the current turn (white or black) and a list of all pieces position
        white and black can either be human input, or an ai class, or random
        
        """
        super(Chess_Env, self).__init__()
        self.history = []
        
        self.white = white
        self.black = black
        self.verbose = verbose
        self.display = display
        
        if from_list is not None:
            self.from_sparse(from_list[0], from_list[1])
            
    def play(self, move):
        self.history.append((deepcopy(self.fen()), self.get_turn(), move))
        self.push_uci(move)
        self.update_viz()
        if self.is_game_over():
            results = self.result().split('-')
            self.history.append(results)
            if self.verbose == 1:
                print("The result of the match is : w-"+str(results[0])+" | b-"+str(results[1]))
            return results  ## w - b result
        else:
            return None
            
    def get_moves(self):
        all_moves = [move.uci() for move in self.legal_moves]
        return all_moves
    
    def get_matrix(self):
        return fen_to_matrix(self.fen())
    
    def get_sparse(self):
        return matrix_to_sparse(self.get_matrix())
    
    def update_viz(self):
        if self.display == 1:
            try:
                display.update(self.fen())
            except:
                print("Video system is not initialized, make sure it is")
    
    def from_matrix(self, matrix, turn):
        self.set_fen(matrix_to_fen(matrix, turn))
        self.update_viz()
        print("Successfully loaded from state")
    
    def from_sparse(self, sparse, turn):
        self.from_matrix(sparse_to_matrix(sparse), turn)
        
    def get_turn(self):
        return self.fen().split(' ')[1]
    
    def from_game(self, game):
        self.set_fen(game.board().fen())
        if self.verbose == 1:
            print(self.fen())
        self.update_viz()
    
    def hard_reset(self):
        self.reset()
        self.history = []
    
    def get_pawn_centric_env(self):
        return get_pawn_centric_rep(self.fen())
    
    def get_pos_centric_env(self):
        return get_pos_centric_rep(self.fen())
    
    
    def get_next_move(self):
        turn = self.get_turn()
        if turn == 'w':
            player_type = self.white
        else:
            player_type = self.black
        dicolor = {'w' : 'white', 'b' : 'black'}
        
        if player_type == "human":
            move = '0'
            cond = True
            possible_moves = self.get_moves()
            while cond:
                print(dicolor[turn] + ' are playing, choose your move carefull')
                print("possible moves are : " + " ".join(list(possible_moves)))
                move = input()
                if move in possible_moves:
                    cond = False
                else:
                    print("*************   Impossible Move, try again  ****************")
                
        elif player_type == "random":
            print("Random move with the color : "+str(dicolor[turn]))
            move = np.random.choice(self.get_moves())
        else:
            print(self)
            print("IA is playing with the color " + str(dicolor[turn]))
            state = self.get_state()  ## todo
            move = player_type.predict() ## todo
        print("Selected move is : "+ str(move))
        return move
            
    
    def get_player(self, t):
        return 1
    
    def close(self):
        display.terminate()