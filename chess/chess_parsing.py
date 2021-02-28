import _pickle as pickle
import os
from tqdm.auto import tqdm
from multiprocess import Pool
from chessboard import display
from chess_env import *
from tqdm.auto import tqdm
import argparse
import gc

def save(file,name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'wb')
    else:
        outfile = open(name+'.pickle', 'wb')
    pickle.dump(file, outfile, protocol=4)
    outfile.close
    
def load(name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'rb')
    else:
        outfile = open(name+'.pickle', 'rb')
    file = pickle.load(outfile)
    outfile.close
    return file

def get_history(x):
    game = x
    env = Chess_Env(verbose = 0, display = 0)
    env.from_game(game)
    moves = [elt.uci() for elt in game.mainline_moves()]
    result = None
    for move in moves:
        result = env.play(move)
    
    history = env.history
    if result is None:
        history.append(['1/2', '1/2'])
    return history[:-1], history[-1]

def parse_game(game):
    try:
        turns, result = get_history(game)
        id1 = games[0].headers['FICSGamesDBGameNo']
        belo, welo = games[0].headers['BlackElo'], games[0].headers['WhiteElo']
        ids = []
        results = []
        fens = []
        col = []
        move = []
        rank = []
        blelo = []
        wielo = []
        for i, t in enumerate(turns):
            fens.append(t[0])
            ids.append(id1)
            results.append(result)
            rank.append(i)
            col.append(t[1])
            move.append(t[2])
            blelo.append(belo)
            wielo.append(welo)
        df = pd.DataFrame({'id': ids, 'rank' : rank, 'fen' : fens, 'color' : col, 'move' : move, 'results' : results, 'white' : welo, 'black' : belo})
        return df
    except:
        return pd.DataFrame({'id': [], 'rank' : [], 'fen' : [], 'color' : [], 'move' : [], 'results' : [], 'white' : [], 'black' : []})
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', help='year to process')
    args = parser.parse_args()

    parameters = int(args.year)
    print(parameters)
    
    file_list = []
    for file in os.listdir('./high/batch'):
        if str(parameters)+"_" in file:
            file_list.append(file.split('.')[0])
    
    print(file_list)
    for file in tqdm(file_list):
        games = load('./high/batch/' + file)
        out = list(map(parse_game, games))
        out = pd.concat(out)
        print(out.shape)
        print(out['id'].nunique())
        out['year'] = parameters
        save(out, './high/parsed/'+file)
        
        del out
        del games
        gc.collect()