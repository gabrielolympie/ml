import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import _pickle as pickle

dico_pieces = {
    '0' : '',
    'r' : 'Tour Noir',
    'n' : 'Cavalier Noir',
    'b' : 'Fou Noir',
    'q' : 'Dame Noir',
    'k' : 'Roi Noir',
    'p' : 'Pion Noir',
    'R' : 'Tour Blanc',
    'N' : 'Cavalier Blanc',
    'B' : 'Fou Blanc',
    'Q' : 'Dame Blanc',
    'K' : 'Roi Blanc',
    'P' : 'Pion Blanc',}

p_number = {'r' : 4,
            'n' : 4,
            'b' : 4,
            'q' : 9,
            'k' : 4,
            'p' : 8,
            'R' : 4,
            'N' : 4,
            'B' : 4,
            'Q' : 9,
            'K' : 4,
            'P' : 8}

## Dictionnaries for easier indexes replacements
pos_to_ind = {'0' : 0}
ind_to_pos = {0 : '0'}
count = 1
for i in range(8):
    for j in range(8):
        l = 8-i
        c = 'abcdefgh'[j]
        pos = str(c)+str(l)
        pos_to_ind[pos] = count
        ind_to_pos[count] = pos
        count+=1
        
pawns = list('0rnbqkpRNBQKP')
paw_to_ind = {pawns[i] : i for i in range(len(pawns))}
ind_to_paw = {i : pawns[i] for i in range(len(pawns))} 

## Storage Utils
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



## Function to switch between different types of representations that are more python friendly
def fen_to_matrix(fen):
    l = fen.split(' ')
    turn = l[1]
    board = l[0].split('/')
    mat = np.zeros((8,8)).astype(int).astype(str)
#    print(board)
    for i, line in enumerate(board):
        row = i
        col = 0
        for j in line:
            try:
                j = int(j)
                col += j
            except:
                mat[row, col] = j
                col += 1
    return mat

def matrix_to_fen(matrix, turn):
    fen = ""
    for row in matrix:
        count = 0
        for col in row:
            if col == '0':
                count += 1
            else:
                fen += str(count) + str(col)
                count = 0
                
        fen += str(count) + '/'
    fen = fen[:-1]
    fen = fen.replace('0', '')
    fen += ' '
    fen += turn
    fen += ' '
    fen += "KQkq"
    fen += ' '
    return fen

def matrix_to_sparse(matrix):
    sparse = []
    for i in range(8):
        for j in range(8):
            l = 8-i
            c = 'abcdefgh'[j]
            if matrix[i,j] != '0':
                sparse.append(matrix[i,j]+str(c)+str(l))
    return np.array(sparse)

def sparse_to_matrix(sparse):
    matrix = np.zeros((8,8)).astype(int).astype(str)
    for elt in sparse:
        matrix[8-int(elt[2]), 'abcdefgh'.rindex(elt[1])] = elt[0]
    return np.array(matrix)

## Function allowing to switch between several representation that are more deep learning friendly
def get_pawn_centric_rep(fen):
    pawn = ""
    pos = []
    pawn_dico = {elt : [] for elt in p_number}
    
    sparse = matrix_to_sparse(fen_to_matrix(fen))
    
    for elt in sparse:
        pawn_dico[elt[0]].append(elt[1:])
    
    for p in p_number:
        s = len(pawn_dico[p])
        for elt in pawn_dico[p]:
            pos.append(elt)
            pawn += p
            
        for i in range(p_number[p] - s):
            pos.append('0')
            pawn += p
            
    return np.array(list(pawn)), np.array(pos)

def pawn_centric_to_sparse(pawn, pos):
    sparse = []
    for i, elt in enumerate(pawn):
        if pos[i] != '0':
            sparse.append(elt+pos[i])
    return np.array(sparse)

def get_pos_centric_rep(fen):
    matrix = fen_to_matrix(fen)
    pos = []
    pawn = []
    for i in range(8):
        for j in range(8):
            l = 8-i
            c = 'abcdefgh'[j]
            pos.append(str(c)+str(l))
            pawn.append(matrix[i, j])
    return np.array(pawn), np.array(pos)

def pos_centric_to_sparse(pawn, pos):
    sparse = []
    for i, elt in enumerate(pawn):
        if elt != '0':
            sparse.append(elt + pos[i])
    return np.array(sparse)