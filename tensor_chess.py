import numpy as np

def create_chess_matrices(pgn):
    """
    Convert PGN moves into 8x8x12 board matrices.

    Returns: A list of 8x8x12 matrices representing board states after each move.
    """
    def initialize_board():

        board = np.zeros((12, 8, 8), dtype=int)

        # Black pieces
        board[0, 0, [0, 7]] = 1  # R
        board[1, 0, [1, 6]] = 1  # N
        board[2, 0, [2, 5]] = 1  # B
        board[3, 0, 3] = 1       # Q
        board[4, 0, 4] = 1       # K
        board[5, 1, :] = 1       # p

        # White pieces
        board[6, 7, [0, 7]] = 1
        board[7, 7, [1, 6]] = 1
        board[8, 7, [2, 5]] = 1
        board[9, 7, 3] = 1
        board[10, 7, 4] = 1
        board[11, 6, :] = 1

        return board

    file_to_col = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    def move_to_indices(move):
        """Convert a chess move into board indices."""
        row = 8 - int(move[1])
        col = file_to_col[move[0]]
        return row, col

    board = initialize_board()
    board_states = [board.copy()]

    for move in pgn:
        # This assumes simple moves without castling, promotions, etc.
        if len(move) >= 2 and move[-1].isdigit():
            dest_row, dest_col = move_to_indices(move[-2:])

            # Eat
            if 'x' in move:
                for layer in range(6 if pgn.index(move)%2 == 0 else 6, 12):
                    if board[layer, dest_col, dest_row] == 1:
                        board[layer, dest_col, dest_row] = 0
                        break

            # Pawn move
            if len(move)==2:
                board[5 if pgn.index(move)%2 == 1 else 11, dest_col, dest_row] = 1
                # Delete the initiale emplacement ######
                if pgn.index(move)%2 == 1:
                    board[5, dest_col-1, dest_row] = 0
                else:
                    board[11, dest_col+1, dest_row] = 0


            pieces_layers = {'R': [0,6], 'N': [1,7], 'B': [2,8], 'Q': [3,9], 'K': [4,10]}

            # R,N,B,Q,K move
            if move[0] in pieces_layers:
                x=pieces_layers[move[0]]
                board[x[0] if pgn.index(move)%2 == 1 else x[1], dest_col, dest_row] = 1

                 # Delete the initiale emplacement
                if len(move)==3 or (len(move)==4 and 'x' in move):
                    # Rook # VOIR SI L4ECRITURE CHANGE SI Y A UN OBSTACLE
                    if move[0]=='R':
                        x=pieces_layers[move[0]]
                        board[x[0] if pgn.index(move)%2 == 1 else x[1],dest_col, :] = 0
                        board[x[0] if pgn.index(move)%2 == 1 else x[1],:, dest_row] = 0
                    # Knight
                    if move[0]=='N':
                        l=pieces_layers[move[0]]
                        for x in [-2,-1,1,2]:
                            for y in [-2,-1,1,2]:
                                if x!=y and  0 <= dest_col+x <= 7 and 0 <= dest_col+y <= 7:
                                    board[l[0] if pgn.index(move)%2 == 1 else l[1], dest_col+x, dest_row+y] = 0
                    # Bishop
                    if move[0]=='B':
                        l=pieces_layers[move[0]]
                        for x in [-1,1]:
                            for y in [-1,1]:
                                dest_col = max(0, min(7, dest_col + x))
                                dest_row = max(0, min(7, dest_row + y))
                                board[l[0] if pgn.index(move)%2 == 1 else l[1],dest_col, dest_row] = 0
                    # Queen
                    if move[0]=='Q':
                        x=pieces_layers[move[0]]
                        board[x[0] if pgn.index(move)%2 == 1 else x[1],:, :] = 0
                        board[x[0] if pgn.index(move)%2 == 1 else x[1],dest_col, dest_row] = 1
                    # King
                    if move[0]=='K':
                        x=pieces_layers[move[0]]
                        board[x[0] if pgn.index(move)%2 == 1 else x[1],:, :] = 0
                        board[x[0] if pgn.index(move)%2 == 1 else x[1],dest_col, dest_row] = 1

                # Delete the initiale emplacement but for 2 two same pieces and color can reach the same emplacement
                if (len(move)==4 and 'x' not in move) or (len(move)==5 and 'x' in move) :
                    x=pieces_layers[move[0]]
                    if move[1].isalpha():
                        col = file_to_col[move[1]]
                        board[x[0] if pgn.index(move)%2 == 1 else x[1],col, :] = 0
                    else:
                        raw = 8 - int(move[1])
                        board[x[0] if pgn.index(move)%2 == 1 else x[1],:, raw] = 0

        # Append the new board state
        board_states.append(board.copy())

    return board_states


pgn_test=['e4', 'c6', 'Nf3', 'd5', 'Nc3', 'dxe4', 'Nxe4', 'Nf6', 'Qe2', 'Nxe4', 'Qxe4', 'Qd5', 'Qh4', 'Qe6+', 'Be2', 'Qg4', 'Qg3', 'Qxg3', 'hxg3', 'Bf5', 'c3', 'e6', 'd4', 'h6', 'Ne5', 'Nd7', 'g4', 'Bc2', 'Kd2', 'Bh7', 'Nxd7', 'Kxd7', 'Bd3', 'Bxd3', 'Kxd3', 'Bd6', 'Be3', 'f5', 'gxf5', 'exf5', 'Rh5', 'f4', 'Bd2', 'g6', 'Rh3', 'b5', 'Rah1', 'h5', 'Rh4', 'Raf8', 'Ke4', 'Rf7', 'f3', 'Re8+', 'Kd3', 'a5', 'Re1', 'Rxe1', 'Bxe1', 'Ke6', 'Bd2', 'Kf5', 'Rh1', 'Re7', 'b3', 'Re6', 'c4', 'bxc4+', 'bxc4', 'a4', 'c5', 'Be7', 'Rb1', 'g5', 'Rb4', 'g4', 'Rxa4', 'Bg5', 'Ra8', 'gxf3', 'gxf3', 'h4', 'Rf8+', 'Rf6', 'Re8', 'Re6', 'Rf8+', 'Rf6', 'Rh8', 'Bh6', 'Ke2', 'h3', 'Kf1', 'Rg6', 'Rf8+', 'Bxf8']

chess_matrices = create_chess_matrices(pgn_test)

number_to_pieces={0:'0',1:'Rb',2:'Nb',3:'Bb',4:'Qb',5:'Kb',6:'pb',7:'Rw',8:'Nw',9:'Bw',10:'Qw',11:'Kw',12:'pw',
                  16:'poop',21:'poop',22:'poop',15:'poop',18:'poop'}
L=[]
L2=[]

for move in chess_matrices:
    i=1
    board = np.zeros((8, 8))
    for layers in move:
        board+=layers*i
        i+=1
    L.append(board)

for board in L:
    result = np.empty(board.shape, dtype=object)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            result[i, j] = number_to_pieces[board[i, j]]
    L2.append(result)


for i ,x in enumerate(L2) :
    print(pgn_test[i])
    print(x)
    print(' ')
