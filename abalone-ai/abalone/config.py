GRID_RADIUS = 5

GROUP_LENGTHS = range(1, 4)
GAME_OVER = 8
BLACK = False
WHITE = True

def initialize(board):
    global GAME_OVER
    if board == 'standard':
        GAME_OVER = 8
    elif board == 'mini':
        GAME_OVER = 2
    else:
        GAME_OVER = 2
        return board
    return INITIAL_POSITIONS[board]

INITIAL_POSITIONS = {
    'mini': {
        WHITE: [
                         (1, -4), (2, -4), (3, -4),
                              (1, -3), (2, -3),
                                 (1, -2),
        ],
        BLACK: [
                                 (-1, 2),
                              (-2, 3), (-1, 3),
                         (-3, 4), (-2, 4), (-1, 4),
        ],
    },
    'standard': {
        WHITE: [
                (0, -4), (1, -4), (2, -4), (3, -4), (4, -4),
            (-1, -3), (0, -3), (1, -3), (2, -3), (3, -3), (4, -3),
                        (0, -2), (1, -2), (2, -2),
        ],
        BLACK: [
                        (-2, 2), (-1, 2), (0, 2),
            (-4, 3), (-3, 3), (-2, 3), (-1, 3), (0, 3), (1, 3),
                (-4, 4), (-3, 4), (-2, 4), (-1, 4), (0, 4),
        ],
    }
}
