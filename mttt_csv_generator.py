import itertools
import csv

# Function to check for a winner
def check_winner(board):
    winning_positions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)             # Diagonals
    ]

    for a, b, c in winning_positions:
        if board[a] == board[b] == board[c] and board[a] != ' ':
            return board[a]

    if ' ' not in board:
        return 'T'  # Tie
    else:
        return 'U'  # Undecided

# Generate all possible Tic Tac Toe positions
def generate_positions():
    positions = []
    for board in itertools.product('XO ', repeat=9):
        positions.append((''.join(board), check_winner(board)))
    return positions

# Save positions to a CSV file
def save_positions_to_csv(positions, file_name='all_positions.csv'):
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['position', 'winner']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for position, winner in positions:
            writer.writerow({'position': position, 'winner': winner})

positions = generate_positions()
save_positions_to_csv(positions)