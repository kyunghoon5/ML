'''

@project       : Queens College CSCI 381/780 Machine Learning
@Instructor    : Dr. Alex Pang

@Date          : Spring 2022

A Object-Oriented Implementation of the TicTacToe Game 

references

'''

import enum


class GamePiece(enum.Enum):
    CROSS = "X"
    CIRCLE = "O"


class GameBoard(object):
    '''
    TODO: explain what the class is about, definition of various terms
    etc
    '''

    def __init__(self):
        self.nsize = 3
        self._board = []
        for r in range(self.nsize):
            self._board.append(['' for c in range(self.nsize)])

        # set the board to its initial state
        self.reset()

    def display(self):
        '''
        display the game board which will look like something like this

           1 | X | 3
           4 | 5 | O
           7 | 8 | 9

        '''

        # TODO
        for row in self._board:
            print(str(row[0]) + " | " + str(row[1]) + " | " + str(row[2]))
        # end TODO

    def reset(self):
        '''
        Reset the game board so that each cell is index from 1 to 9.
        So when display it will look like

           1 | 2 | 3
           4 | 5 | 6
           7 | 8 | 9

        '''

        # TODO
        for row in range(self.nsize):
            for col in range(self.nsize):
                self._board[row][col] = 3 * row + col + 1
        # end TODO

    def place_into(self, symbol, spot):
        '''
        Find the cell that spot is located, then replace the cell by 
        the symbol X or O
        '''
        # TODO
        row = int((spot - 1) / 3)
        col = (spot - 1) % 3
        self._board[row][col] = symbol.value
        # end TODO

    def has_winner(self):
        '''
        Determine if one side has won (ie a winning row, column or a winning diagonal.
        If there is a winner, display who is the winner and return true
        otherwise return false
        '''
        # TODO
        #check for horizontal wins
        for row in self._board:
            if (str(row[0]) == str(row[1]) == str(row[2]) and str(row[0]) != "0"):
                return True
        #check for verical wins
        for col in range(self.nsize):
            col_list = [i[col] for i in self._board]
            if (col_list.count("0") == 3):
                return True
            elif (col_list.count("X") == 3):
                return True
        #check for diagonals
        if (self._board[0][0] == self._board[1][1] == self._board[2][2] and self._board[0][0] != "0"):
            return True
        elif (self._board[0][2] == self._board[1][1] == self._board[2][0] and self._board[0][0] != "0"):
            return True


        return False
        # end TODO

    def is_valid(self, spot):
        '''
        return true if spot is a valid location that you can place a symbol into
        ie. it has not been occupied by an X or an O
        '''


        # TODO
        row = int((spot-1)/3)
        col = (spot-1) % 3
        if self._board[row][col] != "X" and self._board[row][col] != "O":
            return True
        else:
            return False
    # end TODO




def run():
    count = 0
    turn = GamePiece.CROSS

    start = input("Do you want to play Tic-Tac-Toe? (y/n)")
    if start.lower() == "y":
        board = GameBoard()
        board.display()

        while count < 9:

            print(f"It is {turn.value} turn. Which spot you want to pick?")
            spot = input()
            spot = int(spot)

            if board.is_valid(spot):
                board.place_into(turn, spot)
                board.display()

                # check if there is a winner, if yes, announce who is the winner
                # and close the game, otherwise set the turn to the other player

                # TODO
                if (board.has_winner()):
                    print(f" {turn.value}  Winner" + "\n End of Game")
                    return True
                # end TODO

                count = count + 1
                if turn == GamePiece.CROSS:
                    turn = GamePiece.CIRCLE
                elif turn == GamePiece.CIRCLE:
                    turn = GamePiece.CROSS

            else:
                print("Invalid spot. Please try again")

        # TODO announce it is a tie game
        print("It is tie game")
        print("End Game")
        # end TODO


if __name__ == "__main__":
    run()
