# -*- coding: latin-1 -*-
import random
import sys
import math

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *


##
# AIPlayer
# Description: This AI player uses a Genetic algorithm decide the best placements for
# its constructs (Anthill, Tunnel, and food). It gradually learns the best placement
# by playing multiple games where it evals different states based on how well it played
# against its opponent.
# Assignment: Homework #4: Genetic Algorithms
#
# Due Date: February 27th, 2017
#
# @names: Sean Tollisen, Justin Hathaway
##
class AIPlayer(Player):
    # The current generation of genes
    gene_list = []
    # The fitness of the current generation
    gene_fit_list = []
    # The current gene being tested
    currGene = 0
    # The number of turns into the game we're in
    turnCount = 0
    # Since we're running each gene 3 times, hold 3 fitness values for the current gene
    this_gene_fit = [0, 0, 0]
    # how many times this gene's been run
    this_gene_runs = 0
    # state for best gene
    state_to_print = 0
    # current gene's state
    state = 0
    # best fitness for this generation
    bestVal = 0

    # __init__
    # Description: Initializes a list of genes
    ##
    def initialize(self):
        for i in range(0, 20):
            # Bucket of gene values
            geneValues = []
            for i in range(0, 11):
                # Stole the rando AI's initial gen ;)
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if (x, y) not in geneValues:
                        move = (x, y)
                geneValues.append(move)
            for i in range(0, 2):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if (x, y) not in geneValues:
                        move = (x, y)
                geneValues.append(move)
            # stick it to the end of the gene values
            self.gene_list.append(geneValues)
            # start it off with a fitness of 0
            self.gene_fit_list.append(0)

    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Gene Eric")
        # the coordinates of the agent's food and tunnel will be stored in these
        # variables (see getMove() below)
        self.myFood = None
        self.myTunnel = None
        turnCount = 0
        # If it's empty, initialize the stuff
        # I now realize it only runs once, but I left it there just in case
        if not self.gene_list:
            self.initialize()

            ##

    # getPlacement
    #
    # Description: The getPlacement method corresponds to the
    # action taken on setup phase 1 and setup phase 2 of the game.
    # In setup phase 1, the AI player will be passed a copy of the
    # state as currentState which contains the board, accessed via
    # currentState.board. The player will then return a list of 11 tuple
    # coordinates (from their side of the board) that represent Locations
    # to place the anthill and 9 grass pieces. In setup phase 2, the player
    # will again be passed the state and needs to return a list of 2 tuple
    # coordinates (on their opponent's side of the board) which represent
    # Locations to place the food sources. This is all that is necessary to
    # complete the setup phases.
    #
    # Parameters:
    #   currentState - The current state of the game at the time the Game is
    #       requesting a placement from the player.(GameState)
    #
    # Return: If setup phase 1: list of eleven 2-tuples of ints -> [(x1,y1), (x2,y2),…,(x10,y10)]
    #       If setup phase 2: list of two 2-tuples of ints -> [(x1,y1), (x2,y2)]
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        # implemented by students to return their next move
        geneValues = self.gene_list[self.currGene]
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                # just put the grass and stuff on the field already
                moves.append(geneValues[i])
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = geneValues[i][0]
                    y = geneValues[i][1]
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                    # if it's colliding with something on the other side, just choose somewhere random
                    else:
                        move = None
                        while move == None:
                            # Choose any x location
                            x = random.randint(0, 9)
                            # Choose any y location on enemy side of the board
                            y = random.randint(6, 9)
                            # Set the move if this space is empty
                            if currentState.board[x][y].constr == None and (x, y) not in moves:
                                move = (x, y)
                                # Just need to make the space non-empty. So I threw whatever I felt like in there.
                                currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    ##
    # getMove
    # Description: The getMove method corresponds to the play phase of the game
    # and requests from the player a Move object. All types are symbolic
    # constants which can be referred to in Constants.py. The move object has a
    # field for type (moveType) as well as field for relevant coordinate
    # information (coordList). If for instance the player wishes to move an ant,
    # they simply return a Move object where the type field is the MOVE_ANT constant
    # and the coordList contains a listing of valid locations starting with an Ant
    # and containing only unoccupied spaces thereafter. A build is similar to a move
    # except the type is set as BUILD, a buildType is given, and a single coordinate
    # is in the list representing the build location. For an end turn, no coordinates
    # are necessary, just set the type as END and return.
    #
    # Parameters:
    #   currentState - The current state of the game at the time the Game is
    #       requesting a move from the player.(GameState)
    #
    # Return: Move(moveType [int], coordList [list of 2-tuples of ints], buildType [int]
    ##
    def getMove(self, currentState):
        if self.turnCount == 0:
            self.state = currentState
        # self.state_to_print = currentState
        moves = listAllLegalMoves(currentState)
        selectedMove = moves[random.randint(0, len(moves) - 1)];

        # don't do a build move if there are already 3+ ants
        numAnts = len(currentState.inventories[currentState.whoseTurn].ants)
        while (selectedMove.moveType == BUILD and numAnts >= 3):
            selectedMove = moves[random.randint(0, len(moves) - 1)];

        if selectedMove.moveType == END:
            self.turnCount += 1
        return selectedMove

    ##
    # getAttack
    # Description: The getAttack method is called on the player whenever an ant completes
    # a move and has a valid attack. It is assumed that an attack will always be made
    # because there is no strategic advantage from withholding an attack. The AIPlayer
    # is passed a copy of the state which again contains the board and also a clone of
    # the attacking ant. The player is also passed a list of coordinate tuples which
    # represent valid locations for attack. Hint: a random AI can simply return one of
    # these coordinates for a valid attack.
    #
    # Parameters:
    #   currentState - The current state of the game at the time the Game is requesting
    #       a move from the player. (GameState)
    #   attackingAnt - A clone of the ant currently making the attack. (Ant)
    #   enemyLocation - A list of coordinate locations for valid attacks (i.e.
    #       enemies within range) ([list of 2-tuples of ints])
    #
    # Return: A coordinate that matches one of the entries of enemyLocations. ((int,int))
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy.
        return enemyLocations[0]

    # mate
    # Description: Take two parent genes and slice them together to make two babies
    # in a terrifying, gorey ritual of rebirth
    # SO MANY BABIES
    #
    # Parameters:
    #   parent1 - a gene to cut up for stuff
    #   parent2 - another gene to cut up for stuff
    ##
    def mate(self, parent1, parent2):
        # list to hold the BABIES
        newList = []
        # pick a spot to slice the parents
        cut = int(math.floor((random.random() * 11) + 1))
        # Cut the parents and stick their halves together to make children
        child1 = parent1[1:cut]
        child1.append(parent2[cut:13])
        child2 = parent2[1:cut]
        child2.append(parent1[cut:13])

        # If a child has more than one of the same tuple as a position, MUTATE IT
        for i in child1:
            for e in child1:
                if i == e and i != e:
                    if e < 11:
                        child1[e] = (random.randint(0, 9), random.randint(0, 3))
                    else:
                        child1[e] = (random.randint(0, 9), random.randint(6, 9))
        for i in child2:
            for e in child2:
                if i == e and i != e:
                    if e < 11:
                        child2[e] = (random.randint(0, 9), random.randint(0, 3))
                    else:
                        child2[e] = (random.randint(0, 9), random.randint(6, 9))
        # Stick 'em to the new list
        newList.append(child1)
        newList.append(child2)
        return newList

    # newGen
    # Description: Choose a bunch of parents and make BABIES
    # MAKE ME BABIES, RANDOM FLOATING GENETIC MATERIAL
    ##
    def newGen(self):
        # A list of indices indicating the order of fitness
        sortedList = []
        # The best fitness value
        maxVal = 0
        # The former's index
        maxInd = 0
        # a new list so that babies won't make babies
        newList = []
        # Actually sort the parents
        for i in range(0, 20):
            for e in range(0, 20):
                if self.gene_fit_list[e] > maxVal:
                    maxInd = e
                    maxVal = self.gene_fit_list[e]
            sortedList.append(maxInd)
            self.gene_fit_list[maxInd] = 0
        # Select the parents
        for i in range(0, 10):
            parent1 = 0
            parent2 = 0
            looper = True
            # starting at the top of the sorted list, every parent has a 1/4 to be chosen
            # if the parent before them isn't
            while looper:
                if random.random() < .25 or parent1 == 19:
                    looper = False
                else:
                    parent1 += 1
            looper = True
            while looper:
                if (random.random() < .25 or parent2 == 18 or parent2 == 19) and parent2 is not parent1:
                    looper = False
                else:
                    parent2 += 1
                    # Translate that to a parent off the sortedList
        parent1 = self.gene_list[sortedList[parent1]]
        parent2 = self.gene_list[sortedList[parent2]]
        newList.append(self.mate(parent1, parent2))
        gene_list = newList

    ##
    #
    # Description: The last method, registerWin, is called when the game ends and simply
    # indicates to the AI whether it has won or lost the game. This is to help with
    # learning algorithms to develop more successful strategies.
    #
    # Parameters:
    #   hasWon - True if the player has won the game, False if the player lost. (Boolean)
    #
    def registerWin(self, hasWon):
        # If it's won, it's score is 1500-the number of turns it took to win
        # otherwise, it's just the number of turns it survived
        if hasWon:
            self.this_gene_fit[self.this_gene_runs] = 1500 - self.turnCount
        else:
            self.this_gene_fit[self.this_gene_runs] = self.turnCount
        # After 3 games, average the current gene's 3 scores and then roll over to the next one
        if self.this_gene_runs < 2:
            self.this_gene_runs += 1
        else:
            self.this_gene_runs = 0
            totalFit = self.this_gene_fit[0] + self.this_gene_fit[1] + self.this_gene_fit[2]
            self.gene_fit_list[self.currGene] = int(totalFit / 4)
            # After the full generation is tested, make a new one and print out the best one of
            # this generation
            if self.currGene < 19:
                # Also, record this one if it's the best one of this generation
                if self.gene_fit_list[self.currGene] > self.bestVal:
                    self.bestVal = self.gene_fit_list[self.currGene]
                    self.state_to_print = self.state
                self.turnCount = 0
                self.currGene += 1
            else:
                asciiPrintState(self.state_to_print)
                self.currGene = 0
                self.newGen()
                self.bestVal = 0