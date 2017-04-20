# -*- coding: latin-1 -*-
import random
import sys

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *


##
# Authors: Justin Hathaway, Morgan Webber
#
# Assignment: Homework #6: TD Learning
#
# Due Date: April 24th, 2017
#
# Description:
##

##
# AIPlayer
# Description: The responsbility of this class is to interact with the game by
# deciding a valid move based on a given game state. This class has methods that
# will be implemented by students in Dr. Nuxoll's AI course.
#
# Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):
    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "HW6")
        # Variables to store our tunnel and anthill
        self.myTunnel = None
        self.myAnthill = None
        self.STATE_FILE = "states_hathaway19_webber18.csv"
        self.DF = 0.8
        self.alpha = 0.99
        self.greedy = 0.01

        # State memory
        self.stateMem = []

        # Load games from our file
        self.loadStates()


    ###
    #   consolidateState
    #
    #   Description:
    #       Strips a gamestate object of important information and returns a new, consolidated state object
    ###
    def consolidateState(self, state):
        return 0


    ###
    #   loadStates
    #
    #   Description:
    #       loads any existing states into our memory
    ###
    def loadStates(self):
        return 0


    ###
    #   saveStates
    #
    #   Description:
    #       saves any states in our memory to our file
    ###
    def saveStates(self):
        return 0


    ###
    #   rewardAgent
    #
    #   Description:
    #       Given a state, sets the reward for the agent of that state
    #       +1 for winning | -1 for losing | -0.001 for everything else
    ###
    def rewardAgent(self, state):
       return 0

    def TD_0(self, state, alpha, gamma, numOfEpisodes):
        return 0
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
        # Setup phase for placing anthill, grass, and tunnel
        # (hardcoded in for optimal chance of winning)
        if currentState.phase == SETUP_PHASE_1:
            # Indexes 0-1: Anthill, tunnel
            # Indexes 2-10: Grass
            return [(0, 0), (8, 2),
                    (0, 2), (1, 2), (2, 1), (7, 3), \
                    (0, 3), (1, 3), (8, 3), \
                    (2, 2), (9, 3)];
        # Setup phase for placing the opponent's food
        # (tries to place it far away from tunnel and anthill)
        elif currentState.phase == SETUP_PHASE_2:

            # Finds out which player ID is your opponent
            enemy = (currentState.whoseTurn + 1) % 2
            # Variables to hold coordinates of enemy constructs
            enemyTunnelCoords = getConstrList(currentState, enemy, (TUNNEL,))[0].coords
            enemyAnthillCoords = getConstrList(currentState, enemy, (ANTHILL,))[0].coords

            numToPlace = 2
            foodLocations = []
            # Goes through each piece of food to find an optimal place to put it
            for i in range(0, numToPlace):
                LargestDistanceIndex = [(-1, -1)]  # Placeholder coordinate value
                LargestDistance = -100  # Placeholder value (can't be a negative distance)
                foodLocation = None  # To hold location of current index
                while foodLocation == None:
                    # Searches coordinates in opponent's territory
                    for i in range(BOARD_LENGTH):
                        for j in range(6, 10):
                            # Only searches locations that haven't already been added or have placements on them
                            if currentState.board[i][j].constr == None \
                                    and (i, j) not in foodLocations:
                                # Adds distance from tunnel and anthill
                                currentDistance = stepsToReach(currentState, (i, j), enemyTunnelCoords) \
                                                  + stepsToReach(currentState, (i, j),
                                                                 enemyAnthillCoords)
                                # Keeps largest distance
                                if currentDistance > LargestDistance:
                                    # Replaces values for current largest distance
                                    LargestDistance = currentDistance
                                    LargestDistanceIndex = (i, j)
                    foodLocation = LargestDistanceIndex
                foodLocations.append(foodLocation)
            # Gives coordinates to place food on
            return foodLocations
        # Shouldn't reach this point
        else:
            return None

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
        # Variables to hold our inventory, our player ID, and the opponent's player ID
        myInv = getCurrPlayerInventory(currentState)
        me = currentState.whoseTurn
        enemy = (currentState.whoseTurn + 1) % 2
        # Variable to hold total number of ally worker ants
        numOfWorkerAnts = len(getAntList(currentState, me, (WORKER,)))
        # List of all ally worker ants
        myWorkers = getAntList(currentState, me, (WORKER,))
        # Variables to hold list of ally drones
        myDrones = getAntList(currentState, me, (DRONE,))
        # Variables to hold coordinates of enemy queen and anthill coordinates
        enemyQueenCoords = getAntList(currentState, enemy, (QUEEN,))[0].coords
        enemyAnthillCoords = getConstrList(currentState, enemy, (ANTHILL,))[0].coords
        # Variable to hold food
        foods = getConstrList(currentState, None, (FOOD,))

        # the first time this method is called, the food and tunnel locations
        # need to be recorded in their respective instance variables
        if self.myTunnel == None:
            self.myTunnel = getConstrList(currentState, me, (TUNNEL,))[0]
        if self.myAnthill == None:
            self.myAnthill = getConstrList(currentState, me, (ANTHILL,))[0]

        # If all the workers have moved, we're done (checks to see if last worker has moved)
        lastWorker = getAntList(currentState, me, (WORKER,))[numOfWorkerAnts - 1]
        if lastWorker.hasMoved:
            return Move(END, None, None)

        # Generates a random number between 0 and 1 to move Queen back and forth
        # Helps insure that other ants don't collide as often
        rndNum = random.randint(0, 1)
        if rndNum == 0:
            queenPath = createPathToward(currentState, myInv.getQueen().coords,
                                         (0, 1), UNIT_STATS[QUEEN][MOVEMENT])
        else:
            queenPath = createPathToward(currentState, myInv.getQueen().coords,
                                         (1, 1), UNIT_STATS[QUEEN][MOVEMENT])
        if not myInv.getQueen().hasMoved:
            return Move(MOVE_ANT, queenPath, None)

        # Creates enough workers to have 2 on the board (if we have food and anthill empty)
        if myInv.foodCount > 0 and numOfWorkerAnts < 2:
            if getAntAt(currentState, myInv.getAnthill().coords) is None:
                return Move(BUILD, [myInv.getAnthill().coords], WORKER)
        # Creates drones if we already have enough workers
        elif myInv.foodCount >= 2:
            if getAntAt(currentState, myInv.getAnthill().coords) is None:
                return Move(BUILD, [myInv.getAnthill().coords], DRONE)
        # Commands all drones to move to enemy queen coordinates
        for drone in myDrones:
            # Only moves the selected drone if it hasn't moved or if it's not on the enemy hill
            if drone.coords != enemyAnthillCoords:
                if not drone.hasMoved:
                    # Sends drone to opponent's anthill or queen, depending on which is closer
                    distToEnemyAnthill = stepsToReach(currentState, drone.coords, enemyAnthillCoords)
                    distToEnemyQueen = stepsToReach(currentState, drone.coords, enemyQueenCoords)
                    if distToEnemyAnthill < distToEnemyQueen:
                        dronePath = createPathToward(currentState, drone.coords,
                                                     enemyAnthillCoords, UNIT_STATS[DRONE][MOVEMENT])
                    else:
                        dronePath = createPathToward(currentState, drone.coords,
                                                     enemyQueenCoords, UNIT_STATS[DRONE][MOVEMENT])
                    return Move(MOVE_ANT, dronePath, None)
        # Moves all worker ants
        for worker in myWorkers:
            if not worker.hasMoved:
                # Move to anthill or tunnel if worker is carrying food
                if worker.carrying:
                    if (stepsToReach(currentState, worker.coords, self.myAnthill.coords)
                            < (stepsToReach(currentState, worker.coords, self.myTunnel.coords))):
                        # Checks the path to make sure there aren't any collisions
                        path = calcAntMove(currentState, worker, self.myAnthill.coords,
                                           UNIT_STATS[WORKER][MOVEMENT])
                    else:
                        # Checks the path to make sure there aren't any collisions
                        path = calcAntMove(currentState, worker, self.myTunnel.coords,
                                           UNIT_STATS[WORKER][MOVEMENT])
                    return Move(MOVE_ANT, path, None)
                # Move to closest food if worker isn't carrying food
                else:
                    closestFood = getConstrList(currentState, None, (FOOD,))[0]
                    for food in foods:
                        distToClosestFood = stepsToReach(currentState, worker.coords,
                                                         closestFood.coords)
                        distToCurrentFood = stepsToReach(currentState, worker.coords,
                                                         food.coords)

                        if distToCurrentFood < distToClosestFood:
                            closestFood = food

                    # Checks the path to make sure there aren't any collisions
                    path = calcAntMove(currentState, worker, closestFood.coords,
                                       UNIT_STATS[WORKER][MOVEMENT])
                    return Move(MOVE_ANT, path, None)

    ##
    # getAttack
    # Description: The getAttack method is called on the player whenever an ant completes
    # a move and has a valid attack. It is assumed that an attack will always be made
    # because there is no strategic advantage from withholding an attack. The AIPlayer
    # is passed a copy of the state which again contains the board and also a clone of
    # the attacking ant. The player is also passed a list of coordinate tuples which
    # represent valid locations for attack.
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
        # attacks a random enemy using enemy coordinates for valid attacks
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # registerWin
    # Description: The last method, registerWin, is called when the game ends and simply
    # indicates to the AI whether it has won or lost the game. This is to help with
    # learning algorithms to develop more successful strategies.
    #
    # Parameters:
    #   hasWon - True if the player has won the game, False if the player lost. (Boolean)
    ##
    def registerWin(self, hasWon):
        # method template, not implemented
        pass


##
# calcAntMove
#
# Description: This helper method sets the path for a given ant to take. It changes
# the path if a collision with another ant is found.
#
# Parameters:
#   currentState - The current game state
#   antToMove - The ant that the path is being created for
#   endDestination - The coordinate that the ant needs to end up on
#   amountOfMovement - The amount of moves the given ant can move in a turn
#
# Returns: A path (set of coordinates) for the ant to move to ((int,int),(int,int))
##
def calcAntMove(currentState, antToMove, endDestination, amountOfMovement):
    # Initial path for the ant to move towards
    path = createPathToward(currentState, antToMove.coords,
                            endDestination, amountOfMovement)
    # If no valid path towards destination was found, select random move
    # (only lists ants current coordinate)
    if len(path) == 1:
        options = listAllMovementPaths(currentState, antToMove.coords,
                                       amountOfMovement)
        path = random.choice(options)
    else:
        # To avoid collisions with other ants, checks to see if ant on current path
        for coord in path:
            ant = getAntAt(currentState, coord)
            # Skips the coordinate with the current ant
            if coord == antToMove.coords:
                continue
            # When there is an ant on the current path, move randomly
            if ant is not None:
                # Looks at all the legal moves for the ant, picks a random one
                options = listAllMovementPaths(currentState, antToMove.coords,
                                               amountOfMovement)
                path = random.choice(options)
                break
    # Returns the path for the ant to take
    return path



###
#   smallState
#
#   Descriptioin:
#       Represents a consolidated state
###
class smallState():

    def __init__(self):
        return 0

