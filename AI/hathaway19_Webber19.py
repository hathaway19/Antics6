# -*- coding: latin-1 -*-
import random
import sys
import xml.etree.ElementTree as ET

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
        super(AIPlayer, self).__init__(inputPlayerId, "HW6_hathaway_webber")
        # Variables to store our tunnel and anthill
        self.myTunnel = None
        self.myAnthill = None
        self.STATE_FILE = "states_hathaway19_webber18.xml"
        self.DF = 0.8
        self.alpha = 0.99
        self.greedy = 0.01
        self.numGamesPlayed = 0
        self.numTrace = 5

        # Book keeping for getting the initial state
        self.initState = True

        # Food gatherer stuff
        self.myFood = None
        self.myTunnel = None

        # State memory
        self.stateMem = []

        # Tag IDs for XML document
        self.HISTORY = "history"
        self.GAMESTATE = "gs"
        self.TURNTAG = "tn"
        self.FOODLISTTAG = "fl"
        self.FOODTAG = "f"
        self.QUEENTAG = "q"
        self.WLTAG = "wl"
        self.WTAG = "w"
        self.HILLTAG = "h"
        self.TUNNELTAG = "t"
        self.SCORETAG = "s"
        self.EALISTTAG = "ea"
        self.ENEMYANTTAG = "a"
        self.UTILTAG = "u"

        # Load games from our file
        self.loadStates()

        self.utilityArray = []

        # Load are existing states if the file exists
        # if filePath.isfile(self.STATE_FILE):
        #     self.loadStates()


    ###
    #   consolidateState
    #
    #   Description:
    #       Strips a gamestate object of important information and returns a new, consolidated state object
    #       A consolidated state contains the following items:
    #           - The current turn
    #           - The location of my food
    #           - My queen
    #           - My workers
    #           - My structures
    #           - My food score
    #           - The enemy ants
    ###
    def consolidateState(self, state):

        tinyState = smallState()
        tinyState.turn = state.whoseTurn

        # Get the location of each of my foods
        foods = getConstrList(state, None, (FOOD,))
        myFood = []
        for food in foods:
            if food.coords[1] <= 4:
                myFood.append(food.coords)
        tinyState.myFood = myFood

        # Set my queen and my workers
        tinyState.myQueen = getAntList(state, tinyState.turn, (QUEEN,))[0].coords
        workers = getAntList(state, tinyState.turn, (WORKER,))
        workerCoords = []
        for w in workers:
            workerCoords.append(w.coords)
        tinyState.myWorkers = workerCoords

        # Set my structures
        tinyState.myHill = getConstrList(state, tinyState.turn, (ANTHILL,))[0].coords
        tinyState.myTunnel = getConstrList(state, tinyState.turn, (TUNNEL,))[0].coords

        # Set my food score
        tinyState.myFoodScore = getCurrPlayerInventory(state).foodCount

        # Get the enemy ants
        enemyID = PLAYER_ONE
        if tinyState.turn == PLAYER_ONE:
            enemyID = PLAYER_TWO

        enemyAnts = getAntList(state, enemyID, (WORKER, DRONE, SOLDIER, R_SOLDIER))
        enemyAntsCoords = []
        for ea in enemyAnts:
            enemyAntsCoords.append(ea.coords)
        tinyState.enemyAnts = enemyAntsCoords

        # Get the reward for this state
        tinyState.utility = self.rewardAgent(tinyState)

        return tinyState


    ###
    #   loadStates
    #
    #   Description:
    #       loads any existing states into our memory
    ###
    def loadStates(self):
        # Try to load the memory. If we fail, then we have no memory.
        try:
            tree = ET.parse("../" + self.STATE_FILE)
        except Exception as e:
            # Empty file, so do nothing
            return

        # At this point, begin adding in children of root
        root = tree.getroot()

        for state in root.iter(self.GAMESTATE):
            remState = smallState()

            for attribute in state.iter():
                tag = attribute.tag

                if tag == self.TURNTAG:
                    remState.turn = attribute.text
                elif tag == self.FOODTAG:
                    remState.myFood.append(self.textToCoord(attribute.text))

                elif tag == self.QUEENTAG:
                    remState.myQueen = self.textToCoord(attribute.text)

                elif tag == self.WTAG:
                    remState.myWorkers.append(self.textToCoord(attribute.text))

                elif tag == self.HILLTAG:
                    remState.myHill = self.textToCoord(attribute.text)

                elif tag == self.TUNNELTAG:
                    remState.myTunnel = self.textToCoord(attribute.text)

                elif tag == self.SCORETAG:
                    remState.myFoodScore = int(attribute.text)

                elif tag == self.ENEMYANTTAG:
                   remState.enemyAnts.append(self.textToCoord(attribute.text))
                elif tag == self.UTILTAG:
                    remState.utility = float(attribute.text)

            # We can now save the state in RAM
            self.stateMem.append(remState)

    def textToCoord(self, text):
        text = str(text)

        # Remove unwanted characters
        text = text.replace("(", "")
        text = text.replace(" ", "")
        text = text.replace(")", "")

        # Split into array
        text = text.split(",")

        # Turn str to int
        coords = []
        for c in text:
            coords.append(ord(c) - 48)

        if len(coords) > 1:
            list = (coords[0], coords[1])  #return list to stay consistent
            return list
        else:
            return coords[0]


    ###
    #   clearMemoryFile
    #
    #   Description:
    #       Erases all of the data in the memory file
    def clearMemoryFile(self):
        # Just open the file as write-only and then close it.. overwrites the old file
        with open(self.STATE_FILE, "w") as mf:
            mf.close()



    ###
    #   saveStates
    #
    #   Description:
    #       saves any states in our memory to our file in a CSV format
    ###
    def saveStates(self):

        # Wipe our our old memory
        self.clearMemoryFile()

        # Parse our file
        tree = None
        try:
            tree = ET.parse(self.STATE_FILE)
        except Exception as e:
            # Empty file, start a new one
            root = ET.Element(self.HISTORY)
            tree = ET.ElementTree(root)
            tree.write(self.STATE_FILE)

        # Write all of our data to our history
        root = tree.getroot()

        for tinyState in self.stateMem:

            state = ET.SubElement(root, self.GAMESTATE)

            # Turn
            eTurn = ET.SubElement(state, self.TURNTAG)
            eTurn.text = str(tinyState.turn)

            # Food
            eFood = ET.SubElement(state, self.FOODLISTTAG)
            for food in tinyState.myFood:
                foodElement = ET.SubElement(eFood, self.FOODTAG)
                foodElement.text = str(food)

            # Queen
            eQueen = ET.SubElement(state, self.QUEENTAG)
            eQueen.text = str(tinyState.myQueen)

            # Workers
            eWorkers = ET.SubElement(state, self.WLTAG)
            for w in tinyState.myWorkers:
                workerEle = ET.SubElement(eWorkers, self.WTAG)
                workerEle.text = str(w)

            # My Structs
            eHill = ET.SubElement(state, self.HILLTAG)
            eHill.text = str(tinyState.myHill)


            eTunnel = ET.SubElement(state, self.TUNNELTAG)
            eTunnel.text = str(tinyState.myTunnel)

            # my score
            eScore = ET.SubElement(state, self.SCORETAG)
            eScore.text = str(tinyState.myFoodScore)

            # enemy ants
            eEnemyAnts = ET.SubElement(state, self.EALISTTAG)
            for ant in tinyState.enemyAnts:
                eAnt = ET.SubElement(eEnemyAnts, self.ENEMYANTTAG)
                eAnt.text = str(ant)

            # Save the utility of the state
            eUtil = ET.SubElement(state, self.UTILTAG)
            eUtil.text = str(tinyState.utility)

        tree.write(self.STATE_FILE)


    ###
    #   rewardAgent
    #
    #   Description:
    #       Given a state, sets the reward for the agent of that state
    #       +1 for winning | -1 for losing | -0.001 for everything else
    ###
    def rewardAgent(self, state):

        # If we have won, reward +1
        if state.myFoodScore == 11:
            return 1.0

        # If we have lost, reward -1
        if len(state.myWorkers) == 0 and state.myFoodScore < 2:
            return -1.0
        if len(state.myQueen) == 0:
            return -1.0

        # Otherwise, just slightly punish the agent
        return -.01




    # Updated upstream
    def TD_0(self, state, alpha, gamma, numOfEpisodes=1000):
        observation = "stuff"
        for episode in range(numOfEpisodes):
            actionTaken = "stuff"
            newObservation = "stuff"
            reward = "blah"

            #Todo: call method to update utility

            observation = newObservation
            #if done:
            #p    break
        return 0

    ###
    #   adjustUtilities
    #
    #   Description:
    #       Adjusts the previous
    def adjustUtilities(self, observation, newObservation, reward, alpha, gamma):
        oldUtility = "stuff"
        newUtility = "stuff"

        return 0

    def getUtility(self, currentState, nextState):
        pass

    ##
    # findBestMove
    #
    # Description: This method goes through all the potential moves and finds the move that provides
    # the best utility.
    #
    # Parameters:
    #   currentState - The current state of the game at the time the Game is
    #       requesting a placement from the player.(GameState)
    #
    # Return: the move with the best utility
    ##
    def findBestMove(self, currentState):
        # All legal moves that we can currently take
        legalMoves = listAllMovementMoves(currentState)

        # The best move to take
        bestMove = None

        # Utility of the current best move
        utilityOfBestMove = -9999.0 # arbitrarily small

        # Go through all the potential moves to find the one with the best utility
        for move in legalMoves:
            nextState = getNextState(currentState, move)
            utilityOfCurMove = self.getUtility(currentState, nextState)

            if utilityOfCurMove >= utilityOfBestMove:
                utilityOfBestMove = utilityOfCurMove
                bestMove = move

        #Todo: Maybe add randomness here

        return bestMove

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
    # Return: If setup phase 1: list of eleven 2-tuples of ints -> [(x1,y1), (x2,y2),�,(x10,y10)]
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

        move = self.gatherFood(currentState)

        # Save our state in our memory
        if self.initState:
            tinyState = self.consolidateState(currentState)
            self.stateMem.append(tinyState)
            self.initState = False

        if not self.initState:
            if self.rewardAgent(self.consolidateState(currentState)) == 1 or \
                self.rewardAgent(self.consolidateState(currentState)) == -1:
                return move
                # Make sure we don't add a state after we win or lose
            else:
                tinyState = self.consolidateState(getNextState(currentState, move))
                self.stateMem.append(tinyState)

        # Update the previous state's utilities
        currIdx = len(self.stateMem) - 1
        if (currIdx > 0):
            for idx in range(currIdx-1, max(-1, (len(self.stateMem) - 1 - self.numTrace) - 1), -1):
                currUtil = self.stateMem[idx].utility
                rewardAtState = self.rewardAgent(self.stateMem[idx])
                self.stateMem[idx].utility = rewardAtState + self.alpha * \
                    (rewardAtState + self.DF*self.stateMem[idx + 1].utility - currUtil)

        return move


    def gatherFood(self, currentState):
        # Useful pointers
        myInv = getCurrPlayerInventory(currentState)
        me = currentState.whoseTurn

        # the first time this method is called, the food and tunnel locations
        # need to be recorded in their respective instance variables
        if (self.myTunnel == None):
            self.myTunnel = getConstrList(currentState, me, (TUNNEL,))[0]
        if (self.myFood == None):
            foods = getConstrList(currentState, None, (FOOD,))
            self.myFood = foods[0]
            # find the food closest to the tunnel
            bestDistSoFar = 1000  # i.e., infinity
            for food in foods:
                dist = stepsToReach(currentState, self.myTunnel.coords, food.coords)
                if (dist < bestDistSoFar):
                    self.myFood = food
                    bestDistSoFar = dist

        # if the hasn't moved, have her move in place so she will attack
        myQueen = myInv.getQueen()
        if (not myQueen.hasMoved):
            return Move(MOVE_ANT, [myQueen.coords], None)

        # if I don't have a worker, give up.  QQ
        numAnts = len(myInv.ants)
        if (numAnts == 1):
            return Move(END, None, None)

        # if the worker has already moved, we're done
        myWorker = getAntList(currentState, me, (WORKER,))[0]
        if (myWorker.hasMoved):
            return Move(END, None, None)

        # if the worker has food, move toward tunnel
        if (myWorker.carrying):
            path = createPathToward(currentState, myWorker.coords,
                                    self.myTunnel.coords, UNIT_STATS[WORKER][MOVEMENT])
            return Move(MOVE_ANT, path, None)

        # if the worker has no food, move toward food
        else:
            path = createPathToward(currentState, myWorker.coords,
                                    self.myFood.coords, UNIT_STATS[WORKER][MOVEMENT])
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
        # At the end of each game, save our memory into a the xml file
        self.saveStates()

        # Increment our games played and alpha value
        self.numGamesPlayed += 1
        self.alpha = 1.0 / (1.0 + (self.numGamesPlayed / 100.0) * (self.numGamesPlayed / 100.0))

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

        self.turn = None
        self.myFood = []
        self.myQueen = None
        self.myWorkers = []
        self.myHill = None
        self.myTunnel = None
        self.myFoodScore = 0
        self.enemyAnts = []
        self.utility = 0
        self.prevMove = None


