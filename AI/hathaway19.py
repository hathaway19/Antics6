import random
import sys
import math

sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
# AIPlayer
# Description: This AI uses a neural network to approximate the state evaluation
# function.
# Assignment: Homework #5: Artificial Neural Network
#
# Due Date: April 8th, 2017
#
# @names: Justin Hathaway (no partner)
##
class AIPlayer(Player):
    # list of nodes for search tree
    node_list = []

    # maximum depth
    max_depth = 1

    # current index - for recursive function
    cur_array_index = 0

    # highest evaluated move - to be reset every time the generate_states method is called
    highest_evaluated_move = None

    # highest move score - useful for finding highest evaluated move - to be reset
    highest_move_eval = -1

    # this AI's playerID
    me = -1

    # whether or not the playerID has been set up yet
    me_set_up = False

    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "theNeuralNetAI")

        # Learning Rate of AI
        self.learningRate = 0.8
        # Number of hidden nodes and output node
        self.numOfNodes = 6

        print "rate of learning: ", self.learningRate

        # Weights that have been learned
        # (#Hidden nodes^2) for input weights
        # (#Hidden nodes) for bias on nodes
        # (#Hidden nodes) for output to output node
        # (1) for output bias
        # 5x5+5+5+1 = 40 weights total (for 4 inputs)
        # weights array which has already been learned
        self.weights = [0.747, 0.08, 0.810, 0.671,
                        0.560, 0.26, 0.541, 0.204,
                        0.278, 0.434, 0.451, 0.643,
                        0.588, 0.409, 1.012, 0.933,
                        0.797, 0.891, 0.464, 0.178,
                        0.844, 0.896, 0.646, 0.863,
                        0.864, 0.910, 1.046, 1.647,
                        1.317, 1.626, 1.1716, 0.197,
                        0.584, 0.940, 0.766, 0.737,
                        0.107, 0.093, 0.334, 0.860,
                        0.031, 0.948, 0.906, 0.934,
                        0.783, 0.645, 0.23, 0.565,
                        0.356]

        #self.weights = []
        # Calls a method to assign random values between 0 and 1 for all the weights
        #self.assignRandomWeights()

        print self.weights

    # Method to create a node containing the state, evaluation, move, current depth,
    # the parent node, and the index
    def create_node(self, state, evaluation, move, current_depth, parent_index, index):
        node = [state, evaluation, move, current_depth, parent_index, index]
        self.node_list.append(node)

    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
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
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):

        if not self.me_set_up:
            self.me = currentState.whoseTurn

        # searches for best move
        selectedMove = self.move_search(currentState, 0, -(float)("inf"), (float)("inf"))

        # if not None, return move, if None, end turn
        if not selectedMove == None:
            return selectedMove
        else:
            return Move(END, None, None)

    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy.
        return enemyLocations[0]

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
    # move_search - recursive
    #
    # uses Minimax with alpha beta pruning to search for best next move
    #
    # Parameters:
    #   game_state - current state
    #   curr_depth - current search depth
    #   alpha      - the parent node's alpha value
    #   beta       - the previous node's beta value
    #
    # Return
    #   returns a move object
    ##
    def move_search(self, game_state, curr_depth, alpha, beta):

        # if max depth surpassed, return state evaluation
        if curr_depth == self.max_depth + 1:
            return self.evaluate_state(game_state)

        # list all legal moves
        move_list = listAllLegalMoves(game_state)

        # remove end turn move if the list isn't empty
        if not len(move_list) == 1:
            move_list.pop()

        # list of nodes, which contain the state, move, and eval
        node_list = []

        # generate states based on moves, evaluate them and put them into a list in node_list
        for move in move_list:
            state_eval = 0
            state = getNextStateAdversarial(game_state, move)
            state_eval = self.evaluate_state(state)
            # print(state_eval)
            if not state_eval == 0.00001:
                node_list.append([state, move, state_eval])

        self.mergeSort(node_list)

        if not self.me == game_state.whoseTurn:
            move_list.reverse()

        best_nodes = []

        for i in range(0, 5):  # temporary
            if not len(node_list) == 0:
                best_nodes.append(node_list.pop())

        # best_val = -1

        # if not at the max depth, expand all the nodes in node_list and return
        if curr_depth <= self.max_depth:
            for node in best_nodes:
                score = self.move_search(node[0], curr_depth + 1, alpha, beta)
                if game_state.whoseTurn == self.me:
                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        # print("Pruned")
                        break
                else:
                    if score < beta:
                        beta = score
                    if alpha >= beta:
                        # print("Pruned")
                        break

        # if not curr_depth == 0:
        if game_state.whoseTurn == self.me and not curr_depth == 0:
            return alpha
        elif not game_state == self.me and not curr_depth == 0:
            return beta
        else:
            best_eval = -1
            best_node = []

            for node in best_nodes:
                if node[2] > best_eval:
                    best_eval = node[2]
                    best_node = node

            # print(len(best_node))
            if not best_node == []:
                return best_node[1]
            else:
                return None

    ##
    # get_closest_enemy_dist - helper function
    #
    # returns distance to closest enemy from an ant
    ##
    def get_closest_enemy_dist(self, my_ant_coords, enemy_ants):
        closest_dist = 100
        for ant in enemy_ants:
            if not ant.type == WORKER:
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    ##
    # get_closest_enemy_worker_dist - helper function
    #
    # returns distance to closest enemy worker ant
    ##
    def get_closest_enemy_worker_dist(self, my_ant_coords, enemy_ants):
        closest_dist = 100
        for ant in enemy_ants:
            if ant.type == WORKER:
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    ##
    # evaluate_state
    #
    # Evaluates and scores a GameState Object
    #
    # Parameters
    #   state - the GameState object to evaluate
    #
    # Return
    #   a double between 0 and 1 inclusive
    ##
    def evaluate_state(self, state):
        # The AI's player ID
        me = state.whoseTurn
        # The opponent's ID
        enemy = (state.whoseTurn + 1) % 2
        # Get a reference to the player's inventory
        my_inv = state.inventories[me]
        # Get a reference to the enemy player's inventory
        enemy_inv = state.inventories[enemy]
        # Gets both the player's queens
        my_queen = getAntList(state, me, (QUEEN,))
        enemy_queen = getAntList(state, enemy, (QUEEN,))

        # Sees if winning or loosing conditions are already met
        if (my_inv.foodCount == 11) or (enemy_queen is None):
            return 1.0
        if (enemy_inv.foodCount == 11) or (my_queen is None):
            return 0.0

        # List of inputs for neural network to consider
        listOfInputsForNetwork = []

        # Calculate the input value for food
        #foodEval = self.determineInputVal(my_inv.foodCount, enemy_inv.foodCount, 1.0)
        foodEval = my_inv.foodCount
        # Calculate input values for workers carrying and not carrying food
        moveToTunnelEval = self.moveToTunnel(state, my_inv, me)
        moveToFoodEval = self.moveToFood(state, my_inv, me)
        typeOfAntEval = self.typeOfAnt(state, my_inv, me)
        queenEval = self.queenLocation(state, my_inv, me)

        # Add the different inputs to be considered in the network
        listOfInputsForNetwork.append(foodEval)
        listOfInputsForNetwork.append(moveToFoodEval)
        listOfInputsForNetwork.append(typeOfAntEval)
        listOfInputsForNetwork.append(queenEval)
        listOfInputsForNetwork.append(moveToTunnelEval)

        outputs = self.processNetwork(listOfInputsForNetwork)
        #self.backPropagate(listOfInputsForNetwork, 1.0, outputs)
        return outputs[self.numOfNodes -1]

    ##
    # queenLocation
    #
    # One of the inputs for the network, gives a 1 if queen is off anthill
    #
    # Parameters
    #   state - the GameState object to evaluate
    #   my_inv - AI's inventory
    #   me - player id
    #
    # Return
    #   a double between 0 and 1 inclusive
    ##
    def queenLocation(self, state, my_inv, me):
        ah_coords = my_inv.getAnthill().coords
        myQueen = getAntList(state, me, (QUEEN,))

        for queen in myQueen:
            if queen.coords == ah_coords:
                return 1.0
            else:
                return 0.0

    ##
    # typeOfAnt
    #
    # One of the inputs for the network, gives a 1 if worker number is good
    #
    # Parameters
    #   state - the GameState object to evaluate
    #   my_inv - AI's inventory
    #   me - player id
    #
    # Return
    #   a double between 0 and 1 inclusive
    ##
    def typeOfAnt(self, state, my_inv, me):
        numOfWorkers = 0
        for ant in my_inv.ants:
            if ant.type == WORKER:
                numOfWorkers += 1
        if numOfWorkers == 2:
            return 1
        else:
            return 0 - numOfWorkers

    ##
    # moveToFood
    #
    # One of the inputs for the network, encourages workers to move towards food
    #
    # Parameters
    #   state - the GameState object to evaluate
    #   my_inv - AI's inventory
    #   me - player id
    #
    # Return
    #   a double between 0 and 1 inclusive
    ##
    def moveToFood(self, state, my_inv, me):
        score = 1.0
        myWorkers = getAntList(state, me, (WORKER,))
        food_coords = []
        foods = getConstrList(state, None, (FOOD,))

        # Gets a list of all of the food coords
        for food in foods:
            if food.coords[1] < 5:
                food_coords.append(food.coords)
        for worker in myWorkers:
            f1_dist = approxDist(worker.coords, food_coords[0])
            f2_dist = approxDist(worker.coords, food_coords[1])
            if not worker.carrying:
                if f1_dist < f2_dist:
                    if f1_dist == 0:
                        score += 0.2
                    score -= 0.01 * f1_dist
                else:
                    if f2_dist == 0:
                        score += 0.2
                    score -= 0.01 * f2_dist
        return score

    ##
    # queenLocation
    #
    # One of the inputs for the network, encourages workers to move towards the tunnel
    #
    # Parameters
    #   state - the GameState object to evaluate
    #   my_inv - AI's inventory
    #   me - player id
    #
    # Return
    #   a double between 0 and 1 inclusive
    ##
    def moveToTunnel(self, state, my_inv, me):
        score = 0.0
        myWorkers = getAntList(state, me, (WORKER,))
        tunnel = my_inv.getTunnels()
        t_coords = tunnel[0].coords

        for worker in myWorkers:
            t_dist = approxDist(worker.coords, t_coords)
            if worker.carrying:
                if t_dist == 0:
                    score += 0.2
                else:
                    score += 0.10 / t_dist
        return score

    ##
    # merge_sort
    #
    # useful for sorting the move list from least to greatest in nlog(n) time
    ##
    def mergeSort(self, alist):
        if len(alist) > 1:
            mid = len(alist) // 2
            lefthalf = alist[:mid]
            righthalf = alist[mid:]

            self.mergeSort(lefthalf)
            self.mergeSort(righthalf)

            i = 0
            j = 0
            k = 0
            while i < len(lefthalf) and j < len(righthalf):
                if lefthalf[i][2] < righthalf[j][2]:
                    alist[k] = lefthalf[i]
                    i = i + 1
                else:
                    alist[k] = righthalf[j]
                    j = j + 1
                k = k + 1

            while i < len(lefthalf):
                alist[k] = lefthalf[i]
                i = i + 1
                k = k + 1

            while j < len(righthalf):
                alist[k] = righthalf[j]
                j = j + 1
                k = k + 1

    ##
    # assignRandomWeights
    # Description: All the weights are given random values
    #
    # Parameters: nada
    #
    # Returns: void
    ##
    def assignRandomWeights(self):
        # inputs * hidden nodes (nodes^2), bias on nodes (nodes), output of nodes (nodes), output bias (1)sss
        numOfWeights = (self.numOfNodes) * (self.numOfNodes) + (2 * self.numOfNodes) + 1
        # Assigns random weight between 0.0 and 1.0
        for i in range(numOfWeights):
            self.weights.append(random.uniform(0.0, 1.0))

    ##
    # thresholdFunc
    # Description: This method sets the threshold function (or the 'g' function)
    # that is used in the neural network to see if the neuron fires/activates or not.
    #
    # Parameters:
    #   input - the sum of the inputs the neuron receives to apply to the threshold
    #           function
    #
    # Returns: either the output of the threshold function or its derivative
    ##
    def thresholdFunc(self, input, desiredOutput, derivative=False):
        # If we are looking for the delta or the slope
        if derivative:
            return input * (1.0 - input) * (desiredOutput - input)
        # Regular threshold function to find output of node
        else:
            return 1.0 / (1.0 + math.exp(-input))

    ##
    # processNetwork
    # Description: Calculates the output of the current neural network
    #
    # Parameters:
    #   inputs - list of inputs brought into the neural network
    #
    # Return:
    ##
    def processNetwork(self, inputs):
        # Makes an array big enough to hold the outputs of the hidden nodes and output
        # hidden nodes is indexes 0 - (self.numOfNodes - 1)
        # output is the last index
        nodeValues = []
        for i in range(self.numOfNodes):
            nodeValues.append(0)

        weightIndex = 0
        # Get the weights of the hidden nodes and the output
        while weightIndex < self.numOfNodes:
            nodeValues[weightIndex] = self.weights[weightIndex]
            weightIndex += 1

        # Calculate the values of the nodes based on the inputs and their weights
        # Go through all the inputs
        for inputIndex in range(len(inputs)):
            # Go through all the nodes
            for hiddenIndex in range(self.numOfNodes - 1):
                nodeValues[hiddenIndex] += inputs[inputIndex] * self.weights[weightIndex]
                weightIndex += 1

        # Place the resulting node output values into the threshold function
        for j in range(self.numOfNodes - 1):
            nodeValues[j] = self.thresholdFunc(nodeValues[j], 0)

        # Find the sum of the nodes' outputs to help find the output of network
        for hiddenWeightIndex in range(self.numOfNodes - 1):
            nodeValues[self.numOfNodes - 1] += nodeValues[hiddenWeightIndex] * self.weights[weightIndex]

        # Place the sum of the hidden nodes into the threshold function to see the final output
        nodeValues[self.numOfNodes - 1] = self.thresholdFunc(nodeValues[self.numOfNodes - 1], 0)

        # Returns list of the outputs of the hidden and output nodes
        return nodeValues

    ##
    # backPropagate
    # Description: Goes backward through the neural network to find the error from the desired
    # outputs and changes the weights to minimize the outputs from the desired goal
    #
    # Parameters:
    #   inputs - list of inputs brought into the neural network
    #   currentOutput - list of outputs from the nodes including output of network
    #
    # Return: void
    ##
    def backPropagate(self, inputs, desiredOutput, currentOutput):
        # Number of inputs for the network
        numOfInputs = len(inputs)
        # Number of weights for the network (inputs, node biases, hidden node outputs, and output bias)
        numOfWeights = len(self.weights)
        # Number of inputs into the hidden nodes
        numOfInputWeights = (self.numOfNodes - 1) * (numOfInputs - 1) + self.numOfNodes
        # Arrays to hold errors and deltas for each of the nodes
        errorOfHiddenNodes = []
        deltaOfHiddenNodes = []

        deltaOfOutput = self.thresholdFunc(currentOutput[self.numOfNodes - 1],
                                           desiredOutput, derivative=True)

        # Places enough spots in arrays to hold the errors and deltas for each node
        for i in range(self.numOfNodes - 1):
            errorOfHiddenNodes.append(0)
            deltaOfHiddenNodes.append(0)

        # Calculate the deltas and errors of the hidden nodes and not the output of network
        for j in range(len(deltaOfHiddenNodes)):
            #Find the error of the current node
            errorOfHiddenNodes[j] = self.weights[j + numOfInputWeights + 1] * deltaOfOutput
            # Find the delta based on the error
            deltaOfHiddenNodes[j] = currentOutput[j] * (1 - currentOutput[j]) * errorOfHiddenNodes[j]

        # Adds the delta value of the output to the list as well
        deltaOfHiddenNodes.append(deltaOfOutput)

        # Todo: get rid of print statements
        print "**************************************"
        print "old weights!"
        for m in range(numOfWeights - 1):
            print "weight: ", self.weights[m]

        # Go through all the weights in the network
        for currentWeightIndex in range(numOfWeights - 1):
            # Check to see which node the weight belongs to
            # (if in the part of weights that belong to inputs into the nodes)
            if currentWeightIndex < self.numOfNodes:
                currentNodeIndex = currentWeightIndex % self.numOfNodes
                inputEntered = 1.0 # Bias for nodes is always 1 for this assignment
            elif currentWeightIndex > numOfWeights - self.numOfNodes:
                # The current node index is the index of the output node
                currentNodeIndex = self.numOfNodes - 1
                currentInputIndex = currentWeightIndex - (numOfWeights - self.numOfNodes)
                inputEntered = currentOutput[currentInputIndex]
            else:
                currentNodeIndex = (currentWeightIndex - 1) % (self.numOfNodes - 1)
                currentInputIndex = (currentWeightIndex - self.numOfNodes) / (self.numOfNodes - 1)
                inputEntered = inputs[currentInputIndex]

            # Alter the weights based on the learning rate, the input into the node, and the change
            self.weights[currentWeightIndex] += self.learningRate * inputEntered * deltaOfHiddenNodes[currentNodeIndex - 1]

        print "**************************************"
        print "new weights!"
        for m in range(numOfWeights - 1):
            print "weight: ", self.weights[m]

# Unit Tests

##
#  Test 1 (2 inputs, 2 hidden nodes, 1 output)
##
# print "*** Test Case 1 ***",
# player = AIPlayer(0)
# player.numOfNodes = 3 #(2 hidden, 1 output)
# print "player num of nodes: ", player.numOfNodes
# player.learningRate = 0.2
# # 9 weights total (2 input, 2 bias, 2 output of hidden, 1 output)
# player.weights = [0.1, 0.2, 0.4, 0.5, 0.8,
#                   0.66, 0.3, 0.14, 0.22]
# # 2 inputs to test network
# inputs = [1, 1]
# # output that we should receive
# desiredOutput = 0.2
# # current output that we are getting
# currentOutput = player.processNetwork(inputs)
# # error between the desired output and the current output
# # error = target - actual
# error = desiredOutput - currentOutput[player.numOfNodes - 1]
# # Check to see if the error is small enough to stop
# if (-0.1 < error) and (error < 0.1):
#     print "error is in acceptable limits",
# else:
#     print "network still needs to learn",
#     print "current error: ", error
# #edit the weights by back propagating through the network
# player.backPropagate(inputs, desiredOutput, currentOutput)
# if (-0.1 < error) and (error < 0.1):
#     print "error is in acceptable limits",
# else:
#     print "network still needs to learn",
#     print "current error: ", error

# print "*** Test Case 3 ***",
# player = AIPlayer(0)
# player.numOfNodes = 6 #(5 hidden, 1 output)
# print "player num of nodes: ", player.numOfNodes
# player.learningRate = 0.8
# # 9 weights total (2 input, 2 bias, 2 output of hidden, 1 output)
# player.weights = [0.7472926132800758, 0.07757247167086267, 0.8142091589830514,
#                   0.677460353294787, 0.565767633676064,
#                   0.2566264521602335, 0.5406016459529106,
#                   0.2039202335814032, 0.26751601886430165, 0.43394651920846306,
#                   0.4511704880743833, 0.6429024939671476, 0.5883028890327574,
#                   0.40949895001347913, 0.01181458129644708,
#                   0.9235345685100295, 0.7960392172229361, 0.8913173226298294,
#                   0.4640029538057835, 0.17786404745487971,
#                   0.8441472545265545, 0.2952573268412545, 0.6456093841941078,
#                   0.8631780703422346, 0.8641047097206673, 0.5101909942720363,
#                   0.04564304954163079, 0.6468525128311484, 0.3171869628349089,
#                   0.6257850828798243, 0.9715964713459734, 0.19719259755961038,
#                   0.5842139051643824, 0.9400475070586065, 0.7655399686033227,
#                   0.7369782781994593, 0.10693355112651837, 0.09303092400501178,
#                   0.3344600936394564, 0.8601592364099246, 0.031152643579963946,
#                   0.9477010186833652, 0.9058134305833268, 0.9341980480479557,
#                   0.7829029125825527, 0.6448255109034774, 0.22908072645894273,
#                   0.5647849340974963, 0.3563477261625805]
# # 2 inputs to test network
# inputs = [1.0, 0.9, 0.88, 0.0, 0.75]
# # output that we should receive
# desiredOutput = 1.0
# # current output that we are getting
# currentOutput = player.processNetwork(inputs)
# # error between the desired output and the current output
# # error = target - actual
# error = desiredOutput - currentOutput[player.numOfNodes - 1]
# # Check to see if the error is small enough to stop
# if (-0.1 < error) and (error < 0.1):
#     print "error is in acceptable limits",
# else:
#     print "network still needs to learn",
#     print "current error: ", error
# #edit the weights by back propagating through the network
# player.backPropagate(inputs, desiredOutput, currentOutput)
# if (-0.1 < error) and (error < 0.1):
#     print "error is in acceptable limits",
# else:
#     print "network still needs to learn",
#     print "current error: ", error

# ##
# #  Test 2 (8 inputs, 8 hidden nodes, 1 output)
# ##
# print "***Test Case 2 ***",
# player = AIPlayer(0)
# player.numOfNodes = 9 #(8 hidden, 1 output)
# player.learningRate = 0.5
# # 81 weights total (64 input, 8 bias, 8 output of hidden, 1 output)
# player.weights = [0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.6,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.1, 0.2, 0.4, 0.5, 0.8, 0.66, 0.3, 0.14, 0.7,
#                   0.22]
# # 8 inputs to test network
# inputs = [1, 1, 0, 0, 1, 0, 1, 1]
# # output that we should receive
# desiredOutput = 0.2
# # current output that we are getting
# currentOutput = player.processNetwork(inputs)
# # error between the desired output and the current output
# # error = target - actual
# error = desiredOutput - currentOutput[player.numOfNodes - 1]
# # Check to see if the error is small enough to stop
# if (-0.1 < error) and (error < 0.1):
#     print "error is in acceptable limits",
# else:
#     print "network still needs to learn",
# # edit the weights by back propagating through the network
# player.backPropagate(inputs, desiredOutput, currentOutput[player.numOfNodes - 1])
