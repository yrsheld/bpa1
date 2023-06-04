import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction # map states to all possible actions

        self.qInitValue = 0  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        if state in self.Q.keys():
            return max(self.Q[state].values())
        else:
            return self.qInitValue
        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        if state in self.Q.keys() and action in self.Q[state].keys():
            return self.Q[state][action]
        else:
            return self.qInitValue
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        if state not in self.Q.keys():
            return self.getRandomAction(state)
        else:
            qvals = self.Q[state]
            return max(qvals, key=qvals.get)
            #actions, qvals = list(self.Q[state].keys()), list(self.Q[state].values())
            #best_qval = max(qvals)
            
            #return actions[np.random.choice(np.flatnonzero(np.isclose(qvals, best_qval, rtol=0.000001)))]
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            return np.random.choice(all_actions)
            # *********
        else:
            return None

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        # epsilon greedy policy
        if np.random.rand() < self.epsilon:
            return self.getRandomAction(state)
        else:
            return self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        if state not in self.Q.keys():
            all_actions = self.actionFunction(state)
            self.Q[state] = {a: self.qInitValue for a in all_actions}
        
        # update
        td_target = reward + self.discount * self.getValue(nextState)
        update = (1-self.learningRate) * self.Q[state][action]  + self.learningRate * td_target
        self.Q[state][action] = update
        
        # *********
