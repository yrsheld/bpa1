from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        # self.V = ...

        # ************
        self.V={}
        for s in states:
            self.V.update({s:0})
        


        for i in range(iterations):
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # print(actions)
                # print(type(actions))

                if len(actions)<1:
                    newV.update({s:0})
                # **************
                # TODO 2.1. b)
                # if ...
                #
                # else: ...

                # Update value function with new estimate
                # self.V =

                # ***************
                else:
                    values_against_actions = np.zeros(len(actions))
                    
                    # iterating over the possible action
                    for j in range(len(actions)):
                        current_action = actions[j]
                        current_state = s
                        reward = self.mdp.getReward(current_state, current_action, None)
                        succ = self.mdp.getTransitionStatesAndProbs(current_state, current_action) # list of nextstata and prob

                        #iterating over states after taking the current_action
                        for nextstate, prob in succ:
                            values_against_actions[j] += prob*(reward + self.discount*self.V[nextstate])
                    
                    best_actions_idx = np.argmax(values_against_actions)
                    newV.update({s:values_against_actions[best_actions_idx]})

            self.V = newV

            print(f'At iteration {i+1}, start state = {self.V[(4,0)]}')
        


    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2

        # **********

        return self.V[state]

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.

        # **********
        reward = self.mdp.getReward(state, action, None)
        succ = self.mdp.getTransitionStatesAndProbs(state,action) # list of nextstata and prob

        temp = 0
        for nextstate, prob in succ:
            temp += prob*(reward + self.discount*self.V[nextstate])
        return temp




    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:

        # **********
        # TODO 2.4

        # ***********
            action_q_values = np.zeros(len(actions))

            for i in range(len(actions)):
                action_q_values[i] = self.getQValue(state,actions[i])
            
            best_action_idx = np.argmax(action_q_values)

            return actions[best_action_idx]


    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
