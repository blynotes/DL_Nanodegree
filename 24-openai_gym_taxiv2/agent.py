import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.0001
        self.alpha = 1
        self.gamma = 1.0

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        state_policy = self.epsilon_greedy_probs(self.Q[state], self.epsilon)
        # Return action.
        return np.random.choice(np.arange(self.nA), p=state_policy)
    
    def epsilon_greedy_probs(self, Q_s, epsilon):
        """ update action probs according to epsilon-greedy policy """
        self.policy_s = np.ones(self.nA) * epsilon / self.nA
        self.policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return self.policy_s

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        prev_Q = self.Q[state][action]
        self.Q[state][action] = prev_Q + (self.alpha * (reward + (self.gamma * np.dot(self.Q[next_state], self.policy_s) - prev_Q)))

