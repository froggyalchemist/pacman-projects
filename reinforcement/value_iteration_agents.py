# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# value_iteration_agents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learning_agents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state, action, next_state)
              mdp.is_terminal(state)
        """
        super().__init__()
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.run_value_iteration()
    
    def run_value_iteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        # Start with an arbitrary value function estimate V_0 -> self.values is already initialised to 0

        # For k = 0, 1, ..., iterations - 1, compute V_k+1 = T_star · V_k
        for k in range(self.iterations):

            # new_values holds V_{k+1}
            # self.values holds V_k. All Q values for V_{k+1} are estimated from the ones in iteration k.
            new_values = util.Counter()

            # Update values of all states V_k+1(s) = max_a Q_k(s,a)
            for s in self.mdp.get_states():

                actions = self.mdp.get_possible_actions(s)
                if not actions or self.mdp.is_terminal(s):
                    # No actions = no future rewards, so V(s) = 0.
                    new_values[s] = 0

                else:
                    # Find Q_k+1(s, a) for every action, then take the best one
                    q_values = [self.compute_q_value_from_values(s, a) for a in actions]
                    new_values[s] = max(q_values)

            # V_k+1 becomes the current estimate for the next iteration
            self.values = new_values

    def get_value(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # This function implements the following formulas from the lecture slides:
        # R(s,a) = Σ_s' P(s'|s,a) · R(s,a,s')
        # Q*(s,a) = R(s,a) + γ · Σ_s' P(s'|s,a) · V(s')
        #         = Σ_s' P(s'|s,a) · R(s,a,s') + γ · Σ_s' P(s'|s,a) · V(s')
        q_value = 0
        for (s_next, prob) in self.mdp.get_transition_states_and_probs(state, action):
            q_value += prob * self.mdp.get_reward(state, action, s_next) # Reward for performing action a in state s R(s,a)
            q_value += prob * self.discount * self.values[s_next]        # Discounted expected future reward
        return q_value

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.get_possible_actions(state)
        if not actions:
            return None
        else:
            action_values = util.Counter()
            for a in actions:
                # Q-value of taking action a in this state (includes reward and discount)
                action_values[a] = self.compute_q_value_from_values(state, a)

            # Best action is the one with the highest Q value
            return action_values.arg_max()

    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        return self.compute_q_value_from_values(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learning_agents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state)
              mdp.is_terminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """*** YOUR CODE HERE ***"""

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """*** YOUR CODE HERE ***"""
