from PG_structure import State
import numpy as np
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.gid_pairs = []
        self.mappings = []
        self.action_spaces = []
        self.dataset = None
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.gid_pairs[:]
        del self.mappings[:]
        del self.action_spaces[:]
        self.dataset = None

    def __add__(self, other):
        """Define the behavior for the '+' operator."""
        if not isinstance(other, RolloutBuffer):
            return NotImplemented
        
        result = RolloutBuffer()
        result.actions = self.actions + other.actions
        result.states = self.states + other.states
        result.logprobs = self.logprobs + other.logprobs
        result.rewards = self.rewards + other.rewards
        result.state_values = self.state_values + other.state_values
        result.is_terminals = self.is_terminals + other.is_terminals
        result.gid_pairs = self.gid_pairs + other.gid_pairs
        result.mappings = self.mappings + other.mappings
        result.action_spaces = self.action_spaces + other.action_spaces
        # For dataset, you can decide how to handle it. Here, we're simply keeping the left-hand side instance's dataset.
        result.dataset = self.dataset
        return result


    def info2state(self):
        dataset = self.dataset
        for i in range(len(self.gid_pairs)):
            g1_id, g2_id = self.gid_pairs[i]
            nn_mapping = self.mappings[i]
            action = self.actions[i]

            g1 = dataset.look_up_graph_by_gid(g1_id).get_nxgraph()
            g2 = dataset.look_up_graph_by_gid(g2_id).get_nxgraph()
            state = State(g1,g2)
            state.nn_mapping = nn_mapping
            for u,v in nn_mapping.items():
                state.ori_candidates[:,v] = False
                state.ori_candidates[u,:] = False
            matrix = state.ori_candidates
            row = matrix[action[0]]
            non_zero_columns = np.nonzero(row)[0]
            coordinates = [(action[0], col) for col in non_zero_columns]
            state.action_space = coordinates
            self.states.append(state)    
    