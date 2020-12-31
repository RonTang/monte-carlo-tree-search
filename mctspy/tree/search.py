import numpy as np
import ctypes,time

class MonteCarloTreeSearch(object):
    @staticmethod
    def para_rollout(v):
        reward = v.rollout()
        return v.id,reward
    
    def __init__(self, node, pool):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node
        self.pool = pool
    
    def best_action2(self,simulations_number):
        if self.pool:
            
            nodes = (self._tree_policy() for i in range(simulations_number))
            results = self.pool.imap_unordered(MonteCarloTreeSearch.para_rollout,nodes,chunksize = 140)
            #nodes = [self._tree_policy() for i in range(simulations_number)]
            #results = self.pool.map(MonteCarloTreeSearch.para_rollout,nodes,chunksize = 128)
            for result in results:
                node_id,reward = result
                node = ctypes.cast(node_id, ctypes.py_object).value
                node.backpropagate(reward)
         
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)
    def best_action(self, simulations_number):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------

        """
        if self.pool:
            
            nodes = (self._tree_policy() for i in range(simulations_number))
            results = self.pool.imap_unordered(MonteCarloTreeSearch.para_rollout,nodes,chunksize = 24)
            #nodes = [self._tree_policy() for i in range(simulations_number)]
            #results = self.pool.map(MonteCarloTreeSearch.para_rollout,nodes,chunksize = 128)
            for result in results:
                node_id,reward = result
                node = ctypes.cast(node_id, ctypes.py_object).value
                node.backpropagate(reward)
         
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
