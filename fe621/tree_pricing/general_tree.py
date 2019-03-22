from abc import ABC, abstractmethod
from scipy import sparse
from typing import Callable
import numpy as np


class GeneralTree(ABC):
    # Need to add documentation to this; explain persistent variables, etc.
    def __init__(self, current: float, strike: float, ttm: float, rf: float,
                 steps: int=1):
        # Be sure to mention that the variables `self.children` is set at each
        # iteration, and can be accessed `getProbabilities`
        self.current = current
        self.ttm = ttm
        self.strike = strike
        self.rf = rf
        self.steps = steps

        # Computing delta t (i.e. change in time) for each step
        self.deltaT = ttm / self.steps

        # Check implemented method arguments
        self.__checkImplementationMethods()

        # Check steps
        if self.steps < 1:
            raise ValueError('Must have a step size of at least 1.')

        # Construct the price tree
        self.price_tree = self.__constructPriceTree()

    @abstractmethod
    def getJumpProbability(self, child_values: float) -> np.array:
        # Compute and return array of probabilities, given the child values
        # The array will have size 3, with probabilities [p_up, p_mid, p_down]
        # This would be static for equal probability trees
        raise NotImplementedError

    @abstractmethod
    def instrumentValueAtNode(self, node_value: float) -> float:
        # Function to compute the value of the instrument at a node, given its
        # price
        raise NotImplementedError
    
    @abstractmethod
    def instrumentValueFromChildren(self, children_value: np.array) -> float:
        # Compute the instrument value, given the value of the child nodes
        # children_value must be an array of size 3, with arguments child_up,
        # child_mid, and child_down
        raise NotImplementedError

    @abstractmethod
    def getChildren(self) -> np.array:
        # update deltaX (i.e. the steps)
        # Must return an array of size 3
        raise NotImplementedError

    @abstractmethod
    def getVolatility(self) -> float:
        # Compute the volaility
        raise NotImplementedError
    
    def getPriceTree(self) -> np.array:
        """Get the constructed price tree.
        
        Returns:
            np.array -- Constructed price tree (matrix representation).
        """

        return self.price_tree.toarray()

    def __constructPriceTree(self) -> sparse.dok_matrix:
        """Constructs the price tree. It is instantiated as a dictionary of
        keys matrix (DOK) for efficiency. The rows and columns are set to
        (2 * steps) + 1 and N + 1 respectively. For more on the CSC matrix,
        see: http://bit.ly/2HygbCT.
        The price tree is constructed following the algorithm outlined in my
        notes. See: http://bit.ly/2WhyFem.
        
        Returns:
            sparse.dok_matrix -- Correctly sized sparse column matrix to store
                                 the price tree.
        """

        # Compute required rows and columns
        nrow = (2 * self.steps) + 1
        ncolumn = self.steps + 1

        # Instantiate sparse matrix with correct size and type
        price_tree = sparse.dok_matrix((nrow, ncolumn), dtype=float)

        # Setting root of tree to current price
        mid_row_index = np.floor(nrow / 2)
        price_tree[mid_row_index, 0] = self.current

        # Iterate over columns
        for j in range(0, ncolumn - 1):
            # NOTE: The following optimization iterates only over the non-zero
            #       rows. Determined using the triangular pattern of tree data.
            #       Ensures that we will never encounter a node with value 0
            offset = row_low = self.steps - j
            row_high = nrow - offset

            # Iterate over rows:
            for i in range(row_low, row_high):
                # Making current i, j, and value global for external visibility
                self._current_row = i
                self._current_col = j
                self._current_val = price_tree[i, j]

                # Update children indexes
                self.__updateChildIndexes()
                # Get deltaX
                deltaX = self.getChildren()
                # Update child values
                for idx, child_delX in zip(self._child_indexes, deltaX):
                    price_tree[idx[0], idx[1]] = child_delX

        # Delete current row and column variables
        del self._current_row
        del self._current_col
        del self._current_val
        del self._child_indexes

        # Return final price tree
        return price_tree
                

    def __updateChildIndexes(self) -> np.array:
        """Function to update the `self._child_indexes` with the correct values,
        given the current row index (i), `self._current_row`, and the current
        column index (j), `self._current_col`. `self._child_indexes` is set to a
        tuple (len 3) of tuples (len 2; indexes) with the values,
        corresponding to: ((up_i, up_j), (mid_i, mid_j), (down_i, down_j)).
        
        Arguments:
            row_idx {int} -- Current row index.
            col_idx {int} -- Current column index.
        """

        self._child_indexes = (
            [self._current_row - 1, self._current_col + 1],
            [self._current_row, self._current_col + 1],
            [self._current_row + 1, self._current_col + 1]
        )

    def __checkImplementationMethods(self):
        # Methods to check: instrumentValueFromChildren, instrumentValueAtNode,
        # getProbabilities, getChildren
        pass
