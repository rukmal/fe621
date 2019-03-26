from abc import ABC, abstractmethod
from scipy import sparse
from typing import Callable
import numpy as np


class GeneralTree(ABC):
    # Need to add documentation to this; explain persistent variables, etc.
    def __init__(self, tree_root: float, ttm: float, steps: int=1):
        # Be sure to mention that the variables `self.children` is set at each
        # iteration, and can be accessed `getProbabilities`
        self.tree_root = tree_root
        self.ttm = ttm
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

        # Construct value tree
        self.value_tree = self.__constructValueTree()

        # Set global variable with parent node value as instrument value
        self.instrument_value = self.value_tree[self.mid_row_index, 0]

    @abstractmethod
    def valueFromLastCol(self, last_col: np.array) -> np.array:
        # Compute and return the value of the last column, given the prices
        raise NotImplementedError

    @abstractmethod
    def instrumentValueAtNode(self) -> float:
        # Function to compute the value of the instrument at a node, given the
        # underlying price
        raise NotImplementedError

    @abstractmethod
    def getChildren(self) -> np.array:
        # update deltaX (i.e. the steps)
        # Must return an array of size 3
        raise NotImplementedError
    
    def getPriceTree(self) -> np.array:
        """Get the constructed price tree.
        
        Returns:
            np.array -- Constructed price tree (matrix representation).
        """

        return self.price_tree.toarray()
    
    def getValueTree(self) -> np.array:
        """Get the constructed value tree.
        
        Returns:
            np.array -- Constructed value tree (matrix representation).
        """

        return self.value_tree.toarray()
    
    def getInstrumentValue(self) -> float:
        """Get the value of the instrument as implied by the value tree.
        
        Returns:
            float -- Value of the instrument as implied by the value tree.
        """

        return self.instrument_value


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
        self.nrow = (2 * self.steps) + 1
        self.ncolumn = self.steps + 1

        # Instantiate sparse matrix with correct size and type
        price_tree = sparse.dok_matrix((self.nrow, self.ncolumn), dtype=float)

        # Setting root of tree to given value
        self.mid_row_index = np.floor(self.nrow / 2)
        price_tree[self.mid_row_index, 0] = self.tree_root

        # Iterate over columns
        for j in range(0, self.ncolumn - 1):
            # NOTE: The following optimization iterates only over the non-zero
            #       rows. Determined using the triangular pattern of tree data.
            #       Ensures that we will never encounter a node with value 0
            offset = row_low = self.steps - j
            row_high = self.nrow - offset

            # Iterate over rows:
            for i in range(row_low, row_high):
                # Skip to next iteration if current node is 0
                if price_tree[i, j] == 0:
                    continue

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

        # Delete intermediate exposed variables
        del self._current_row
        del self._current_col
        del self._current_val
        del self._child_indexes

        # Return final price tree
        return price_tree

    def __constructValueTree(self) -> sparse.dok_matrix:
        # Creating copy of price tree for the value tree
        value_tree = self.price_tree.copy()
        # Applying value function to the last column of child nodes
        last_row = self.valueFromLastCol(
            last_col=value_tree[:, self.ncolumn - 1].toarray()
        )

        # Updating last column values
        # NOTE: I realize that the loop here is inefficient, but dok_matrix does
        #       not support sliced value setting (as far as I can tell)
        for i in range(0, self.nrow):
            value_tree[i, self.ncolumn - 1] = last_row[i]

        # Iterate over columns (starting with the one-before-last column)
        for j in reversed(range(0, self.ncolumn - 1)):
            # NOTE: The following optimization iterates only over the non-zero
            #       rows. Determined using the triangular pattern of tree data.
            #       Ensures that we will never encounter a node with value 0
            offset = row_low = self.steps - j
            row_high = self.nrow - offset

            for i in range(row_low, row_high):
                # Skip to next iteration if current node is 0
                if value_tree[i, j] == 0:
                    continue

                # Making current i, j and value global for external visiblity
                self._current_row = i
                self._current_col = j
                self._current_val = value_tree[i, j]

                # Update children indexes
                self.__updateChildIndexes()

                # Building 3x1 array of child values, making globally visible
                child_row_range = range(self._child_indexes[0][0],
                                        self._child_indexes[2][0] + 1)
                self._child_values = value_tree[child_row_range, j + 1]\
                    .toarray()

                # Set value of current node
                value_tree[i, j] = self.instrumentValueAtNode()
        
        # Delete intermediate exposed variables
        del self._current_row
        del self._current_col
        del self._current_val
        del self._child_indexes
        del self._child_values

        # Return final value tree
        return value_tree

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
