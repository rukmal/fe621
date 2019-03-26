from abc import ABC, abstractmethod
from scipy import sparse
import numpy as np


class GeneralTree(ABC):
    """Abstract class enabling efficient implementation of any generalized
    binomial or trinomial tree pricing or analysis algorithm.
    
    This implementation of a general tree follows the algorithm outlined in
    my notes. See: http://bit.ly/2WjfkJu.

    This class may be inherited by a subclass that implements a specific pricing
    algorithm, while this abstract class handles tree construction, reverse
    traversal and price computation, given implementations of functions for
    computing price of children from a current node, the value of the last
    column (i.e. bottow row of leaf nodes) of a constructed price tree before
    recombination, and the value of a node given the children values.
    
    This generalized tree computation methodology allows this class to be used
    as a base for any arbitrary tree pricing or analysis tool, including
    multiplicative and additive trees. Tree values are strategically exposed at
    runtime when building and traversing the tree for added flexibility. Details
    of specific exposed runtime variables are discussed further in the
    specific function docstrings.

    Requires that `GeneralTree.childrenPrice`,
    `GeneralTree.instrumentValueAtNode`, and `GeneralTree.valueFromLastCol`
    be overridden and implemented. Specific requirements for these abstract
    methods are outlined in their respective docstrings below.
    
    Raises:
        NotImplementedError -- Raised when not implemented.
    """

    # Need to add documentation to this; explain persistent variables, etc.
    def __init__(self, price_tree_root: float, steps: int=1,
                 build_price_tree: bool=True, build_value_tree: bool=True):
        """Initialization method for the abstract `GeneralTree` class.
        
        Constructs both the price and value tree, and isolates the instrument
        price from the computed value tree.

        Provides flags to suppress the construction of the price tree and the
        value tree for flexibility. This option allows for an externally
        constructed price or value tree to be used by setting it to the
        `price_tree` and `value_tree` class variables respectively.
        
        Arguments:
            price_tree_root {float} -- Value of the root of the price tree.
        
        Keyword Arguments:
            steps {int} -- Number of steps to construct (default: {1}).
            build_price_tree {bool} -- Price tree flag (default: {True}).
            build_value_tree {bool} -- Value tree flag (default: {True}).
        
        Raises:
            ValueError -- Raised when the number of steps is invalid.
            RuntimeError -- Raised when invalid sequence is attempted. That is,
                            if the value tree is attempted to be constructed
                            without a price tree being constructed first.
        """

        self.price_tree_root = price_tree_root
        self.steps = steps

        # Check steps
        if self.steps < 1:
            raise ValueError('Must have a step size of at least 1.')

        # Construct the price tree
        if build_price_tree:
            self.price_tree = self.__constructPriceTree()

        # Construct value tree (check that price tree is constructed first)
        if build_value_tree:
            try:
                self.price_tree
            except NameError:
                raise RuntimeError('Price tree not constructed yet.')
            self.value_tree = self.__constructValueTree()

        # Set global variable with parent node value as instrument value
        try:
            self.instrument_value = self.value_tree[self.mid_row_index, 0]
        except NameError:
            raise RuntimeError('Value tree not constructed yet.')

    @abstractmethod
    def valueFromLastCol(self, last_col: np.array) -> np.array:
        """Abstract function to compute the instrument values, given the last
        column of the price matrix. That is, the bottom row of leaf nodes on
        the price tree.
        
        At runtime, the implementing class can access the current indexes,
        current node price, current child indexes, and current child values
        from the variables `self._current_row`, `self._current_col`,
        `self._current_val`, `self._child_indexes`, and `self._child_values`,
        respectively.

        See documentation for `GeneralTree.__constructValueTree` for more.

        It is required that the returned array has the same dimensions as
        argument `last_col`.
        
        Arguments:
            last_col {np.array} -- Last column of the price tree. That is, the
                                   bottom row of leaf nodes on the price tree.
        
        Raises:
            NotImplementedError -- Raised when not implemented.
        
        Returns:
            np.array -- Array of size equal to argument `last_col`.
        """

        raise NotImplementedError

    @abstractmethod
    def instrumentValueAtNode(self) -> float:
        """Abstract function to compute the instrument value at a given node.

        The implementing class can access the current indexes, current node
        price, current child indexes, and current child values from the
        variables `self._current_row`, `self._current_col`,
        `self._current_val`, `self._child_indexes`, and `self._child_values`,
        respectively.

        See documentation for `GeneralTree.__constructValueTree` for more.
        
        Raises:
            NotImplementedError -- Raised when not implemented.
        
        Returns:
            float -- Value to be set at the current node.
        """

        raise NotImplementedError

    @abstractmethod
    def childrenPrice(self) -> np.array:
        """Abstract function to compute the price of child nodes, from the
        position of the current node.

        The implementing class can access the current indexes, current node
        price, and current child indexes from the variables `self._current_row`,
        `self._current_col`, `self._current_val`, and `self._child_indexes`,
        respectively.
        
        See documentation for `GeneralTree.__constructPriceTree` for more.

        It is required that the returned array has size 3, with the format
        [up_child_price, mid_child_price, down_child_price].
        
        Raises:
            NotImplementedError -- Raised when not implemented.
        
        Returns:
            np.array -- Array of length 3 with format [up_child_price,
                        mid_child_price, down_child_price].
        """

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
        """Constructs the price tree.
        
        It is instantiated as a dictionary of keys matrix (DOK) for efficiency.
        The rows and columns are set to (2 * steps) + 1 and N + 1 respectively.
        For more on the DOK matrix, see: http://bit.ly/2HygbCT.

        The price tree is constructed following the algorithm outlined in my
        notes. See: http://bit.ly/2WhyFem.

        This function calls `childrenPrice` to get the price to set at
        the child nodes. To aid in this process, select variables are exposed
        and can be accessed via the `self` object in the class implementing
        the `childrenPrice` abstract method.
        
        Specifically, the following variables are static and set once:
            `self.nrow` -- Number of rows of the price tree matrix.
            `self.ncolumn` -- Number of columns of the price tree matrix.
            `self.mid_row_index` -- Index of the middle row of the matrix.
        
        The following variables are updated on each iteration, and deleted on
        completion of the price tree construction:
            `self._current_row` -- Current row of the iteration.
            `self._current_col` -- Current column of the iteration.
            `self._current_val` -- Price value at the current node.
            `self._child_indexes` -- Current indexes of the children nodes. Has
                                     format [up_idx, mid_idx, low_idx].
        
        Returns:
            sparse.dok_matrix -- Correctly sized DOK sparse matrix to store the
                                 price tree.
        """

        # Compute required rows and columns
        self.nrow = (2 * self.steps) + 1
        self.ncolumn = self.steps + 1

        # Instantiate sparse matrix with correct size and type
        price_tree = sparse.dok_matrix((self.nrow, self.ncolumn), dtype=float)

        # Setting root of tree to given value
        self.mid_row_index = np.floor(self.nrow / 2)
        price_tree[self.mid_row_index, 0] = self.price_tree_root

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
                deltaX = self.childrenPrice()
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
        """Constructs the value tree.

        This tree is also represented as a dictionary of keys matrix (DOK) for
        efficiency. It has the same dimensions as the price tree.

        The value tree is constructed following the algorithm outlined in my
        notes. See: http://bit.ly/2WrByt9.

        This function calls `valueFromLastCol` and `instrumentValueAtNode` to
        compute the initial last-row (i.e. bottom leaf nodes of the tree) values
        and the value of a given node at traversal, respectively. To aid in this
        process, select variables are exposed and can be accessed via the `self`
        object in the class implementeing the `valueFromLastCol` and
        `instrumentValueAtNode` abstract methods.

        The following variables are updated on each iteration, and deleted on
        completion of the value tree construction:
            `self._current_row` -- Current row of the iteration.
            `self._current_col` -- Current column of the iteration.
            `self._current_val` -- Price value at the current node.
            `self._child_values` -- Value of the current children. Has format
                                    [up_child, mid_child, down_child].
            `self._child_indexes` -- Current indexes of the children nodes. Has
                                     format [up_idx, mid_idx, low_idx].
        
        Returns:
            sparse.dok_matrix -- Value tree DOK sparse matrix with the same
                                 dimensions as `self.price_tree`.
        """

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
                # Expose corresponding current node price from `price_tree`
                self._current_val = self.price_tree[i, j]

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
