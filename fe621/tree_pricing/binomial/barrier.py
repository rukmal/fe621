from ..general_tree import GeneralTree

import numpy as np


class Barrier(GeneralTree):
    """Barrier option pricing with a Trigeorgis tree.
    
    Implemented with the `GeneralTree` abstract class.
    """

    def __init__(self, current: float, strike: float, ttm: float, rf: float,
                 volatility: float, barrier: float, barrier_type: str,
                 opt_type: str, opt_style: str, steps: int=1):
        """Initialization method for the `Barrier` class.
        
        Arguments:
            current {float} -- Current asset price.
            strike {float} -- Strike price of the option.
            ttm {float} -- Time to maturity of the option (in years).
            rf {float} -- Risk-free rate (annualized).
            volatility {float} -- Volatility of the underlying asset price.
            barrier {float} -- Barrier of the option (sometimes called 'H').
            barrier_type {str} -- Barrier type, 'O' for out and 'I' for in.
            opt_type {str} -- Option type, 'C' for Call, 'P' for Put.
            opt_style {str} -- Option style, 'E' for European, 'A' for American.
        
        Keyword Arguments:
            steps {int} -- Number of steps to construct (default: {1}).
        """

        # Ensuring valid option type and style
        if opt_type not in ['C', 'P'] or opt_style not in ['A', 'E']:
            raise ValueError('`opt_type` must be \'C\' or \'P\' and `opt_style`\
                must be \'A\' or \'E\'.')
        
        # Ensuring valid barrier option type
        if barrier_type not in ['I', 'O']:
            raise ValueError('`barrier_type` must be \'I\' for In type options,\
                or \'O\' for Out type options.')
        
        # Inferring barrier option characteristic from current price
        # i.e. either Up or Down
        if barrier > current:
            self.barrier_characteristic = 'U'
        else:
            self.barrier_characteristic = 'D'
        
        # Setting class variables
        self.opt_type = opt_type
        self.opt_style = opt_style
        self.rf = rf
        self.volatility = volatility
        self.strike = strike
        self.barrier = barrier
        self.barrier_log = np.log(barrier)
        self.barrier_type = barrier_type
        
        # Computing deltaT
        deltaT = ttm / steps

        # Computing upward and downward jumps for children
        # Do this only once so it doesn't have to be recomputed each time
        # Upward additive deltaX
        self.deltaXU = np.sqrt((np.power(rf - (np.power(volatility, 2) / 2), 2)\
                               * np.power(deltaT, 2)) + (np.power(volatility,
                               2) * deltaT))
        # Down deltaX = -1 * upDeltaX
        self.deltaXD = -1 * self.deltaXU

        # Computing jump probabilities for value tree construction
        # Do this only once so it doesn't have to be recomputed each time
        self.jumpU = 0.5 + (0.5 * (rf - (np.power(volatility, 2) / 2)) * deltaT\
                            / self.deltaXU)
        self.jumpD = 1 - self.jumpU

        # Define discount factor for each jump
        self.disc = np.exp(-1 * rf * deltaT)

        # Initializing GeneralTree, with root set to log price for Trigeorgis
        super().__init__(price_tree_root=np.log(current), steps=steps)
    
    def childrenPrice(self) -> np.array:
        """Function to compute the price of children nodes, given the price at
        the current node.
        
        Returns:
            np.array -- Array of length 3 corresponding to [up_child_price,
                        mid_child_price, down_child_price].
        """

        # Computing up and downward child additive values (mid is 0)
        up_child_price = self._current_val + self.deltaXU
        down_child_price = self._current_val + self.deltaXD
        
        # Computing barrier indicator for each of the child prices
        up_child_price = up_child_price \
            * self.barrierIndicator(np.exp(up_child_price))
        down_child_price = down_child_price \
            * self.barrierIndicator(np.exp(down_child_price))

        return np.array([up_child_price, 0, down_child_price])

    def instrumentValueAtNode(self) -> float:
        """Function to compute the instrument value at the given node.

        Intelligently adapts to the specificed option style (`self.opt_style`)
        and type (`self.opt_type`) to work with both European options, and the
        path-dependent American option style.
        
        Returns:
            float -- Value of the option at the given node.
        """

        # Value implied by children
        child_implied_value = self.disc * ((self.jumpU * self._child_values[0])\
                                + (self.jumpD * self._child_values[2]))

        # American option special case
        # NOTE: It is path dependent, so evaluate option value at current node
        #       and return if higher than `child_implied_value`
        if self.opt_style == 'A':
            # Computing value of option if exercied at current node
            # NOTE: Using `valueFromLastCol` here as it is the same computation;
            #       casting current node value to array and passing thru
            option_value = self.valueFromLastCol(last_col=np.array([
                self._current_val]))[0]

            # If value is higher than `child_implied_value`, exercise now
            if option_value > child_implied_value:
                return option_value

        return child_implied_value

    def valueFromLastCol(self, last_col: np.array) -> np.array:
        """Function to compute the option value of the last column (i.e. last
        row of leaf nodes) of the price tree.
        
        Arguments:
            last_col {np.array} -- Last column of the price tree.
        
        Returns:
            np.array -- Value of the option corresponding to the input prices.
        """

        # Computing prices of each of the values
        underlying_prices = np.array([0 if i == 0 else np.exp(i)
                                        for i in last_col])

        # Computing barrier indicator function values for last_col values
        indicator_val = np.array([[self.barrierIndicator(i)]
                                    for i in underlying_prices])

        # Call option (same for European and American)
        if self.opt_type == 'C':
            # Computing non-floored call option value
            non_floor_val = np.exp(last_col) - self.strike
        
        # Put option (same for European and American)
        if self.opt_type == 'P':
            # Computing non-floored put option value
            non_floor_val = self.strike - np.exp(last_col)

            # Replacing values equal to (self.strike - 1) with 0. This is to
            # adjust for the fact that zero nodes would have this value in
            # the tree.
            # This is a special case adjustment that must be made to
            # computation. This is purely for clarity.
            non_floor_val = np.where(non_floor_val == (self.strike - 1), 0,
                                     non_floor_val)

        # Floor to 0
        floor_val = np.where(non_floor_val > 0, non_floor_val, 0)

        # Return element-wise product of floor_val and barrier indicator
        return np.multiply(indicator_val, floor_val)
    
    def barrierIndicator(self, underlying_price: float) -> int:
        """Indicator for the barrier option.
        
        Returns the corresponding value for the barrier indicator function,
        depending on the barrier type and barrier characteristic.
        
        Arguments:
            underlying_price {float} -- Log price of the underlying asset.
        
        Returns:
            int -- 1 or 0 depending on indicator function.
        """

        # Down and out
        if (self.barrier_characteristic == 'D') and (self.barrier_type == 'O'):
            return 1 if underlying_price > self.barrier else 0

        # Down and in
        if (self.barrier_characteristic == 'D') and (self.barrier_type == 'I'):
            return 1 if underlying_price <= self.barrier else 0
        
        # Up and out
        if (self.barrier_characteristic == 'U') and (self.barrier_type == 'O'):
            return 1 if underlying_price < self.barrier else 0
        
        # Up and in
        if (self.barrier_characteristic == 'U') and (self.barrier_type == 'I'):
            return 1 if underlying_price >= self.barrier else 0

    def getPriceTree(self) -> np.array:
        """Function to get the price tree. Overrides superclass function of the
        same name to return the real price tree as opposed to to the
        log-price tree.
        
        Returns:
            np.array -- Constructed price tree.
        """

        # Getting log price tree from superclass method
        log_price_tree = super().getPriceTree()
        # Computing real price tree
        price_tree_unadj = np.exp(log_price_tree)

        # Replacing all instances of value '1' with zero, as it would have
        # previously been a zero node before exponentiation
        return np.where(price_tree_unadj == 1, 0, price_tree_unadj)
