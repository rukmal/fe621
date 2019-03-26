from . import cfg
from datetime import datetime
import re


def isCallOption(name: str) -> bool:
    """Function to determine the type of an option contract from its
    standard name.

    Arguments:
        name {str} -- Name of the option contract.

    Returns:
        bool -- True if Call option, False otherwise.
    """

    # Defining search Regex, see https://regexr.com for more info
    option_type_search_re = '\d{6}(C|P{1})\d{8}'

    # Searching name for option type designation
    match = re.search(pattern=option_type_search_re, string=name)

    # Return True if Call, False otherwise
    return match[1] is 'C'


def getOptionType(name: str) -> str:
    """Function to return the type of the option as a string, 'P' for Put, and
    'C' for Call.
    
    Arguments:
        name {str} -- Name of the option contract.
    
    Returns:
        str -- 'C' if call option, 'P' if put option.
    """

    return 'C' if isCallOption(name=name) else 'P'


def getStrikePrice(name: str) -> float:
    """Function to extract the strike price of an option from its
    standard name.

    Arguments:
        name {str} -- Name of the option contract.

    Returns:
        float -- Strike price of the option contract.
    """

    # Defning search Regex, see https://regexr.com for more info
    strike_price_search_re = '\d{6}.{1}(\d{5})(\d{3})'

    # Searching name for strike price
    match = re.search(pattern=strike_price_search_re, string=name)

    # Computing strike price (first part in $, second in 1/1000th $)
    strike = float(match[1]) + (float(match[2]) / 1000)

    return strike


def getExpiration(name: str) -> datetime:
    """Function to get the expiration date of the option as a datetime object,
    given the standard name of the option.

    Arguments:
        name {str} -- Name of the option contract.

    Returns:
        datetime -- Expiration date of the option contract.
    """

    # Defining search Regex, see https://regexr.com for more info
    exp_date_search_re = '.{1,}(\d{2})(\d{2})(\d{2}).{1}\d{8}'

    # Searching name for the expiration date
    match = re.search(pattern=exp_date_search_re, string=name)

    # Adding 2000 to current year (expressed in 2 digits in standard name)
    exp_year = '20' + match[1]

    # Creating date object with expiration date information
    exp_date = datetime(year=int(exp_year), month=int(match[2]),
                        day=int(match[3]))

    # Return date object with option expiration date
    return exp_date


def getTTM(name: str, current_date: str) -> float:
    """Function to compute the time to maturity (TTM) in days, given
    the current date and the standard name of an option.

    Arguments:
        name {str} -- Name of the option contract.
        current_date {str} -- Date in YYYY-MM-DD format.

    Returns:
        float -- TTM (in years, with 365 days per year).
    """

    exp_date = getExpiration(name=name)

    # Creating current date object by parsing input string
    current_date_format = '%Y-%m-%d'
    current_date = datetime.strptime(current_date, current_date_format)

    # Defining days in year
    days = cfg.days_in_year

    # Getting number of days from current date to expiration date
    ttm_days = (exp_date - current_date).days

    # Converting to years, return
    return float(ttm_days / days)
