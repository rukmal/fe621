library("Rblpapi")

# Connect to Bloomberg Terminal backend service
blpConnect(host = "localhost", port = 8194)


#----------------------------------
# Data Download Functionality
#----------------------------------


getPrice <- function(security, startTime, endTime, timeZone) {
  # Downloads and returns the closing price of a given security
  # for each minute in the trading day.
  #
  # Args:
  #   security: Name of the security to be downloaded.
  #   startTime: Datetime object with the start time.
  #   endTime: Datetime object with the end time.
  #   timeZone: Time zone of the target start and end times.
  #
  # Returns:
  #   DataFrame with the closing price for each minute in the
  #   trading day.
  
  # Getting price data
  data <- getBars(security = security, barInterval = 1,
                  startTime = startTime, endTime = endTime,
                  tz = timeZone)
  
  # Isolate time and closing price
  data <- data[c("times", "close")]
  
  # Rename columns
  colnames(data) <- c("Dates", "Close")
  
  # Return
  data
}


createOptionName <- function(security, dates, prices, type, suffix) {
  # Creates the Bloomberg-standard option name, given a security, date, price,
  # option type and suffix.
  #
  # Args:
  #   security: Name of the security to be included in the option price.
  #   dates: Dates to be included in option name.
  #   prices: Prices to be included in the option name.
  #   type: Type of the option ("C" or "P").
  #   suffix: Suffix for option name (typically "Index" or "Equity").
  #
  # Returns:
  #   Vector of Bloomberg-compatible option names.

  # Empty vector to store names
  names <- c()

  # Iterate over each date and price
  for (date in dates) {
    for (price in prices) {
      # Building option name
      name <- paste(security, date, paste(type, price, sep = ""), suffix)

      # Appending to list of option names
      names <- c(names, name)
    }
  }

  # Returning names
  names
}


#----------------------------------
# DATA1
#----------------------------------


# Define Start and End times (DATA1)
data1Start <- ISOdatetime(year = 2019, month = 2, day = 6,
                          hour = 9, min = 30, sec = 0)
data1End <- ISOdatetime(year = 2019, month = 2, day = 6,
                        hour = 16, min = 0, sec = 0)

# Defining time zone
timeZone = "America/New_York"

# Defining top-level securities
securities <- c("SPY US Equity", "AMZN US Equity", "VIX Index")

# Getting prices for each of the top-level securities
for (security in securities) {
  data <- getPrice(security, data1Start, data1End, timeZone)
  write.csv(data, file = paste(security, "DATA1", "csv", sep = "."),
            row.names = FALSE)
}

# Expiration dates
expDates <- c("2/15/19", "3/15/19", "4/18/19")

# Defining put and call prices for SPY and AMZN options
# Grabbing prices for 15% +/- current price

# Defining bounds
lowerBoundPct <- 0.85
upperBoundPct <- 1.15

# Current SPY price
spyCurrent <- 270
spyPrices <- c(floor(
  lowerBoundPct * spyCurrent):ceiling(upperBoundPct * spyCurrent))

# Function to round to the nearest 'base', given an input 'x'. This is to
# compute strike prices for AMZN options, which are in intervals of 5.
# Source: http://r.789695.n4.nabble.com/Rounding-to-the-nearest-5-td863189.html
mround <- function(x, base) {
  base * round(x / base)
}

# Current AMZN price (need to do this manually because of option strikes)
# Closest option price to 95% of price is at $1557.50 and 105% is $1722.50
amznCurrent <- 1640
roundingLevel <- 5
amznPrices <- seq(mround(amznCurrent * lowerBoundPct, roundingLevel),
                  mround(amznCurrent * upperBoundPct, roundingLevel), by=5)

# Creating option names for SPY and AMZN
spyOptions <- createOptionName("SPY", expDates, spyPrices, "C", "Equity")
spyOptions <- c(spyOptions, createOptionName("SPY", expDates, spyPrices,
                                             "P", "Equity"))

amznOptions <- createOptionName("AMZN", expDates, amznPrices, "C", "Equity")
amznOptions <- c(amznOptions, createOptionName("AMZN", expDates, amznPrices,
                                               "P", "Equity"))

# Getting prices for each of the options
for (option in c(amznOptions, spyOptions)) {
  data <- getPrice(option, data1Start, data1End, timeZone)
  # Only print to file if option exists
  if (all(dim(data) > 0)) {
    optionFileName <- gsub("/", "-", option) # Need to do this for Windows
    write.csv(data, file = paste(optionFileName, "csv", sep = "."),
              row.names = FALSE)
  }
}


#----------------------------------
# DATA2
#----------------------------------

# Define Start and End times (DATA2)
data2Start <- ISOdatetime(year = 2019, month = 2, day = 7,
                          hour = 9, min = 30, sec = 0)
data2End <- ISOdatetime(year = 2019, month = 2, day = 7,
                        hour = 16, min = 0, sec = 0)

# Getting prices for each of the top-level securities
for (security in securities) {
  data <- getPrice(security, data2Start, data2End, timeZone)
  write.csv(data, file = paste(security, "DATA2", "csv", sep = "."),
            row.names = FALSE)
}
