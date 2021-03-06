# -*- coding: utf-8 -*-
"""02-03-pandas-foundations.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pfJIlN8KlItYPvGhK6_-RUzyDOw8C6_b
"""

#pandas Foundations
## 1. Data ingestion & inspection
## 2. Exploratory data analysis
## 3. Time series in pandas
## 4. Case Study - Sunlight in Austin

"""## 1. Data ingestion & inspection"""

import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/world_ind_pop_data.csv'
df = pd.read_csv(fname, index_col=0)

df.head()
df.tail()
df.info()

"""**Inspecting your data**

You can use the DataFrame methods .head() and .tail() to view the first few and last few rows of a DataFrame. In this exercise, we have imported pandas as pd and loaded population data from 1960 to 2014 as a DataFrame df. This dataset was obtained from the World Bank.

Your job is to use df.head() and df.tail() to verify that the first and last rows match a file on disk. 

In later exercises, you will see how to extract values from DataFrames with indexing, but for now, manually copy/paste or type values into assignment statements where needed. 

Select the correct answer for the first and last values in the 'Year' and 'Total Population' columns.

**Possible Answers**

- [ ] First: 1980, 26183676.0; Last: 2000, 35.
- [x] First: 1960, 92495902.0; Last: 2014, 15245855.0.
- [ ] First: 40.472, 2001; Last: 44.5, 1880.
- [ ] First: CSS, 104170.0; Last: USA, 95.203.

**DataFrame data types**

Pandas is aware of the data types in the columns of your DataFrame. It is also aware of null and NaN ('Not-a-Number') types which often indicate missing data. In this exercise, we have imported pandas as pd and read the world population data into a DataFrame df which contains some NaN values — a value often used as a place-holder for missing or otherwise invalid data entries.

Your job is to use df.info() to determine information about the total count of non-null entries and infer the total count of null entries, which likely indicates missing data.

Select the best description of this data set from the following:

**Possible Answers**

- [ ] The data is all of type float64 and none of it is missing.
- [ ] The data is of mixed type, and 9914 of it is missing.
- [x] The data is of mixed type, and 3460 float64s are missing.
- [ ] The data is all of type float64, and 3460 float64s are missing.
"""

# NumPy and pandas working together

import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/world_population.csv'
df = pd.read_csv(fname, index_col=0)

# Import numpy
import numpy as np

# Create array of DataFrame values: np_vals
np_vals = df.values

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.log10(np_vals)

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(df)

# Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'df', 'df_log10']]

# Building DataFrames from scratch

# Zip lists to build a DataFrame
list_keys = ['Country', 'Total']
list_values = [['United States', 'Soviet Union', 'United Kingdom'], [1118, 473, 273]]

# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys,list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)

# Labeling your data

list_keys = ['a', 'b', 'c', 'd']
list_values = [['1980', '1981', '1982'],
               ['Blondie', 'Christopher Cross', 'Joan Jett'], 
               ['Call Me', 'Arthur\'s Theme', 'I Love Rock and Roll'],
               [6, 3, 7]]
zipped = list(zip(list_keys,list_values))
data = dict(zipped)
df = pd.DataFrame(data)

# Build a list of labels: list_labels
list_labels = ['year','artist','song','chart weeks']

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels

# Building DataFrames with broadcasting

cities = ['Manheim', 'Preston park', 'Biglerville', 'Indiana', 'Curwensville', 
          'Crown', 'Harveys lake', 'Mineral springs', 'Cassville', 'Hannastown', 
          'Saltsburg', 'Tunkhannock', 'Pittsburgh', 'Lemasters', 'Great bend']

# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state':state, 'city':cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Importing & exporting data

data_file = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/world_population.csv'

# Read in the file: df1
df1 = pd.read_csv(data_file)

# Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv(data_file, header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)

# Delimiters, headers, and extensions

file_messy = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/messy_stock_data.tsv'

# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head())

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment="#")

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
file_clean = 'tmp_clean_stock_data.csv'
df2.to_csv(file_clean, index=False)

# Save the cleaned up DataFrame to an Excel file without the index
df2.to_excel('file_clean.xlsx', index=False)

# Plotting with pandas

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'
df = pd.read_csv(fname, usecols=['Temperature (deg F)'])

# Create a plot with color='red'
df.plot(color='red')

# Add a title
plt.title('Temperature in Austin')

# Specify the x-axis label
plt.xlabel('Hours since midnight August 1, 2010')

# Specify the y-axis label
plt.ylabel('Temperature (degrees F)')

# Display the plot
plt.show()

# Plotting DataFrames

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'
df = pd.read_csv(fname, usecols=['Temperature (deg F)', 'Dew Point (deg F)', 'Pressure (atm)'])

# Plot all columns (default)
df.plot()
plt.show()

# Plot all columns as subplots
df.plot(subplots=True)
plt.show()

# Plot just the Dew Point data
column_list1 = ['Dew Point (deg F)']
df[column_list1].plot()
plt.show()

# Plot the Dew Point and Temperature data, but not the Pressure data
column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot()
plt.show()

"""## 2. Exploratory data analysis"""

# Visual exploratory data analysis

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/clean_stock_data.csv'
df = pd.read_csv(fname)

# Create a list of y-axis column names: y_columns
y_columns = ['AAPL', 'IBM']

# Generate a line plot
df.plot(x='Month', y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()

# pandas scatter plots

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/auto-mpg.csv'
df = pd.read_csv(fname)

sizes = [ 51.12044694,  56.78387977,  49.15557238,  49.06977358,
        49.52823321,  78.4595872 ,  78.93021696,  77.41479205,
        81.52541106,  61.71459825,  52.85646225,  54.23007578,
        58.89427963,  39.65137852,  23.42587473,  33.41639502,
        32.03903011,  27.8650165 ,  18.88972581,  14.0196956 ,
        29.72619722,  24.58549713,  23.48516821,  20.77938954,
        29.19459189,  88.67676838,  79.72987328,  79.94866084,
        93.23005042,  18.88972581,  21.34122243,  20.6679223 ,
        28.88670381,  49.24144612,  46.14174741,  45.39631334,
        45.01218186,  73.76057586,  82.96880195,  71.84547684,
        69.85320595, 102.22421043,  93.78252358, 110.        ,
        36.52889673,  24.14234281,  44.84805372,  41.02504618,
        20.51976563,  18.765772  ,  17.9095202 ,  17.75442285,
        13.08832041,  10.83266174,  14.00441945,  15.91328975,
        21.60597587,  18.8188451 ,  21.15311208,  24.14234281,
        20.63083317,  76.05635059,  80.05816704,  71.18975117,
        70.98330444,  56.13992036,  89.36985382,  84.38736544,
        82.6716892 ,  81.4149056 ,  22.60363518,  63.06844313,
        69.92143863,  76.76982089,  69.2066568 ,  35.81711267,
        26.25184749,  36.94940537,  19.95069229,  23.88237331,
        21.79608472,  26.1474042 ,  19.49759118,  18.36136808,
        69.98970461,  56.13992036,  66.21810474,  68.02351436,
        59.39644014, 102.10046481,  82.96880195,  79.25686195,
        74.74521151,  93.34830013, 102.05923292,  60.7883734 ,
        40.55589449,  44.7388015 ,  36.11079464,  37.9986264 ,
        35.11233175,  15.83199594, 103.96451839, 100.21241654,
        90.18186347,  84.27493641,  32.38645967,  21.62494928,
        24.00218436,  23.56434276,  18.78345471,  22.21725537,
        25.44271071,  21.36007926,  69.37650986,  76.19877818,
        14.51292942,  19.38962134,  27.75740889,  34.24717407,
        48.10262495,  29.459795  ,  32.80584831,  55.89556844,
        40.06360581,  35.03982309,  46.33599903,  15.83199594,
        25.01226779,  14.03498009,  26.90404245,  59.52231336,
        54.92349014,  54.35035315,  71.39649768,  91.93424995,
        82.70879915,  89.56285636,  75.45251972,  20.50128352,
        16.04379287,  22.02531454,  11.32159874,  16.70430249,
        18.80114574,  18.50153068,  21.00322336,  25.79385418,
        23.80266582,  16.65430211,  44.35746794,  49.815853  ,
        49.04119063,  41.52318884,  90.72524338,  82.07906251,
        84.23747672,  90.29816462,  63.55551901,  63.23059357,
        57.92740995,  59.64831981,  38.45278922,  43.19643409,
        41.81296121,  19.62393488,  28.99647648,  35.35456858,
        27.97283229,  30.39744886,  20.57526193,  26.96758278,
        37.07354237,  15.62160631,  42.92863291,  30.21771564,
        36.40567571,  36.11079464,  29.70395123,  13.41514444,
        25.27829944,  20.51976563,  27.54281821,  21.17188565,
        20.18836167,  73.97101962,  73.09614831,  65.35749368,
        73.97101962,  43.51889468,  46.80945169,  37.77255674,
        39.6256851 ,  17.24230306,  19.49759118,  15.62160631,
        13.41514444,  55.49963323,  53.18333207,  55.31736854,
        42.44868923,  13.86730874,  16.48817545,  19.33574884,
        27.3931002 ,  41.31307817,  64.63368105,  44.52069676,
        35.74387954,  60.75655952,  79.87569835,  68.46177648,
        62.35745431,  58.70651902,  17.41217694,  19.33574884,
        13.86730874,  22.02531454,  15.75091031,  62.68013142,
        68.63071356,  71.36201911,  76.80558184,  51.58836621,
        48.84134317,  54.86301837,  51.73502816,  74.14661842,
        72.22648148,  77.88228247,  78.24284811,  15.67003285,
        31.25845963,  21.36007926,  31.60164234,  17.51450098,
        17.92679488,  16.40542438,  19.96892459,  32.99310928,
        28.14577056,  30.80379718,  16.40542438,  13.48998471,
        16.40542438,  17.84050478,  13.48998471,  47.1451025 ,
        58.08281541,  53.06435374,  52.02897659,  41.44433489,
        36.60292926,  30.80379718,  48.98404972,  42.90189859,
        47.56635225,  39.24128299,  54.56115914,  48.41447259,
        48.84134317,  49.41341845,  42.76835191,  69.30854366,
        19.33574884,  27.28640858,  22.02531454,  20.70504474,
        26.33555201,  31.37264569,  33.93740821,  24.08222494,
        33.34566004,  41.05118927,  32.52595611,  48.41447259,
        16.48817545,  18.97851406,  43.84255439,  37.22278157,
        34.77459916,  44.38465193,  47.00510227,  61.39441929,
        57.77221268,  65.12675249,  61.07507305,  79.14790534,
        68.42801405,  54.10993164,  64.63368105,  15.42864956,
        16.24054679,  15.26876826,  29.68171358,  51.88189829,
        63.32798377,  42.36896092,  48.6988448 ,  20.15170555,
        19.24612787,  16.98905358,  18.88972581,  29.68171358,
        28.03762169,  30.35246559,  27.20120517,  19.13885751,
        16.12562794,  18.71277385,  16.9722369 ,  29.85984799,
        34.29495526,  37.54716158,  47.59450219,  19.93246832,
        30.60028577,  26.90404245,  24.66650366,  21.36007926,
        18.5366546 ,  32.64243213,  18.5366546 ,  18.09999962,
        22.70075058,  36.23351603,  43.97776651,  14.24983724,
        19.15671509,  14.17291518,  35.25757392,  24.38356372,
        26.02234705,  21.83420642,  25.81458463,  28.90864169,
        28.58044785,  30.91715052,  23.6833544 ,  12.82391671,
        14.63757021,  12.89709155,  17.75442285,  16.24054679,
        17.49742615,  16.40542438,  20.42743834,  17.41217694,
        23.58415722,  19.96892459,  20.33531923,  22.99334585,
        28.47146626,  28.90864169,  43.43816712,  41.57579979,
        35.01567018,  35.74387954,  48.5565546 ,  57.77221268,
        38.98605581,  49.98882458,  28.25412762,  29.01845599,
        23.88237331,  27.60710798,  26.54539622,  31.14448175,
        34.17556473,  16.3228815 ,  17.0732619 ,  16.15842026,
        18.80114574,  18.80114574,  19.42557798,  20.2434083 ,
        20.98452475,  16.07650192,  16.07650192,  16.57113469,
        36.11079464,  37.84783835,  27.82194848,  33.46359332,
        29.5706502 ,  23.38638738,  36.23351603,  32.40968826,
        18.88972581,  21.92965639,  28.68963762,  30.80379718]

# Generate a scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)

# Add the title
plt.title('Fuel efficiency vs Horse-power')

# Add the x-axis label
plt.xlabel('Horse-power')

# Add the y-axis label
plt.ylabel('Fuel efficiency (mpg)')

# Display the plot
plt.show()

# pandas box plots

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/auto-mpg.csv'
df = pd.read_csv(fname)

# Make a list of the column names to be plotted: cols
cols = ['weight', 'mpg']

# Generate the box plots
df[cols].plot(kind='box', subplots=True)

# Display the plot
plt.show()

# pandas hist, pdf and cdf

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/tips.csv'
df = pd.read_csv(fname)

# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', density=True, bins=30, range=(0,.3))
# plt.show()

# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', density=True, cumulative=True, bins=30, range=(0,.3))
plt.show()

# Statistical exploratory data analysis

import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/auto-mpg.csv'
df = pd.read_csv(fname)

df['mpg'].median()

"""**Fuel efficiency**

From the automobiles data set, which value corresponds to the median value of the 'mpg' column? 

Your job is to select the 'mpg' column and call the .median() method on it. The automobile DataFrame has been provided as df.

Possible Answers
- [ ] 29.0
- [ ] 23.45
- [x] 22.75
- [ ] 32
"""

# Bachelor's degrees awarded to women

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/percent-bachelors-degrees-women-usa.csv'
df = pd.read_csv(fname)

# Print the minimum value of the Engineering column
print(df['Engineering'].min())

# Print the maximum value of the Engineering column
print(df['Engineering'].max())

# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')

# Plot the average percentage per year
mean.plot()

# Display the plot
plt.show()

# Median vs mean

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/titanic.csv'
df = pd.read_csv(fname)

# Print summary statistics of the fare column with .describe()
print(df.fare.describe())

# Generate a box plot of the fare column
df.fare.plot(kind='box')

# Show the plot
plt.show()

# Quantiles

import matplotlib.pyplot as plt
import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/life_expectancy_at_birth.csv'
df = pd.read_csv(fname)

# Print the number of countries reported in 2015
print(df['2015'].count())

# Print the 5th and 95th percentiles
print(df.quantile([0.05,0.95]))

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()

# Standard deviation of temperature

import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'
df = pd.read_csv(fname, usecols=['Date', 'Temperature (deg F)'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d %H:%M')

import datetime as dt
date_after = dt.datetime(2010, 1, 1)
date_before = dt.datetime(2010, 1, 31)
january = df[(df.Date >= date_after) & (df.Date <= date_before)]

date_after = dt.datetime(2010, 3, 1)
date_before = dt.datetime(2010, 3, 31)
march = df[(df.Date >= date_after) & (df.Date <= date_before)]


# Print the mean of the January and March data
print(january.mean(), march.mean())

# Print the standard deviation of the January and March data
print(january.std(), march.std())

# Separating populations

import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/auto-mpg.csv'
df = pd.read_csv(fname)

df[df['origin'] == 'Asia']
# df[df.origin == 'Asia']

"""**Filtering and counting**

How many automobiles were manufactured in Asia in the automobile dataset? 

The DataFrame has been provided for you as df. Use filtering and the .count() member method to determine the number of rows where the 'origin' column has the value 'Asia'.

As an example, you can extract the rows that contain 'US' as the country of origin using df[df['origin'] == 'US'].

**Possible Answers**

- [ ] 68
- [x] 79
- [ ] 245
- [ ] 392
"""

# Separate and summarize

import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/auto-mpg.csv'
df = pd.read_csv(fname)

# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df.loc[df['origin'] == 'US']

# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)

# Separate and plot

import pandas as pd
fname = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/titanic.csv'
titanic = pd.read_csv(fname)

# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)

# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')

# Display the plot
plt.show()

"""## 3. Time series in pandas"""

# Indexing time series

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'

df1 = pd.read_csv(filename)
df2 = pd.read_csv(filename, parse_dates=['Date'])
df3 = pd.read_csv(filename, index_col='Date', parse_dates=True)

df3.loc['2010-Aug-01']

"""**Reading and slicing times**

For this exercise, we have read in the same data file using three different approaches:

- `df1 = pd.read_csv(filename)`
- `df2 = pd.read_csv(filename, parse_dates=['Date'])`
- `df3 = pd.read_csv(filename, index_col='Date', parse_dates=True)`

Use the `.head()` and `.info()` methods in the IPython Shell to inspect the DataFrames. Then, try to index each DataFrame with a datetime string. Which of the resulting DataFrames allows you to easily index and slice data by dates using, for example, `df1.loc['2010-Aug-01']`?

**Possible Answers**

- [ ] df1.
- [ ] df1 and df2.
- [ ] df2.
- [ ] df2 and df3.
- [x] df3.
"""

# Creating and using a DatetimeIndex

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'
df = pd.read_csv(filename)
date_list = df['Date'].tolist()
temperature_list = df['Temperature (deg F)'].tolist()

# Prepare a format string: time_format
time_format='%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list, format=time_format)  

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)

# Partial string indexing and slicing

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'
df = pd.read_csv(filename)
date_list = df['Date'].tolist()

temperature_list = df['Temperature (deg F)'].tolist()
time_format='%Y-%m-%d %H:%M'

my_datetimes = pd.to_datetime(date_list, format=time_format)  
ts0 = pd.Series(temperature_list, index=my_datetimes)

# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']

# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc['2010-12-15':'2010-12-31']

# Reindexing the Index

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'
df = pd.read_csv(filename)
date_list = df['Date'].tolist()

temperature_list = df['Temperature (deg F)'].tolist()
time_format='%Y-%m-%d %H:%M'

my_datetimes = pd.to_datetime(date_list, format=time_format)  
ts0 = pd.Series(temperature_list, index=my_datetimes)
ts1 = ts0.loc['2010-07-01':'2010-07-17']
ts2 = ts0.loc['2010-07-01':'2010-07-15']

# Reindex without fill method: ts3
ts3 = ts2.reindex(ts1.index)

# Reindex with fill method, using forward fill: ts4
ts4 = ts2.reindex(ts1.index, method='ffill')

# Combine ts1 + ts2: sum12
sum12 = ts1 + ts2

# Combine ts1 + ts3: sum13
sum13 = ts1 + ts3

# Combine ts1 + ts4: sum14
sum14 = ts1 + ts4

# Resampling time series data

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'

df = pd.read_csv(filename, index_col='Date', parse_dates=True)
df.rename(columns = {'Temperature (deg F)':'Temperature', 
                     'Dew Point (deg F)':'DewPoint', 
                     'Pressure (atm)':'Pressure'}, inplace = True)

# Resampling and frequency

# Downsample to 6 hour data and aggregate by mean: df1
df1 = df['Temperature'].resample('6h').mean()

# Downsample to daily data and count the number of data points: df2
df2 = df['Temperature'].resample('D').count()

# Separating and resampling

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'

df = pd.read_csv(filename, index_col='Date', parse_dates=True)
df.rename(columns = {'Temperature (deg F)':'Temperature', 
                     'Dew Point (deg F)':'DewPoint', 
                     'Pressure (atm)':'Pressure'}, inplace = True)

# Extract temperature data for August: august
august = df['Temperature']['2010-August']

# Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample('D').max()

# Extract temperature data for February: february
february = df['Temperature']['2010-Feb']

# Downsample to obtain the daily lowest temperatures in February: february_lows
february_lows = february.resample('D').min()

# Rolling mean and frequency

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'

df = pd.read_csv(filename, index_col='Date', parse_dates=True)
df.rename(columns = {'Temperature (deg F)':'Temperature', 
                     'Dew Point (deg F)':'DewPoint', 
                     'Pressure (atm)':'Pressure'}, inplace = True)

# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']['2010-Aug-01':'2010-Aug-15']

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(window=24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()

# Resample and roll with it

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'

df = pd.read_csv(filename, index_col='Date', parse_dates=True)
df.rename(columns = {'Temperature (deg F)':'Temperature', 
                     'Dew Point (deg F)':'DewPoint', 
                     'Pressure (atm)':'Pressure'}, inplace = True)

# Extract the August 2010 data: august
august = df['Temperature']['2010-Aug']

# Resample to daily data, aggregating by max: daily_highs
daily_highs = august.resample('D').max()

# Use a rolling 7-day window with method chaining to smooth the daily high temperatures in August
daily_highs_smoothed = august.resample('D').max().rolling(window=7).mean()
print(daily_highs_smoothed)

# Manipulating time series data

# Method chaining and filtering

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/austin_airport_departure_data_2015_july.csv'
df = pd.read_csv(filename, header=10, index_col='Date (MM/DD/YYYY)', parse_dates=True)

# Strip extra whitespace from the column names: df.columns
df.columns = df.columns.str.strip()

# Extract data for which the destination airport is Dallas: dallas
dallas = df['Destination Airport'].str.contains('DAL')

# Compute the total number of Dallas departures each day: daily_departures
daily_departures = dallas.resample('D').sum()

# Generate the summary statistics for daily Dallas departures: stats
stats = daily_departures.describe()

# Missing values and interpolation

import pandas as pd
import numpy as np
import datetime as dt

data = np.array(range(0, 17))
start_date = dt.datetime(2016, 7, 1)
index = pd.date_range(start_date, periods=17)
ts1 = pd.Series(data, index)

data = np.array(range(0, 11))
index1 = pd.date_range(start='2016-07-01', periods=1)
index2 = pd.date_range(start='2016-07-04', end='2016-07-08')
index3 = pd.date_range(start='2016-07-11', end='2016-07-15')
index = index1.append(index2).append(index3)
ts2 = pd.Series(data, index)

# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')

# Compute the absolute difference of ts1 and ts2_interp: differences 
differences = np.abs(ts1 - ts2_interp)

# Generate and print summary statistics of the differences
print(differences.describe())

# Time zones and conversion

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/austin_airport_departure_data_2015_july.csv'
df = pd.read_csv(filename, header=10, parse_dates=True)
df.columns = df.columns.str.strip()

# Build a Boolean mask to filter for the 'LAX' departure flights: mask
mask = df['Destination Airport'] == 'LAX'

# Use the mask to subset the data: la
la = df[mask]

# Combine two columns of data to create a datetime series: times_tz_none 
times_tz_none = pd.to_datetime( la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'] )

# Localize the time to US/Central: times_tz_central
times_tz_central = times_tz_none.dt.tz_localize('US/Central')

# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')

# Time series visualization

# Plotting time series, datetime indexing

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'

df = pd.read_csv(filename, parse_dates=True, usecols=['Temperature (deg F)', 'Date'])
df.rename(columns = {'Temperature (deg F)':'Temperature'}, inplace = True)

# Plot the raw data before setting the datetime index
df.plot()
plt.show()

# Convert the 'Date' column into a collection of datetime objects: df.Date
df.Date = pd.to_datetime(df.Date)

# Set the index to be the converted 'Date' column
df.set_index('Date', inplace=True)

# Re-plot the DataFrame to see that the axis is now datetime aware!
df.plot()
plt.show()

# Plotting date ranges, partial indexing

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'

df = pd.read_csv(filename, parse_dates=True, index_col='Date')
df.rename(columns = {'Temperature (deg F)':'Temperature', 
                     'Dew Point (deg F)':'DewPoint', 
                     'Pressure (atm)':'Pressure'}, inplace = True)

# Plot the summer data
df.Temperature['2010-Jun':'2010-Aug'].plot()
plt.show()
plt.clf()

# Plot the one week data
df.Temperature['2010-06-10':'2010-06-17'].plot()
plt.show()
plt.clf()

"""## 4. Case Study - Sunlight in Austin

**What method should we use to read the data?**

The first step in our analysis is to read in the data. Upon inspection with a certain system tool, we find that the data appears to be ASCII encoded with comma delimited columns, but has no header and no column labels. 

Which of the following is the best method to start with to read the data files?

**Possible Answers**

- [x] pd.read_csv()
- [ ] pd.to_csv()
- [ ] pd.read_hdf()
- [ ] np.load()
"""

# Reading in a data file

# Import pandas
import pandas as pd

# Read in the data file: df
data_file = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/NOAA_QCLCD_2011_hourly_13904.txt'
df = pd.read_csv(data_file)

# Print the output of df.head()
print(df.head())

# Read in the data file with header=None: df_headers
df_headers = pd.read_csv(data_file, header=None)

# Print the output of df_headers.head()
print(df_headers.head())

# Re-assigning column names

import pandas as pd

# Read in the data file: df
data_file = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/NOAA_QCLCD_2011_hourly_13904.txt'
df = pd.read_csv(data_file, header=None)

# Split on the comma to create a list: 
column_labels = 'Wban,date,Time,StationType,sky_condition,sky_conditionFlag,visibility,visibilityFlag,wx_and_obst_to_vision,wx_and_obst_to_visionFlag,dry_bulb_faren,dry_bulb_farenFlag,dry_bulb_cel,dry_bulb_celFlag,wet_bulb_faren,wet_bulb_farenFlag,wet_bulb_cel,wet_bulb_celFlag,dew_point_faren,dew_point_farenFlag,dew_point_cel,dew_point_celFlag,relative_humidity,relative_humidityFlag,wind_speed,wind_speedFlag,wind_direction,wind_directionFlag,value_for_wind_character,value_for_wind_characterFlag,station_pressure,station_pressureFlag,pressure_tendency,pressure_tendencyFlag,presschange,presschangeFlag,sea_level_pressure,sea_level_pressureFlag,record_type,hourly_precip,hourly_precipFlag,altimeter,altimeterFlag,junk'
column_labels_list = column_labels.split(",")

# Assign the new column labels to the DataFrame: df.columns
df.columns = column_labels_list

# Remove the appropriate columns: df_dropped
list_to_drop = ['sky_conditionFlag',
 'visibilityFlag',
 'wx_and_obst_to_vision',
 'wx_and_obst_to_visionFlag',
 'dry_bulb_farenFlag',
 'dry_bulb_celFlag',
 'wet_bulb_farenFlag',
 'wet_bulb_celFlag',
 'dew_point_farenFlag',
 'dew_point_celFlag',
 'relative_humidityFlag',
 'wind_speedFlag',
 'wind_directionFlag',
 'value_for_wind_character',
 'value_for_wind_characterFlag',
 'station_pressureFlag',
 'pressure_tendencyFlag',
 'pressure_tendency',
 'presschange',
 'presschangeFlag',
 'sea_level_pressureFlag',
 'hourly_precip',
 'hourly_precipFlag',
 'altimeter',
 'record_type',
 'altimeterFlag',
 'junk']
df_dropped = df.drop(list_to_drop, axis='columns')

# Print the output of df_dropped.head()
print(df_dropped.head())

# Cleaning and tidying datetime data

# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped['date'].astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Concatenate the new date and Time columns: date_string
date_string = df_dropped['date'] + df_dropped['Time']

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
print(df_clean.head())

# Cleaning the numeric columns

# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-6-20 8:00:00':'2011-6-20 9:00:00', 'dry_bulb_faren'])

# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')

# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-6-20 8:00:00':'2011-6-20 9:00:00', 'dry_bulb_faren'])

# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')

# Statistical exploratory data analysis

# Print the median of the dry_bulb_faren column
print(df_clean['dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the time range '2011-Apr':'2011-Jun'
print(df_clean.loc['2011-Apr':'2011-Jun', 'dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the month of January
print(df_clean.loc['2011-Jan', 'dry_bulb_faren'].median())

# Signal variance

import pandas as pd
filename = 'https://raw.githubusercontent.com/chesterheng/ai-for-industry/main/datasets/weather_data_austin_2010.csv'
df.columns = df.columns.str.strip()

df_climate = pd.read_csv(filename, parse_dates=True, index_col='Date')
df_climate.rename(columns = {'Temperature (deg F)':'Temperature', 
                     'Dew Point (deg F)':'DewPoint', 
                     'Pressure (atm)':'Pressure'}, inplace = True)

# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())

# Sunny or cloudy

# Using df_clean, when is sky_condition 'CLR'?
is_sky_clear = df_clean['sky_condition']=='CLR'

# Filter df_clean using is_sky_clear
sunny = df_clean.loc[is_sky_clear]

# Resample sunny by day then calculate the max
# sunny_daily_max = sunny.resample('D').max()
sunny_daily_max = sunny.resample('D').agg(['max'])

# See the result
sunny_daily_max.head()

# Using df_clean, when does sky_condition contain 'OVC'?
is_sky_overcast = df_clean['sky_condition'].str.contains('OVC')

# Filter df_clean using is_sky_overcast
overcast = df_clean.loc[is_sky_overcast]

# Resample overcast by day then calculate the max
# overcast_daily_max = overcast.resample('D').max()
overcast_daily_max = overcast.resample('D').agg(['max'])

# See the result
overcast_daily_max.head()

# Calculate the mean of sunny_daily_max
sunny_daily_max_mean = sunny_daily_max.mean()

# Calculate the mean of overcast_daily_max
overcast_daily_max_mean = overcast_daily_max.mean()

# Print the difference (sunny minus overcast)
print(sunny_daily_max_mean - overcast_daily_max_mean)

# Visual exploratory data analysis

# Weekly average temperature and visibility

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

df_clean['visibility'] = pd.to_numeric(df_clean['visibility'], errors='coerce')

# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
weekly_mean = df_clean[['visibility','dry_bulb_faren']].resample('W').mean()

# Print the output of weekly_mean.corr()
print(weekly_mean.corr())

# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots=True)
plt.show()

# Daily hours of clear sky

# Using df_clean, when is sky_condition 'CLR'?
is_sky_clear = df_clean['sky_condition'] == 'CLR'

# Resample is_sky_clear by day
resampled = is_sky_clear.resample('D')

# Calculate the number of sunny hours per day
sunny_hours = resampled.sum()

# Calculate the number of measured hours per day
total_hours = resampled.count()

# Calculate the fraction of hours per day that were sunny
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
sunny_fraction.plot(kind='box')
plt.show()

# Heat or humidity

# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
monthly_max = df_clean[['dew_point_faren','dry_bulb_faren']].resample('M').max()

# Generate a histogram with bins=8, alpha=0.5, subplots=True
monthly_max.plot(kind='hist', bins=8, alpha=0.5, subplots=True)

# Show the plot
plt.show()

# Probability of high temperatures

# Extract the maximum temperature in August 2010 from df_climate: august_max
august_max = df_climate.loc['2010-Aug','Temperature'].max()
print(august_max)

# Resample August 2011 temps in df_clean by day & aggregate the max value: august_2011
august_2011 = df_clean.loc['2011-Aug','dry_bulb_faren'].resample('D').max()

# Filter for days in august_2011 where the value exceeds august_max: august_2011_high
august_2011_high = august_2011.loc[august_2011 > august_max]

# Construct a CDF of august_2011_high
# august_2011_high.plot(kind='hist', normed=True, cumulative=True, bins=25)
august_2011_high.plot(kind='hist', density=True, cumulative=True, bins=25)

# Display the plot
plt.show()