# import the pandas library
import pandas as pd
import os

# set the working directory to the project directory
os.chdir('/Users/paigeleeseberg/Downloads/Python-Projects/Project6-London_Bike_Sharing/src')

# specify the path to the CSV file
csv_file = '../data/london_merged.csv'

# read in the csv file as a pandas dataframe
bikes = pd.read_csv(csv_file)

# explore the data
bikes.info()

# get the shape of the dataframe
print("DataFrame Shape:")
print(bikes.shape)

# display the first few rows of the dataframe
print("First Few Rows of DataFrame:")
print(bikes.head())

# count the unique values in the weather_code column
weather_code_counts = bikes.weather_code.value_counts()
print("Weather Code Counts:")
print(weather_code_counts)

# count the unique values in the season column
season_counts = bikes.season.value_counts()
print("Season Counts:")
print(season_counts)

# specifying the column names that I want to use
new_cols_dict = {
    'timestamp': 'time',
    'cnt': 'count',
    't1': 'temp_real_C',
    't2': 'temp_feels_like_C',
    'hum': 'humidity_percent',
    'wind_speed': 'wind_speed_kph',
    'weather_code': 'weather',
    'is_holiday': 'is_holiday',
    'is_weekend': 'is_weekend',
    'season': 'season'
}

# renaming the columns to the specific column names
bikes.rename(new_cols_dict, axis=1, inplace=True)

# check the column names after renaming
print("Column Names After Renaming:")
print(bikes.columns)

# changing the humidity values to percentage
bikes.humidity_percent = bikes.humidity_percent / 100

# display the first few rows to check the changes in humidity
print("First Few Rows After Changing Humidity:")
print(bikes.head())

# creating a season dictionary so that we can map the integers 0-3 to the actual written values
season_dict = {
    '0.0': 'spring',
    '1.0': 'summer',
    '2.0': 'autumn',
    '3.0': 'winter'
}

# creating a weather dictionary so that we can map the integers to the actual written values
weather_dict = {
    '1.0': 'Clear',
    '2.0': 'Scattered Clouds',
    '3.0': 'Broken Clouds',
    '4.0': 'Cloudy',
    '7.0': 'Rain',
    '10.0': 'Rain with thunderstorm',
    '26.0': 'Snowfall'
}

# changing the season column data type to string
bikes.season = bikes.season.astype('str')
# print unique values before mapping
print("Unique Values in Season Column Before Mapping:")
print(bikes.season.unique())
# mapping the values 0-3 to the actual written seasons
bikes.season = bikes.season.map(season_dict)
# print unique values after mapping
print("Unique Values in Season Column After Mapping:")
print(bikes.season.unique())

# changing the weather column data type to string
bikes.weather = bikes.weather.astype('str')
# print unique values before mapping
print("Unique Values in Weather Column Before Mapping:")
print(bikes.weather.unique())
# mapping the values to the actual written weathers
bikes.weather = bikes.weather.map(weather_dict)
# print unique values after mapping
print("Unique Values in Weather Column After Mapping:")
print(bikes.weather.unique())

# display the first few rows to check the changes in season and weather
print("First Few Rows After Changing Season and Weather:")
print(bikes.head())


# writing the final dataframe to an Excel file that we will use in our Tableau visualization
bikes.to_excel('/Users/paigeleeseberg/Downloads/Python-Projects/Project6-London_Bike_Sharing/london_bikes_final.xlsx', sheet_name='Data', index=False)

print("Final DataFrame saved to 'london_bikes_final.xlsx'")
