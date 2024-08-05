## London Bike Sharing Project

### Overview

This project involves analyzing bike-sharing data from London. The goal is to explore, clean, and visualize the dataset to uncover insights about bike-sharing patterns. The final cleaned dataset is used to create visualizations in Tableau.

### Requirements

1. **Data Handling:**
   - Read the dataset from the provided CSV file (`london_merged.csv`).
   - Perform data cleaning and preprocessing using Pandas, including renaming columns, adjusting data types, and mapping categorical values.

2. **Data Cleaning and Preprocessing:**
   - Rename columns for clarity.
   - Convert humidity values to percentages.
   - Map numerical values to categorical labels (seasons and weather codes).

3. **Exploration and Analysis:**
   - Print the shape and the first few rows of the dataframe to understand its structure.
   - Display counts of unique values for relevant columns (e.g., weather codes, seasons).

4. **Output:**
   - Save the cleaned dataframe to an Excel file (`london_bikes_final.xlsx`) for use in Tableau.

5. **Visualization:**
   - Import the final Excel file into Tableau to create visualizations that reveal patterns and insights in the bike-sharing data.

### File Structure

- `README.md`: Provides an overview of the project, outlines the requirements, and explains how to execute the data processing and visualization tasks.
- `requirements.txt`: Lists dependencies required for the Python data processing tasks.
- `data/`: Directory containing the dataset file `london_merged.csv`.
- `src/`: Directory containing the Python script for data processing (`data_processing.py`).
- `london_bikes_final.xlsx`: Final cleaned dataset prepared for Tableau visualization.

### Goals

- **Data Collection**: Acquire the bike-sharing dataset.
- **Data Exploration and Manipulation**: Use Pandas to clean and preprocess the data.
- **Visualization**: Create insightful visualizations in Tableau based on the cleaned data.

### Final Visualization

You can view the final visualizations created in Tableau Public using the following link:

[London Bike Sharing Dashboard](https://public.tableau.com/app/profile/paige.leeseberg/viz/LondonBikeRides_17223759852780/Dashboard1)

### Dependencies

To run the data processing script, make sure you have the following Python packages installed:

- `pandas==2.1.0`
- `openpyxl==3.1.2`

You can install these dependencies using:

```bash
pip install -r requirements.txt
