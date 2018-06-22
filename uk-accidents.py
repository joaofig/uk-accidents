import numpy as np
import pandas as pd


def load_data():
    """Loads the CSV files and appends them into a single DataFrame"""
    column_types = {'Accident_Index': np.string_, 'LSOA_of_Accident_Location': np.string_}
    uk2015 = pd.read_csv("data/DfTRoadSafety_Accidents_2015.csv", dtype=column_types)
    uk2016 = pd.read_csv("data/dftRoadSafety_Accidents_2016.csv", dtype=column_types)
    return uk2015.append(uk2016)


def analyze_uk_accidents():
    uk_accidents = load_data()
    print(uk_accidents.describe())


if __name__ == "__main__":
    analyze_uk_accidents()