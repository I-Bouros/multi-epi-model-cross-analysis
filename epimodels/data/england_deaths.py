#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""Processing script for deaths data from [1]_.

It computes region-specific the age-structured daily number of deaths which
are the stored in separate csv files.

References
----------
.. [1] Data.gov.uk. Coronavirus (COVID-19) in the UK , GOV.UK,
       https://coronavirus.data.gov.uk/details/download.

"""

import datetime
import os
import pandas as pd
import numpy as np


def read_death_data(
        death_file='England_deaths.csv'):
    """
    Parses the csv document containing the age structured regional
    daily death data.

    Parameters
    ----------
    death_file
        The name of the age structured regional death data file used.

    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'death_data/')
    data = pd.read_csv(
        os.path.join(path, death_file))

    return data


def process_death_data(
        data,
        start_date='2020-02-15',
        end_date='2022-01-28'):
    """
    Computes the matrix of age-structured number of deaths for a given region.

    Parameters
    ----------
    data
        (pandas.Dataframe) Dataframe of age-structured daily number of deaths
        in a given region.
    start_date
        The initial date from which the number of deaths are calculated.
    end_date
        The final date from which the number of deaths are calculated.

    """
    data = data.sort_values('date')

    # Keep only those values within the date range
    data = data[data['date'] >= start_date]
    data = data[data['date'] <= end_date]

    # Keep only ages we want
    data = data[~data['age'].isin(['00_59', '60+'])]

    # Add a column 'Time', which is the number of days from start_date
    start = datetime.date(*map(int, start_date.split('-')))
    data['time'] = [(datetime.date(*map(int, x.split('-'))) - start).days + 1
                    for x in data['date']]

    # Keep only those columns we need
    data = data[
        [
            'region', 'date', 'time', 'age', 'deaths'
        ]]

    age_groups = [
        '0-1', '1-5', '5-15', '15-25', '25-45', '45-65', '65-75', '75+']
    deaths = pd.DataFrame(columns=age_groups)
    for t in data['time'].unique():
        daily_data = data[data['time'] == t]
        newrow = process_ages(age_groups, daily_data)

        deaths = deaths.append(newrow, ignore_index=True)

    return deaths.to_numpy()


def process_ages(age_groups, data):
    """
    Parses daily data into the correct age structure types.

    Parameters
    ----------
    age_groups
        List of the names for the age groups the data is split into.
    data
        (pandas.Dataframe) Dataframe of age-structured daily number of deaths
        in a given region.

    """
    newrow = {}
    # Process 0-1
    newrow[age_groups[0]] = 0

    # Process 1-5
    newrow[age_groups[1]] = data[data['age'].isin(['00_04'])]['deaths'].sum()

    # Process 5-15
    newrow[age_groups[2]] = data[data['age'].isin(
        ['05_09', '10_15'])]['deaths'].sum()

    # Process 15-25
    newrow[age_groups[3]] = data[data['age'].isin(
        ['15_19', '20_24'])]['deaths'].sum()

    # Process 25-45
    newrow[age_groups[4]] = data[data['age'].isin(
        ['25_29', '30_34', '35_39', '40_44'])]['deaths'].sum()

    # Process 45-65
    newrow[age_groups[5]] = data[data['age'].isin(
        ['45_49', '50_54', '55_59', '60_64'])]['deaths'].sum()

    # Process 65-75
    newrow[age_groups[6]] = data[data['age'].isin(
        ['65_69', '70_74'])]['deaths'].sum()

    # Process 75+
    newrow[age_groups[7]] = data[data['age'].isin(
        ['75_79', '80_84', '85_89', '90+'])]['deaths'].sum()

    return newrow


def process_regions(region):
    """
    Processes regions into standard `epimodels` format.

    """
    if region == 'South West':
        return 'SW'
    elif region == 'London':
        return 'London'
    elif region == 'South East':
        return 'SE'
    elif region == 'East of England':
        return 'EE'
    elif region in ['East Midlands', 'West Midlands']:
        return 'Mid'
    elif region in ['North East', 'Yorkshire and The Humber']:
        return 'NE'
    else:
        return 'NW'


def main():
    """
    Computes the matrix of age-structured number of deaths for all regions.

    """
    data = read_death_data()

    # Rename the columns of interest
    data = data.rename(columns={'areaName': 'region'})

    data['region'] = [process_regions(x) for x in data['region']]

    all_regions = ['EE', 'London', 'Mid', 'NE', 'NW', 'SE', 'SW']

    for region in all_regions:
        deaths = process_death_data(
            data[data['region'] == region],
            start_date='2020-03-12',
            end_date='2021-06-25')

        # Transform recorded deaths to csv file
        path_ = os.path.join(
            os.path.dirname(__file__), 'death_data/')
        path = os.path.join(
                path_,
                '{}_deaths.csv'.format(region))

        np.savetxt(path, deaths, fmt="%d", delimiter=',')


if __name__ == '__main__':
    main()
