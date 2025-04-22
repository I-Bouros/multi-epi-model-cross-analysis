#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""Processing script for cases data from [1]_.

It computes the region-specific total daily number of cases which
are then stored in separate csv files.

References
----------
.. [1] Data.gov.uk. Coronavirus (COVID-19) in the UK , GOV.UK,
       https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download.

"""

import datetime
import os
import pandas as pd
import numpy as np


def read_case_data(
        case_file: str = 'England_reported_cases.csv'):
    """
    Parses the csv document containing the regional total daily case data.

    Parameters
    ----------
    case_file : str
        The name of the regional case data file used.

    Returns
    -------
    pandas.Dataframe
        Dataframe of the total daily number of cases in all given
        regions.

    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'case_data/')
    data = pd.read_csv(
        os.path.join(path, case_file))

    return data


def process_case_data(
        data: pd.DataFrame,
        start_date: str = '2020-02-15',
        end_date: str = '2022-01-28'):
    """
    Computes the matrix of total number of cases for a given region.

    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe of the total daily number of cases
        in a given region.
    start_date : str
        The initial date (year-month-date) from which the number of cases are
        calculated.
    end_date : str
        The final date (year-month-date) from which the number of cases are
        calculated.

    Returns
    -------
    numpy.array
        Processed total daily number of cases in a given region as
        a matrix.

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
            'region', 'date', 'time', 'age', 'cases'
        ]]

    age_groups = [
        '0-1', '1-5', '5-15', '15-25', '25-45', '45-65', '65-75', '75+']
    lst_cases = []
    for t in data['time'].unique():
        daily_data = data[data['time'] == t]
        newrow = process_ages(age_groups, daily_data)
        lst_cases.append(newrow)

    cases = pd.DataFrame(lst_cases)

    return cases.to_numpy()


def process_ages(age_groups: list, data: pd.DataFrame):
    """
    Parses daily number of cases data into the correct age structure types.

    Parameters
    ----------
    age_groups : list
        List of the names of the age groups the cases data is split into.
    data : pandas.Dataframe
        Dataframe of the total daily number of cases in a given
        region.

    Returns
    -------
    pandas.Dataframe
        Processed dataframe row of the total daily number of cases
        in a given region.

    """
    newrow = {}

    # Process 0-1
    newrow[age_groups[0]] = 0

    # Process 1-5
    newrow[age_groups[1]] = data[data['age'].isin(['00_04'])]['cases'].sum()

    # Process 5-15
    newrow[age_groups[2]] = data[data['age'].isin(
        ['05_09', '10_15'])]['cases'].sum()

    # Process 15-25
    newrow[age_groups[3]] = data[data['age'].isin(
        ['15_19', '20_24'])]['cases'].sum()

    # Process 25-45
    newrow[age_groups[4]] = data[data['age'].isin(
        ['25_29', '30_34', '35_39', '40_44'])]['cases'].sum()

    # Process 45-65
    newrow[age_groups[5]] = data[data['age'].isin(
        ['45_49', '50_54', '55_59', '60_64'])]['cases'].sum()

    # Process 65-75
    newrow[age_groups[6]] = data[data['age'].isin(
        ['65_69', '70_74'])]['cases'].sum()

    # Process 75+
    newrow[age_groups[7]] = data[data['age'].isin(
        ['75_79', '80_84', '85_89', '90+'])]['cases'].sum()

    return newrow


def process_regions(region: str):
    """
    Processes region names into standard `epimodels` name format.

    Parameters
    ----------
    region : str
        Name of the region being processed.

    Returns
    -------
    str
        Name of the region in the standard `epimodels` name format.

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
    Computes the matrix of the total daily number of cases for
    all regions.

    Returns
    -------
    csv
        Processed cases data files for each different region found in the
        default file.

    """
    data = read_case_data()

    # Rename the columns of interest
    data = data.rename(columns={'areaName': 'region'})

    data['region'] = [process_regions(x) for x in data['region']]

    all_regions = ['EE', 'London', 'Mid', 'NE', 'NW', 'SE', 'SW']

    for region in all_regions:
        cases = process_case_data(
            data[data['region'] == region],
            start_date='2020-03-12',
            end_date='2021-06-25')

        # Transform recorded cases to csv file
        path_ = os.path.join(
            os.path.dirname(__file__), 'case_data/')
        path = os.path.join(
                path_,
                '{}_cases.csv'.format(region))

        np.savetxt(path, cases, fmt="%d", delimiter=',')


if __name__ == '__main__':
    main()
