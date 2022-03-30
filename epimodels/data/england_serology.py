#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""Processing script for serology data from [1]_.

It computes region-specific the age-structured weekly number of tests and
positive results from the REACT1 study data which are then stored in separate
csv files.

References
----------
.. [1] Nicholson et al. (2021). Improving local prevalence estimates of
       SARS-CoV-2 infections using a causal debiasing framework. In Nature
       Microbiology (Vol. 7, Issue 1, pp. 97–107).
       Springer Science and Business Media LLC.
       https://doi.org/10.1038/s41564-021-01029-0

"""

import datetime
import os
import pandas as pd
import numpy as np


def read_tests_data(tests_file):
    """
    Parses the csv document containing the age structured regional
    serology data.

    Parameters
    ----------
    tests_file : csv
        The name of the age structured regional serology data file used.

    Returns
    -------
    pandas.Dataframe
        Dataframe of age-structured daily number of tests and positive results
        in all given regions.

    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'serology_data/REACT_data/')
    data = pd.read_csv(
        os.path.join(path, tests_file))

    return data


def process_tests_data(
        data,
        start_date='2020-04-27',
        end_date='2020-06-01'):
    """
    Computes the matrix of age-structured number of tests and positive results
    for a given region.

    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe of age-structured daily number of serology
        in a given region.
    start_date : str
        The initial date from which the number of tests and positive results
        are calculated.
    end_date : str
        The final date from which the number of tests and positive results
        are calculated.

    Returns
    -------
    tuple of numpy.array
        Tuple of processed regional tests data and psotive results data as a
        matrix.

    """
    data = data.sort_values('date')

    # Keep only those values within the date range
    data = data[data['date'] >= start_date]
    data = data[data['date'] <= end_date]

    # Add a column 'Time', which is the number of days from start_date
    start = datetime.date(*map(int, start_date.split('-')))
    data['time'] = [(datetime.date(*map(int, x.split('-'))) - start).days + 1
                    for x in data['date']]

    # Keep only those columns we need
    data = data[
        [
            'region', 'date', 'time', 'age', 'positives', 'tests'
        ]]

    age_groups = [
        '0-1', '1-5', '5-15', '15-25', '25-45', '45-65', '65-75', '75+']
    positives = pd.DataFrame(columns=age_groups)
    tests = pd.DataFrame(columns=age_groups)
    for t in data['time'].unique():
        # Process positive results
        daily_data = data[data['time'] == t]
        newrow = process_ages(age_groups, daily_data, type='positives')

        positives = positives.append(newrow, ignore_index=True)

        # Process test numbers
        daily_data = data[data['time'] == t]
        newrow = process_ages(age_groups, daily_data, type='tests')

        tests = tests.append(newrow, ignore_index=True)

    return positives.to_numpy(), tests.to_numpy()


def process_ages(age_groups, data, type):
    """
    Parses daily data into the correct age structure types.

    Parameters
    ----------
    age_groups : list
        List of the names for the age groups the data is split into.
    data : pandas.Dataframe
        Dataframe of age-structured daily number of tests
        or positive results in a given region.
    type : str
        Column name for the data we want to parse for.

    Returns
    -------
    pandas.Dataframe
        Processed dataframe row.

    """
    newrow = {}
    # Process 0-1
    newrow[age_groups[0]] = 0

    # Process 1-5
    newrow[age_groups[1]] = 0

    # Process 5-15
    newrow[age_groups[2]] = \
        data[data['age'].isin(['5-12'])][type].sum() + \
        np.floor(0.5 * data[data['age'].isin(['13-17'])][type].sum())

    # Process 15-25
    newrow[age_groups[3]] = \
        data[data['age'].isin(['18-24'])][type].sum() + \
        np.ceil(0.5 * data[data['age'].isin(['13-17'])][type].sum())

    # Process 25-45
    newrow[age_groups[4]] = \
        data[data['age'].isin(['25-34', '35-44'])][type].sum()

    # Process 45-65
    newrow[age_groups[5]] = \
        data[data['age'].isin(['45-54', '55-64'])][type].sum()

    # Process 65-75
    newrow[age_groups[6]] = \
        np.floor(0.5 * data[data['age'].isin(['65+'])][type].sum())

    # Process 75+
    newrow[age_groups[7]] = \
        np.ceil(0.5 * data[data['age'].isin(['65+'])][type].sum())

    return newrow


def process_regions(region):
    """
    Processes regions into standard `epimodels` name format.

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
    elif region in ['East Midlands', 'West Midlands', 'Midlands']:
        return 'Mid'
    elif region in ['North East', 'Yorkshire and The Humber',
                    'North East and Yorkshire']:
        return 'NE'
    else:
        return 'NW'


def main(files):
    """
    Computes the matrix of age-structured number of tests and the matrix
    of age-structured number of positive results for all regions.

    Parameters
    ----------
    files : list
        List of file names from which to extract the data.

    Returns
    -------
    csv
        Processed death data files for each different region found in the
        fiven file.

    """

    data = pd.concat(
        [read_tests_data(file) for file in files],
        ignore_index=True)

    # Rename the columns of interest
    data = data.rename(columns={
        'nhs_region': 'region',
        'react_week_start_date': 'date',
        'age_group': 'age',
        'number_positive': 'positives',
        'number_samples': 'tests'})
    data['region'] = [process_regions(x) for x in data['region']]

    all_regions = ['EE', 'London', 'Mid', 'NE', 'NW', 'SE', 'SW']

    for region in all_regions:
        positives, tests = process_tests_data(
            data[data['region'] == region],
            start_date='2020-05-04',
            end_date='2021-07-12')

        # Transform recorded deaths to csv file
        path_ = os.path.join(
            os.path.dirname(__file__), 'serology_data/')
        path = os.path.join(
                path_,
                '{}_positives_{}.csv'.format(region, files[0][21:-6]))

        path1 = os.path.join(
                path_,
                '{}_tests_{}.csv'.format(region, files[0][21:-6]))

        np.savetxt(path, positives, fmt="%d", delimiter=',')
        np.savetxt(path1, tests, fmt="%d", delimiter=',')


if __name__ == '__main__':
    main(['Region_England_cases_nhs_{}.csv'.format(w) for w in range(1, 14)])
