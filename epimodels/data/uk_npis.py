#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""Processing script for the non-pharmaceutical interventions data from [1]_.

It computes the time-dependent country-specific levels of the different
preventive interventions applied which are then stored in separate csv files.

References
----------
.. [1] Thomas Hale, Noam Angrist, Rafael Goldszmidt, Beatriz Kira, Anna
       Petherick, Toby Phillips, Samuel Webster, Emily Cameron-Blake, Laura
       Hallas, Saptarshi Majumdar, and Helen Tatlow. A global panel database
       of pandemic policies (oxford COVID-19 government response tracker).
       Nature Human Behaviour, 5(4):529–538, March 2021.

"""

import time
from time import mktime
import datetime
import os
import pandas as pd
import numpy as np


def read_npis_data(npis_file: str):
    """
    Parses the csv document containing the daily country-specific levels of
    NPIs data.

    Parameters
    ----------
    npis_file : str
        The name of the country-specific NPIs data file used.

    Returns
    -------
    pandas.Dataframe
        Dataframe of the daily country-specific levels of NPIs.

    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'npi_data/raw_data/')
    data = pd.read_csv(
        os.path.join(path, npis_file))

    return data


def read_flags_data(flags_file: str):
    """
    Parses the csv document containing the daily country-specific flags for
    NPIs data.

    Parameters
    ----------
    flags_file : str
        The name of the country-specific flags for NPIs data file used.

    Returns
    -------
    pandas.Dataframe
        Dataframe of the daily country-specific flags for NPIs.

    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'npi_data/raw_data/')
    data = pd.read_csv(
        os.path.join(path, flags_file))

    return data


def process_npis_data(
        data: pd.DataFrame,
        start_date: str = '15Feb2020',
        end_date: str = '25Jun2020'):
    """
    Computes the changes in levels of NPIs for a given country for a given
    country and the times of changes.

    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe of the daily levels of NPIs for a given country for a given
        country.
    start_date : str
        The initial date (year-month-date) from which the levels of NPIs are
        calculated.
    end_date : str
        The final date (year-month-date) from which the levels of NPIs are
        calculated.

    Returns
    -------
    numpy.array
        Processed distinct levels of NPIs data as a matrix.
    numpy.array
        Paired time of changes in the levels of NPIs data as a matrix.

    """
    # Process dates
    data['processed-date'] = [process_dates(x) for x in data['date']]
    data = data.sort_values('processed-date')

    # Keep only those values within the date range
    data = data[data['processed-date'] >= start_date]
    data = data[data['processed-date'] <= end_date]

    # Add a column 'Time', which is the number of days from start_date
    start = process_dates(start_date)
    data['time'] = [(x - start).days + 1
                    for x in data['processed-date']]

    # Keep only those columns we need
    data = data[
        [
            'date', 'time', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
            'h1'
        ]]

    interventions = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'h1']
    times_npis = []
    npi_levels = pd.DataFrame(columns=interventions)
    small_data = data.sort_values('time').drop_duplicates(
        subset=interventions, keep='first')
    for _, t in small_data[interventions].iterrows():
        # Process npis
        new_npis = small_data[
            (small_data[interventions] == t.tolist()).all(1)][interventions]
        new_time = small_data[
            (small_data[interventions] == t.tolist()).all(1)]['time']

        new_npis = pd.DataFrame(new_npis, columns=interventions)

        npi_levels = pd.concat([npi_levels, new_npis], ignore_index=True)
        times_npis.append(new_time.tolist())

    return npi_levels.to_numpy(), np.array(times_npis)


def process_flags_data(
        data: pd.DataFrame,
        start_date: str = '15Feb20200',
        end_date: str = '25Jun2020'):
    """
    Computes the changes in flags for NPIs for a given country for a given
    country and the times of changes.

    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe of the daily flags for NPIs for a given country for a given
        country.
    start_date : str
        The initial date (year-month-date) from which the flags for NPIs are
        calculated.
    end_date : str
        The final date (year-month-date) from which the flags for NPIs are
        calculated.

    Returns
    -------
    numpy.array
        Processed distinct flags for NPIs data as a matrix.
    numpy.array
        Paired time of changes in the flags for NPIs data as a matrix.

    """
    # Fill NA with Os
    data = data.fillna(0)

    # Process dates
    data['processed-date'] = [process_dates(x) for x in data['date']]
    data = data.sort_values('processed-date')

    # Keep only those values within the date range
    data = data[data['processed-date'] >= start_date]
    data = data[data['processed-date'] <= end_date]

    # Add a column 'Time', which is the number of days from start_date
    start = process_dates(start_date)
    data['time'] = [(x - start).days + 1
                    for x in data['processed-date']]

    # Keep only those columns we need
    data = data[
        [
            'date', 'time', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
            'h1'
        ]]

    interventions = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'h1']
    times_flags = []
    npi_flags = pd.DataFrame(columns=interventions)
    small_data = data.sort_values('time').drop_duplicates(
        subset=interventions, keep='first')
    for _, t in small_data[interventions].iterrows():
        # Process flags
        new_flags = small_data[
            (small_data[interventions] == t.tolist()).all(1)][interventions]
        new_time = small_data[
            (small_data[interventions] == t.tolist()).all(1)]['time']

        new_flags = pd.DataFrame(new_flags, columns=interventions)

        npi_flags = pd.concat([npi_flags, new_flags], ignore_index=True)
        times_flags.append(new_time.tolist())

    return npi_flags.to_numpy(), np.array(times_flags)


def process_dates(date: str):
    """
    Processes dates into `datetime` format.

    Parameters
    ----------
    date : str
        Date (DDMonYYYY) as it appears in the data frame.

    Returns
    -------
    datetime.datetime
        Date processed into correct format.

    """
    struct = time.strptime(date, '%d%b%Y')
    return datetime.datetime.fromtimestamp(mktime(struct))


def main():
    """
    Produces files for the time-dependent levels of NPIs and their flags
    as well as the times these changes occur.

    Returns
    -------
    csv
        Processed files for the time-dependent levels of NPIs and their flags
        as well as the times these changes occur.

    """
    # Read npis and flags data
    npis_data = read_npis_data('uk_npis.csv')
    flags_data = read_flags_data('uk_flags.csv')

    # Process data
    npi_levels, times_npis = process_npis_data(npis_data,
                                               start_date='15Feb2020',
                                               end_date='25Jun2020')
    long_npi_levels, long_times_npis = process_npis_data(
        npis_data,
        start_date='15Feb2020',
        end_date='25Jun2021')

    npi_flags, times_flags = process_flags_data(flags_data,
                                                start_date='15Feb2020',
                                                end_date='25Jun2020')
    long_npi_flags, long_times_flags = process_flags_data(
        flags_data,
        start_date='15Feb2020',
        end_date='25Jun2021')

    # Transform recorded matrix of serial intervals to csv file
    path_ = os.path.join(
        os.path.dirname(__file__), 'npi_data/')

    np.savetxt(os.path.join(path_, 'uk_npis.csv'),
               npi_levels, delimiter=',', fmt='%i')
    np.savetxt(os.path.join(path_, 'times_npis.csv'),
               times_npis, delimiter=',', fmt='%i')

    np.savetxt(os.path.join(path_, 'long_uk_npis.csv'),
               long_npi_levels, delimiter=',', fmt='%i')
    np.savetxt(os.path.join(path_, 'long_times_npis.csv'),
               long_times_npis, delimiter=',', fmt='%i')

    np.savetxt(os.path.join(path_, 'uk_flags.csv'),
               npi_flags, delimiter=',', fmt='%i')
    np.savetxt(os.path.join(path_, 'times_flags.csv'),
               times_flags, delimiter=',', fmt='%i')

    np.savetxt(os.path.join(path_, 'long_uk_flags.csv'),
               long_npi_flags, delimiter=',', fmt='%i')
    np.savetxt(os.path.join(path_, 'long_times_flags.csv'),
               long_times_flags, delimiter=',', fmt='%i')


if __name__ == '__main__':
    main()
