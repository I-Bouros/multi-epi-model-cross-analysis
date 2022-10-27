#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""Processing script for the contact matrices and Google mobility data from
[1]_ and [2]_.

It computes the baseline and time-dependent region-specific contact matrices
which are then stored in separate csv files.

References
----------
.. [1] Prem K, Cook AR, Jit M (2017) Projecting social contact matrices in 152
       countries using contact surveys and demographic data. PLOS Computational
       Biology 13(9): e1005697.
       https://doi.org/10.1371/journal.pcbi.1005697

.. [2] COVID-19 Community Mobility Reports
       https://www.google.com/covid19/mobility/
"""

import time
from time import mktime
import datetime
import os
import pandas as pd
import numpy as np


def read_contact_matrices(
        file_index: int = 2,
        state: str = 'United Kingdom of Great Britain'):
    """
    Read the baseline contact matices for different activities recorded
    for the given state from the appropriate Excel file.

    Parameters
    ----------
    file_index : int
        Index of the file containg the baseline contact matrices
        used in the model.
    state : str
        Name of the country whose the baseline contact matrices are used in
        the model.

    Retruns
    -------
    list of pandas.Dataframe
        List of the baseline contact matices for each activitiy recorded
        for different for the given state.

    """
    # Select contact matrices from the given state and activity
    path = os.path.join(
            os.path.dirname(__file__), 'raw_contact_matrices/')
    school = pd.read_excel(
        os.path.join(path, 'MUestimates_school_{}.xlsx').format(file_index),
        sheet_name=state, header=None).to_numpy()
    home = pd.read_excel(
        os.path.join(path, 'MUestimates_home_{}.xlsx').format(file_index),
        sheet_name=state, header=None).to_numpy()
    work = pd.read_excel(
        os.path.join(path, 'MUestimates_work_{}.xlsx').format(file_index),
        sheet_name=state, header=None).to_numpy()
    others = pd.read_excel(
        os.path.join(path, 'MUestimates_other_locations_{}.xlsx').format(
            file_index),
        sheet_name=state, header=None).to_numpy()

    return school, home, work, others


def compute_contact_matrices(
        region: str,
        start_date: str = '15/02/2020',
        end_date: str = '04/04/2021',
        mobility_file: str = '2020_2021_GB_Region_Mobility_Report.csv'):
    """
    Computes timelines of percentages of deviation from the baseline levels in
    the activities recorded, using Google mobility data, for the selected
    region and between given dates.

    Parameters
    ----------
    region : str
        Region of the country for which the deviation percentages are
        calculated.
    start_date : str
        Initial date (day/month/year) from which the deviation percentages are
        calculated.
    end_date : str
        Final date (day/month/year) from which the deviation percentages are
        calculated.
    mobility_file : str
        Name of the Google mobility data file used for the computation.

    Returns
    -------
    pandas.Dataframe
        Dataframe of the daily multipliers for the given region for each of the
        activities recorded.

    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'google_mobility/')
    data = pd.read_csv(
        os.path.join(path, mobility_file))

    # Keep only data in correct region
    data = data[data['uk_region'] == region]

    # Rename the columns of interest
    data = data.rename(columns={
        'retail_and_recreation_percent_change_from_baseline': 'shop',
        'grocery_and_pharmacy_percent_change_from_baseline': 'grocery',
        'parks_percent_change_from_baseline': 'parks',
        'transit_stations_percent_change_from_baseline': 'transit',
        'workplaces_percent_change_from_baseline': 'work',
        'residential_percent_change_from_baseline': 'home'})

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
            'uk_region', 'sub_region_1', 'population', 'sub_region_2', 'date',
            'time', 'shop', 'grocery', 'parks', 'transit', 'work', 'home'
        ]]

    activities = ['shop', 'grocery', 'parks', 'transit', 'work', 'home']
    multipliers = pd.DataFrame(columns=activities)
    for t in data['time'].unique():
        daily_data = data[data['time'] == t]
        daily_data = daily_data[daily_data['sub_region_2'].isna()]
        newrow = {}
        for a in activities:
            daily_data['{}_subtotal'.format(a)] = daily_data[
                'population'] * daily_data[a]

            newrow[a] = daily_data['{}_subtotal'.format(
                a)].sum() / daily_data['population'].sum()

        multipliers = multipliers.append(newrow, ignore_index=True)

    return multipliers


def process_dates(date: str):
    """
    Processes dates into `datetime` format.

    Parameters
    ----------
    date : str
        Date (day/month/year) as it appears in the data frame.

    Returns
    -------
    datetime.datetime
        Date processed into correct format.

    """
    struct = time.strptime(date, '%d/%m/%Y')
    return datetime.datetime.fromtimestamp(mktime(struct))


def change_age_groups(matrix: np.array):
    """
    Reprocess the contact matrix so that it has the appropriate age groups.

    Parameters
    ----------
    matrix : numpy.array
        Contact matrix with old age groups.

    Returns
    -------
    numpy.array
        New contact matrix with correct age groups.

    """
    new_matrix = np.empty((8, 8))

    ind_old = [
        np.array([0]),
        np.array([0]),
        np.array(range(1, 3)),
        np.array(range(3, 5)),
        np.array(range(5, 9)),
        np.array(range(9, 13)),
        np.array(range(13, 15)),
        np.array([15])]

    for i in range(8):
        for j in range(8):
            new_matrix[i, j] = np.mean(
                matrix[ind_old[i][:, None], ind_old[j]][:, None])

    return new_matrix


def main():
    """
    Combines the timelines of deviation percentages and baseline
    activity-specific contact matrices to get weekly, region-specific
    contact matrices.

    Returns
    -------
    csv
        Processed files for the baseline and region-specific time-dependent
        contact matrices for each different region found in the default file.

    """
    activity = ['school', 'home', 'work', 'others']
    baseline_matrices = read_contact_matrices()
    baseline_contact_matrix = np.zeros_like(baseline_matrices[0])
    for ind, a in enumerate(activity):
        baseline_contact_matrix += baseline_matrices[ind]

    all_regions = ['EE', 'London', 'Mid', 'NE', 'NW', 'SE', 'SW']

    for region in all_regions:
        multipliers = compute_contact_matrices(
            region, start_date='15/02/2020', end_date='25/06/2021',)
        days = range(multipliers.shape[0])
        weeks = [days[x:x+7] for x in range(0, len(days), 7)]
        week_mean = pd.Series(
            np.zeros(6),
            index=['shop', 'grocery', 'parks', 'transit', 'work', 'home'])
        for w, week in enumerate(weeks):
            contact_matrix = np.zeros_like(baseline_matrices[0])
            house_contact_matrix = np.zeros_like(baseline_matrices[0])
            nonhouse_contact_matrix = np.zeros_like(baseline_matrices[0])
            to_replace = np.where(multipliers.iloc[week].mean().notna())[0]
            for _ in to_replace:
                week_mean[_] = multipliers.iloc[week].mean()[_]
            week_multi = week_mean/100 + 1
            act_week_multi = pd.Series(
                [
                    week_multi.get(key='work'),
                    week_multi.get(key='home'),
                    week_multi.get(key='work'),
                    week_multi.get(
                        key=['shop', 'grocery', 'parks', 'transit']).mean(),
                ])
            act_week_multi.index = activity
            for ind, a in enumerate(activity):
                if a == 'home':
                    house_contact_matrix += act_week_multi.get(
                        key=a) * baseline_matrices[ind]
                else:
                    nonhouse_contact_matrix += act_week_multi.get(
                        key=a) * baseline_matrices[ind]

                contact_matrix = house_contact_matrix + nonhouse_contact_matrix

            # Transform recorded matrix of serial intervals to csv file
            path_ = os.path.join(
                os.path.dirname(__file__), 'final_contact_matrices/')
            path = os.path.join(
                    path_,
                    '{}_W{}.csv'.format(region, w+1))
            path1 = os.path.join(
                    path_,
                    'house_{}_W{}.csv'.format(region, w+1))
            path2 = os.path.join(
                    path_,
                    'nonhouse_{}_W{}.csv'.format(region, w+1))
            path3 = os.path.join(path_, 'BASE.csv')

            np.savetxt(
                path, change_age_groups(contact_matrix),
                delimiter=',')
            np.savetxt(
                path1, change_age_groups(house_contact_matrix),
                delimiter=',')
            np.savetxt(
                path2, change_age_groups(nonhouse_contact_matrix),
                delimiter=',')
            np.savetxt(
                path3, change_age_groups(baseline_contact_matrix),
                delimiter=',')


if __name__ == '__main__':
    main()
