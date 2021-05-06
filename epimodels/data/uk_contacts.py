#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
"""

import time
from time import mktime
import datetime
import os
import pandas as pd
import numpy as np


def read_contact_matrices(
        file_index=2,
        state='United Kingdom of Great Britain'):
    """Write a new csv file for 60-day-long serial intervals
    to be used for the analysis of the Australian data.

    Parameters
    ----------
    name
        Name given to the serial intervals file.

    """
    # Select contact matrices from the given state and activity
    path = os.path.join(
            os.path.dirname(__file__), 'raw_contact_matrices/')
    school = pd.read_excel(
        os.path.join(path, 'MUestimates_school_{}.xlsx').format(file_index),
        sheet_name=state).to_numpy()
    home = pd.read_excel(
        os.path.join(path, 'MUestimates_home_{}.xlsx').format(file_index),
        sheet_name=state).to_numpy()
    work = pd.read_excel(
        os.path.join(path, 'MUestimates_work_{}.xlsx').format(file_index),
        sheet_name=state).to_numpy()
    others = pd.read_excel(
        os.path.join(path, 'MUestimates_other_locations_{}.xlsx').format(
            file_index),
        sheet_name=state).to_numpy()

    return school, home, work, others


def compute_contact_matrices(
        region,
        start_date='01/01/2021',
        end_date='04/04/2021',
        mobility_file='2021_GB_Region_Mobility_Report.csv'):
    """
    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'google_mobility/')
    data = pd.read_csv(
        os.path.join(path, '2021_GB_Region_Mobility_Report.csv'))

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


def process_dates(date):
    struct = time.strptime(date, '%d/%m/%Y')
    return datetime.datetime.fromtimestamp(mktime(struct))


def main():
    activity = ['school', 'home', 'work', 'others']
    baseline_matrices = read_contact_matrices()
    all_regions = ['EE', 'London', 'Mid', 'NE', 'NW', 'SE', 'SW']

    for region in all_regions:
        multipliers = compute_contact_matrices(region)
        days = range(multipliers.shape[0])
        weeks = [days[x:x+7] for x in range(0, len(days), 7)]
        for w, week in enumerate(weeks):
            contact_matrix = np.zeros_like(baseline_matrices[0])
            week_multi = multipliers.iloc[week].mean()/100 + 1
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
                contact_matrix += act_week_multi.get(
                    key=a) * baseline_matrices[ind]
            # Transform recorded matrix of serial intervals to csv file
            path_ = os.path.join(
                os.path.dirname(__file__), 'final_contact_matrices/')
            path = os.path.join(
                    path_,
                    '{}_W{}.csv'.format(region, w+1))
            np.savetxt(path, contact_matrix, delimiter=',')


if __name__ == '__main__':
    main()
