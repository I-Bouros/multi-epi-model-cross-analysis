#
# This file is part of EPIMODELS
# (https://github.com/I-Bouros/multi-epi-model-cross-analysis.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""Processing script for the fatality ratio data from [1]_ .

It computes the age-dependent case fatality ratios (CRF) and infection
fatality ratios (IFR) which are then stored in separate csv files.

References
----------
.. [1] Verity R, Okell LC, Dorigatti I, Winskill P, Whittaker C, Imai N,
       Cuomo-Dannenburg G, Thompson H, Walker PGT, Fu H, Dighe A, Griffin JT,
       Baguelin M, Bhatia S, Boonyasiri A, Cori A, Cucunubá Z, FitzJohn R,
       Gaythorpe K, … Ferguson NM (2020). Estimates of the severity of
       coronavirus disease 2019: a model-based analysis. In The Lancet
       Infectious Diseases (Vol. 20, Issue 6, pp. 669–677). Elsevier BV.
       https://doi.org/10.1016/s1473-3099(20)30243-7

"""

import os
import pandas as pd
import numpy as np


def read_fatality_ratios_data(fr_file: str):
    """
    Parses the csv document containing the fatality ratios data.

    Parameters
    ----------
    fr_file : str
        The name of the fatality ratios data file used.

    Returns
    -------
    pandas.Dataframe
        Dataframe of the fatality ratios.

    """
    # Select data from the given state
    path = os.path.join(
            os.path.dirname(__file__), 'fatality_ratio_data/')
    data = pd.read_csv(
        os.path.join(path, fr_file))

    return data


def process_ages(age_groups: list, data: pd.DataFrame, type: str):
    """
    Parses wwekly data into the correct age structure types.

    Parameters
    ----------
    age_groups : list
        List of the names for the age groups the data is split into.
    data : pandas.Dataframe
        Dataframe of age-structured weekly number of tests
        or positive results in a given region.
    type : str
        Column name for the data we want to parse for.

    Returns
    -------
    pandas.Dataframe
        Processed dataframe row.

    """
    newcol = {}
    # Process 0-1
    newcol[age_groups[0]] = data[data['age'] == '0-9'][type].values

    # Process 1-5
    newcol[age_groups[1]] = data[data['age'] == '0-9'][type].values

    # Process 5-15
    newcol[age_groups[2]] = \
        0.5 * data[data['age'] == '0-9'][type].values + \
        0.5 * data[data['age'] == '10-19'][type].values

    # Process 15-25
    newcol[age_groups[3]] = \
        0.5 * data[data['age'].isin(['10-19'])][type].values + \
        0.5 * data[data['age'].isin(['20-29'])][type].values

    # Process 25-45
    newcol[age_groups[4]] = \
        0.25 * data[data['age'].isin(['20-29'])][type].values + \
        0.5 * data[data['age'].isin(['30-39'])][type].values + \
        0.25 * data[data['age'].isin(['40-49'])][type].values

    # Process 45-65
    newcol[age_groups[5]] = \
        0.25 * data[data['age'].isin(['40-49'])][type].values + \
        0.5 * data[data['age'].isin(['50-59'])][type].values + \
        0.25 * data[data['age'].isin(['60-69'])][type].values

    # Process 65-75
    newcol[age_groups[6]] = \
        0.5 * data[data['age'].isin(['60-69'])][type].values + \
        0.5 * data[data['age'].isin(['70-79'])][type].values

    # Process 75+
    newcol[age_groups[7]] = \
        0.5 * data[data['age'].isin(['70-79'])][type].values + \
        0.5 * data[data['age'].isin(['80+'])][type].values

    return newcol


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
    data = read_fatality_ratios_data('verity_data.csv')

    # Rename the columns of interest
    data = data.rename(columns={
        'Age group': 'age',
        'CFR Adjusted for censoring, demography, and under-ascertainment':
            'cfr',
        'IFR': 'ifr'})

    # Keep only columns of interest
    data = data[['age', 'cfr', 'ifr']]

    # Process values of the type
    data = data.apply(lambda x: [np.float64(x_value.split('%')[0]) / 100
                      for x_value in x] if x.name in ['cfr', 'ifr'] else x)

    age_groups = [
        '0-1', '1-5', '5-15', '15-25', '25-45', '45-65', '65-75', '75+']

    # Process cfr results
    newcol = process_ages(age_groups, data, type='cfr')
    cfr_values = pd.DataFrame.from_dict(
        newcol, orient='index', columns=['cfr'])
    cfr_values.index.name = 'age'

    # Process ifr results
    newcol = process_ages(age_groups, data, type='ifr')
    ifr_values = pd.DataFrame.from_dict(
        newcol, orient='index', columns=['ifr'])
    ifr_values.index.name = 'age'

    # Transform recorded matrix of serial intervals to csv file
    path_ = os.path.join(
        os.path.dirname(__file__), 'fatality_ratio_data/')
    path = os.path.join(path_, 'CFR.csv')
    path1 = os.path.join(path_, 'IFR.csv')

    cfr_values.to_csv(path, index='False')
    ifr_values.to_csv(path1, index='False')


if __name__ == '__main__':
    main()
