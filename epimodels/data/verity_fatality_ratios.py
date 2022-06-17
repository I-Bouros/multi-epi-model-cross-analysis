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


def change_age_groups(matrix: np.array):
    """
    Reprocess the fatality ratios so that it has the appropriate age groups.

    Parameters
    ----------
    matrix : numpy.array
        Fatality ratios with old age groups.

    Returns
    -------
    numpy.array
        New fatality ratios with correct age groups.

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
    baseline_fatality_ratios = read_fatality_ratios_data()
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
                contact_matrix += act_week_multi.get(
                    key=a) * baseline_matrices[ind]

            # Transform recorded matrix of serial intervals to csv file
            path_ = os.path.join(
                os.path.dirname(__file__), 'final_contact_matrices/')
            path = os.path.join(
                    path_,
                    '{}_W{}.csv'.format(region, w+1))
            path1 = os.path.join(path_, 'BASE.csv')

            np.savetxt(path, change_age_groups(contact_matrix), delimiter=',')
            np.savetxt(
                path1, change_age_groups(baseline_contact_matrix),
                delimiter=',')


if __name__ == '__main__':
    main()
