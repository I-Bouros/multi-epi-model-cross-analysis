#
# Compute Japan Contacts
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
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

import os
import pandas as pd
import numpy as np


def read_contact_matrices(
        file_index: int = 1,
        state: str = 'Luxembourg'):
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


def main():
    """
    Combines the timelines of deviation percentages and baseline
    activity-specific contact matrices to get region-specific
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
    house_baseline_contact_matrix = np.zeros_like(baseline_matrices[0])
    school_baseline_contact_matrix = np.zeros_like(baseline_matrices[0])
    work_baseline_contact_matrix = np.zeros_like(baseline_matrices[0])
    other_baseline_contact_matrix = np.zeros_like(baseline_matrices[0])

    for ind, a in enumerate(activity):
        baseline_contact_matrix += baseline_matrices[ind]

        if a == 'home':
            house_baseline_contact_matrix += baseline_matrices[ind]
        elif a == 'school':
            school_baseline_contact_matrix += baseline_matrices[ind]
        elif a == 'work':
            work_baseline_contact_matrix += baseline_matrices[ind]
        else:
            other_baseline_contact_matrix += baseline_matrices[ind]

    # Transform recorded matrix of serial intervals to csv file
    path_ = os.path.join(
        os.path.dirname(__file__), 'final_contact_matrices/')
    path6 = os.path.join(path_, 'BASE_Luxembourg.csv')
    path7 = os.path.join(path_, 'house_BASE_Luxembourg.csv')
    path8 = os.path.join(path_, 'school_BASE_Luxembourg.csv')
    path9 = os.path.join(path_, 'work_BASE_Luxembourg.csv')
    path10 = os.path.join(path_, 'other_BASE_Luxembourg.csv')

    np.savetxt(
        path6, baseline_contact_matrix,
        delimiter=',')
    np.savetxt(
        path7, house_baseline_contact_matrix,
        delimiter=',')
    np.savetxt(
        path8, school_baseline_contact_matrix,
        delimiter=',')
    np.savetxt(
        path9, work_baseline_contact_matrix,
        delimiter=',')
    np.savetxt(
        path10, other_baseline_contact_matrix,
        delimiter=',')


if __name__ == '__main__':
    main()
