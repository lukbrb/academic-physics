""" Module made to ease the data preparation for each classifier used in the project."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils._bunch import Bunch
import os



GALAXY_FILENAME = "data/GalaxyZoo1_DR_table2.csv"
DATA_PACKAGE = "lblearn"
DATA_PATH = os.path.join(os.getcwd(), DATA_PACKAGE, GALAXY_FILENAME)

def load_galaxies(*, return_X_y=False, as_frame=False):
    if return_X_y:
        raise NotImplementedError('The option for returning X and y is not yet available.')
    if as_frame:
        raise NotImplementedError('Option for returning pandas DataFrame is not yet available.')

    data_filename = DATA_PATH
    df = pd.read_csv(data_filename)

    feature_names = [
        "NVOTE",
        "P_EL",
        "P_CW",
        "P_ACW",
        "P_EDGE",
        "P_DK",
        "P_MG",
        "P_CS",
        "P_EL_DEBIASED",
        "P_CS_DEBIASED"
    ]

    target_names = ['SPIRAL', 'ELLIPTICAL', 'UNCERTAIN']

    df.ELLIPTICAL = df.ELLIPTICAL.replace(1, 2)
    df.UNCERTAIN = df.UNCERTAIN.replace(1, 3)
    data = df.drop(['OBJID', 'RA', 'DEC', 'SPIRAL', 'ELLIPTICAL', 'UNCERTAIN'], axis=1).values
    target_matrix = np.array(df[['SPIRAL', 'ELLIPTICAL', 'UNCERTAIN']])
    target = target_matrix.max(axis=1) - 1

    frame = None
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR="The galaxy Zoo 1 data, table 2. More info on the dataset is to come.",
        feature_names=feature_names,
        filename=data_filename,
        data_module="lblearn.data_preparation",
    )


def get_prepared_data(filename: str, process=False) -> tuple:
    df = pd.read_csv(filename)
    target_names = ['SPIRAL', 'ELLIPTICAL', 'UNCERTAIN']
    data = df.drop(['OBJID', 'RA', 'DEC'], axis=1)
    X = data.drop(target_names, axis=1).values  # We get rid of the labels
    y = data[['SPIRAL', 'ELLIPTICAL', 'UNCERTAIN']].values  # We select the labels only

    if not process:
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def _flatten_target_columns(columns: list) -> np.ndarray:
    targets = np.hstack(columns)
    values = targets[(targets == 10) | (targets == 100) | (targets == 1000)]
    values[values == 10] = 0
    values[values == 100] = 1
    values[values == 1000] = 2
    return values
