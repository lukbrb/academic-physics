""" Module made to ease the data preparation for each classifier used in the project."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# From sklearn.utils._bunch import Bunch
class DataSet(dict):
    """Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> from sklearn.utils import Bunch
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass


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

    with open("table2.rst") as fp:
        fdescr = fp.read()

    return DataSet(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
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
