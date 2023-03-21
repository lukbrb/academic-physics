""" Module made to ease the data preparation for each classifier used in the project."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_prepared_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    data = df.drop(['OBJID','RA','DEC'], axis=1)
    X = data.drop(['SPIRAL','ELLIPTICAL','UNCERTAIN'], axis=1).values  # We get rid of the labels
    y = data[['SPIRAL','ELLIPTICAL','UNCERTAIN']].values # We select the labels only

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test