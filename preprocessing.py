import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_headers_1 = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "result"
]

def preprocess():
    # TRAIN_DATA_FILENAME = "NNdata/adult-train.csv"
    # TEST_DATA_FILENAME = "NNdata/adult-test.csv"
    # feature_columns = data_headers_1[0:-1]
    # label_column = data_headers_1[-1]
    #
    # df_train_raw = pd.read_csv(TRAIN_DATA_FILENAME, sep=',', names=data_headers_1)
    # df_test_raw = pd.read_csv(TEST_DATA_FILENAME, sep=',', names=data_headers_1)
    #
    # # Fix categorical data
    # df_train_raw = df_train_raw.apply(LabelEncoder().fit_transform)
    # df_train_negative = df_train_raw[df_train_raw["result"].isin([0])]
    # df_train_positive = df_train_raw[df_train_raw["result"].isin([1])]
    # df_train_negative = df_train_negative.sample(n=7841)
    # df_train = pd.concat([df_train_positive, df_train_negative], axis=0).reset_index()
    # df_train = df_train.sample(frac=1).reset_index(drop=True)
    #
    # df_test_raw = df_test_raw.apply(LabelEncoder().fit_transform)
    # df_test_negative = df_test_raw[df_test_raw["result"].isin([0])]
    # df_test_positive = df_test_raw[df_test_raw["result"].isin([1])]
    # df_test_negative = df_test_negative.sample(n=3846)
    # df_test = pd.concat([df_test_positive, df_test_negative], axis=0).reset_index()
    # df_test = df_test.sample(frac=1).reset_index(drop=True)
    #
    # X_train = df_train[feature_columns]
    # y_train = df_train[label_column]
    # X_test = df_test[feature_columns]
    # y_test = df_test[label_column]

    DATA_FILENAME = "NNdata/winequality-red.csv"
    df = pd.read_csv(DATA_FILENAME, sep=';')
    X = df.loc[:, df.columns != "quality"]
    y = df.loc[:, df.columns == "quality"].apply(lambda x: x >= 6)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return (X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    preprocess()