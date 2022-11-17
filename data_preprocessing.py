import numpy as np
import pandas as pd
from utils import generate_features_values, train_test_split


def preprocessing_german(path, test_size=0.3, include_s=True):
    """ Convert data to numeric values """
    headers = 'Checkin account,Duration,Credit history,Purpose,Credit amount,Savings,Present employment,' \
              'Installment rate,Sex,Other debtors,Residence duration,Property,Age,Other installment,Housing,' \
              'Credits number,Job,Number of people,Telephone,Foreign worker,Risk'.split(',')
    data = pd.read_csv(f"{path}/german.data", names=headers, sep=' ')
    purposes = ['car', 'car_new', 'furniture/equipment', 'radio/TV', 'domestic appliances',
                'repairs', 'education', 'vacation', 'retraining', 'business', 'others']

    data["Sex"] = data["Sex"].replace(["A91", "A92", "A93", "A94", "A95"], [
        "male", "female", "male", "male", "female"])
    data["Risk"] = data["Risk"].replace([1, 2], ["good", "bad"])
    data["Checkin account"] = data["Checkin account"].replace(
        ["A11", "A12", "A13", "A14"], ['little', 'moderate', 'hight', None])
    data["Credit history"] = data["Credit history"].replace(
        ["A30", "A31", "A32", "A33", "A34"], np.arange(5))
    # data["Purpose"] = data["Purpose"].replace(generate_features_values('A4', 11, 0), np.arange(11))
    data["Purpose"] = data["Purpose"].replace(
        generate_features_values('A4', 11, 0), purposes)
    # data["Savings"] = data["Savings"].replace(generate_features_values('A6', 6), np.arange(6))
    data["Savings"] = data["Savings"].replace(generate_features_values(
        'A6', 5), ['little', 'moderate', 'rich', 'quite_rich', None])
    data["Present employment"] = data["Present employment"].replace(
        generate_features_values('A7', 6), np.arange(6))
    # data["Other debtors"] = data["Other debtors"].replace(generate_features_values('A10', 4), np.arange(4))
    data["Other debtors"] = data["Other debtors"].replace(
        generate_features_values('A10', 3), [None, 'co_appli', 'guarantor'])
    data["Property"] = data["Property"].replace(
        generate_features_values('A12', 5), np.arange(5))
    # data["Other installment"] = data["Other Installement"].replace(generate_features_values('A14', 4), np.arange(4))
    data["Other installment"] = data["Other installment"].replace(
        generate_features_values('A14', 3), ['bank', 'stores', 'none'])
    # data["Housing"] = data["Housing"].replace(generate_features_values('A15', 3), np.arange(3))
    data["Housing"] = data["Housing"].replace(
        generate_features_values('A15', 3), ['rent', 'own', 'free'])
    data["Job"] = data["Job"].replace(
        generate_features_values('A17', 5), np.arange(5))
    data["Telephone"] = data["Telephone"].replace(
        generate_features_values('A19', 2), [None, "yes"])
    data["Foreign worker"] = data["Foreign worker"].replace(
        generate_features_values('A20', 2), ["yes", None])
    data["Credit amount"] = (data["Credit amount"] - np.mean(data["Credit amount"])) / np.std(
        data["Credit amount"])  # np.log(data["Credit amount"])

    data["Duration"] = (data["Duration"] - np.mean(data["Duration"])) / np.std(
        data["Duration"])  # np.log(data["Credit amount"])

    data["Age"] = (data["Age"] - np.mean(data["Age"])) / np.std(data["Age"])  # np.log(data["Credit amount"])

    # Purpose of Dummies
    data = data.merge(pd.get_dummies(data.Purpose, drop_first=True,
                                     prefix='Purpose'), left_index=True, right_index=True)

    # Dummies of Sex
    data = data.merge(pd.get_dummies(data.Sex, drop_first=True,
                                     prefix='Sex'), left_index=True, right_index=True)

    # Dummies of Other debtors
    data = data.merge(pd.get_dummies(
        data["Other debtors"], drop_first=True, prefix='Other_debtors'), left_index=True, right_index=True)

    # Dummies of Other installment
    data = data.merge(pd.get_dummies(
        data["Other installment"], drop_first=True, prefix='Other_install'), left_index=True, right_index=True)

    # Dummies of Foreign worker
    data = data.merge(pd.get_dummies(
        data["Foreign worker"], drop_first=True, prefix='Foreign worker'), left_index=True, right_index=True)

    # Dummies of Housing
    data = data.merge(pd.get_dummies(
        data["Housing"], drop_first=True, prefix='Housing'), left_index=True, right_index=True)

    # Dummies of Telephone
    data = data.merge(pd.get_dummies(
        data["Telephone"], drop_first=True, prefix='Telephone'), left_index=True, right_index=True)

    # Dummies of Telephone
    data = data.merge(pd.get_dummies(
        data["Savings"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)

    # Dummies of Checkin account
    data = data.merge(pd.get_dummies(
        data["Checkin account"], drop_first=True, prefix='Checkin account'), left_index=True, right_index=True)

    # Dummies of Checkin account
    data = data.merge(pd.get_dummies(
        data["Risk"], drop_first=True, prefix='Risk'), left_index=True, right_index=True)

    # Delete old features
    del data["Other installment"]
    del data["Other debtors"]
    del data["Sex"]
    del data["Purpose"]
    del data["Foreign worker"]
    del data["Housing"]
    del data["Telephone"]
    del data["Savings"]
    del data["Checkin account"]
    del data["Risk"]
    S = data["Sex_male"].values

    if not include_s:
        del data["Sex_male"]

    X = data.drop("Risk_good", 1).values
    y = data["Risk_good"].values
    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=test_size)
    return X_train, X_test, y_train, y_test, S_train, S_test
