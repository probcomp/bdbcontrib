# -*- coding: utf-8 -*-
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# QUESTION 1: How to deal with missing features?

def handle_missing(D, strategy):
    D = pd.DataFrame(D)
    if strategy == 'drop':
        return D.dropna()
    elif strategy == 'impute':
        # Mean imputation for numerical Dry_Mass_kg.
        D['Dry_Mass_kg'] = Imputer(strategy='mean', axis=1).fit_transform(
            D['Dry_Mass_kg']).T
        # Mode imputation for categorical Dry_Mass_kg.
        D['Class_of_Orbit'] = Imputer(
            strategy='most_frequent', axis=1).fit_transform(
            D['Class_of_Orbit']).T
        return D

# ------------------------------------------------------------------------------
# QUESTION 2: How to code categorical predictor Class_of_Orbit?

def handle_coding(X, strategy):
    X = pd.DataFrame(X)
    if strategy == 'no-coding':
        return X, None
    if strategy == 'dummy-coding':
        encoder = OneHotEncoder(sparse=False)
        # Compute codes.
        codes = encoder.fit_transform(
            [[orbit] for orbit in X['Class_of_Orbit']])
        # Append codes.
        X = pd.concat((X,
                pd.DataFrame(
                    codes,
                    columns=['OrbitD0','OrbitD1','OrbitD2','OrbitD3'],
                    index=X.index)),
                    axis=1)
        del X['OrbitD3']
        del X['Class_of_Orbit']
        return X, encoder

# ------------------------------------------------------------------------------
# QUESTION 3: Which classifier to use (and which SVM kernel)?

def build_classifier(X, Y, classifier, kwargs):
    if classifier == 'forest':
        return RandomForestClassifier(
            random_state=np.random.RandomState(0)).fit(X, Y)
    elif classifier == 'svm':
        return SVC(
            random_state=np.random.RandomState(0),
            probability=True, **kwargs).fit(X, Y)

# ------------------------------------------------------------------------------
# QUESTION 4: How to predict a 2-dimensional response variable?

def predict(X, Y, x, classifier, kwargs, response):
    rng = np.random.RandomState(0)
    if response == 'joint':
        Y = Y['Country_of_Operator'].astype(str)+'-'+Y['Purpose'].astype(str)
        enc = LabelEncoder()
        Y = enc.fit_transform(Y)
        classifier_joint = build_classifier(X, Y, classifier, kwargs)
        # Preidct
        y_prob = classifier_joint.predict_proba([x])[0]
        y = rng.choice(xrange(len(y_prob)), p=y_prob, size=20)
        return [map(int, enc.inverse_transform(s).split('-')) for s in y]

    elif response == 'separate':
        classifier_country = build_classifier(
            X, Y['Country_of_Operator'], classifier, kwargs)
        y_prob = classifier_country.predict_proba([x])[0]
        countries = np.random.choice(xrange(len(y_prob)), p=y_prob, size=20)

        classifier_purpose = build_classifier(
            X, Y['Purpose'], classifier, kwargs)
        y_prob = classifier_purpose.predict_proba([x])[0]
        purposes = rng.choice(xrange(len(y_prob)), p=y_prob, size=20)
        return zip(countries, purposes)

# ------------------------------------------------------------------------------
# Experiment dispatcher.

def run_experiment(missing, coding, classifier, kernel, response):
    # Load dataset.
    D = pd.read_csv('resources/satellites.csv')

    # Throw away all other data for the query.
    D = D[['Country_of_Operator','Purpose','Class_of_Orbit','Dry_Mass_kg']]

    # Convert strings to integers.
    orbit_encoder = LabelEncoder()
    country_encoder = LabelEncoder()
    purpose_encoder = LabelEncoder()

    D['Purpose'] = purpose_encoder.fit_transform(D['Purpose'])
    D['Class_of_Orbit'] = orbit_encoder.fit_transform(D['Class_of_Orbit'])
    D['Country_of_Operator'] = country_encoder.fit_transform(
        D['Country_of_Operator'])

    # Handle missing data.
    D = handle_missing(D, strategy=missing)

    # Separate into features and response.
    X = D[['Dry_Mass_kg','Class_of_Orbit']]
    Y = D[['Country_of_Operator','Purpose']]

    # Dummy code the variables.
    X, enc = handle_coding(X, strategy=coding)

    # The query sample.
    orbit_query = orbit_encoder.transform(['GEO'])
    dry_mass_query = 500
    x = (dry_mass_query, orbit_query)
    if enc is not None:
        orbit_dummy = enc.transform(orbit_query)[0][:-1]
        x = np.hstack((dry_mass_query, orbit_dummy))

    # Predict!
    y = predict(X, Y, x, classifier, {}, response)

    # Convert to string-string pairs.
    simulations = [country_encoder.inverse_transform([a])[0] +'-'+
        purpose_encoder.inverse_transform([b])[0] for (a,b) in y]

    # Plot histogram.
    histogram = pd.Series(simulations).value_counts()
    f = open('skl_output','a')
    f.write(str((missing, coding, classifier, kernel, response)))
    f.write(histogram.to_string())
    f.write('\n')
    f.close()

if __name__ == '__main__':
    # Choices for sklearn.
    missing = ['drop', 'impute']
    coding = ['no-coding', 'dummy-coding']
    classifier = ['svm', 'forest']
    kernel = [{'kernel':'rbf'}, {'kernel':'linear'}, {'kernel':'poly'},
        {'kernel':'sigmoid'}]
    response = ['joint', 'separate']

    # SVM
    for m, c, k, r in itertools.product(missing, coding, kernel, response):
        print m, c, 'svm', k, r
        run_experiment(m, c, 'svm', k, r)

    # Random Forest
    for m, c, r in itertools.product(missing, coding, response):
        print m, c, 'forest', r
        run_experiment(m,c, 'forest', {}, r)
