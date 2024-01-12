import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN

def data_preparation(filename):
    descriptions = pd.read_excel(filename)
    client_bank_descriptions = descriptions[
        (descriptions.Place_of_definition == 'Вказує позичальник') |
        (descriptions.Place_of_definition == 'параметри, повязані з виданим продуктом')
    ]
    client_bank_fields = client_bank_descriptions["Field_in_data"]

    data = pd.read_excel("sample_data.xlsx")

    data = data.loc[:, list(set(client_bank_fields).intersection(data.columns))]
    data = data.dropna(axis=1)
    data.head()
    return data

def clip_data(data):
    return np.where(np.sign(data) >= 0, 1, -1) * np.clip(np.abs(data), 1e-9, None)

def min_max(data):
    return (data - data.min()) / (data.max() - data.min())

def voronin(data: pd.DataFrame, weights: np.array, direction: np.array, delta=0.1) -> np.array:
    data = data.copy()
    data.loc[direction] = 1 / clip_data(data[direction].values)
    data = data.values
    criteria_sum = np.sum(data, axis=1, keepdims=True)
    normalized_criteria_values = data / clip_data(criteria_sum)
    integro = np.dot(weights, 1 / (1 - normalized_criteria_values))
    return integro

def bayesian_regression_analysis(minimax_data):
    viz_data = minimax_data.T
    model = BayesianRidge()
    model.fit(viz_data, data["give"])
    predicted = model.predict(viz_data)
    plt.figure(figsize=(10, 5))
    plt.scatter(predicted, np.zeros_like(predicted), c=data["give"])
    plt.show()
    return predicted

def fraud_detection(minimax_data):
    outliers_fraction = 0.1
    X = minimax_data.T
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)
    norm_data = StandardScaler().fit_transform(minimax_data.T)
    compressed = PCA(n_components=2).fit_transform(norm_data)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=compressed[:, 0], y=compressed[:, 1], hue=np.where(y_pred, "Шахрайство", "Не шахрайство"))
    plt.show()
    return

def fraud_no_fraud(minimax_data):
    outliers_fraction = 0.1
    X = minimax_data.T
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)
    credit_given = minimax_data.T[data["give"] & ~y_pred].reset_index(drop=True)

    outliers_fraction = 0.026
    X = credit_given
    clf = KNN(contamination=outliers_fraction)
    clf.fit(X)
    y_pred = clf.predict(X)

    norm_data = StandardScaler().fit_transform(credit_given)
    compressed = PCA(n_components=2).fit_transform(norm_data)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
      x=compressed[:, 0],
      y=compressed[:, 1],
      hue=np.where(y_pred, "Не поверне кошти", "Поверне кошти")
    )
    plt.show()
    return

if __name__ == '__main__':
    data = data_preparation("data_description.xlsx")
    print(data)
    
    minimax_info = pd.read_excel("d_segment_data_description_cleaning_minimax.xlsx")
    minimax_info = minimax_info[["Field_in_data", "Minimax"]].dropna()
    minimax_data = data.T.loc[list(set(data.columns).intersection(minimax_info["Field_in_data"])), :]
    
    criteria_count = len(minimax_data)
    criteria_values = minimax_data.astype(float).reset_index(drop=True)
    direction = (minimax_info.set_index("Field_in_data").loc[minimax_data.index, :]["Minimax"] == "max").values
    integro = min_max(voronin(criteria_values, np.ones(criteria_count) / criteria_count, direction))
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.sort(integro.clip(0, 0.2)), label="credit scores")
    plt.hlines(0.025, 0, len(integro), color="r", label=f"threshold ({(integro <= 0.025).sum() / len(integro):.3%})")
    plt.legend()
    plt.grid()
    plt.show()

    scor_d_line = data["give"] = integro <= 0.025
    np.savetxt('Integro_Scor.txt', scor_d_line)
    print('scor_d_line= ', scor_d_line)

    bayesian_regression_analysis(minimax_data)
    fraud_detection(minimax_data)
    fraud_no_fraud(minimax_data)
