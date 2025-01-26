{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "EqzmiY6DLtth"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from imblearn.over_sampling import SMOTE\n",
        "from imblearn.under_sampling import RandomUnderSampler\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.read_csv('Creditcard_data.csv')"
      ],
      "metadata": {
        "id": "fTIMTx8tL2XW"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(data.head())\n",
        "print(data.info())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ezwfsvX8L2rP",
        "outputId": "f6be7d05-3dec-45ce-a7b0-112572a49c85"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
            "0     0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
            "1     0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
            "2     1 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
            "3     1 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
            "4     2 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
            "\n",
            "         V8        V9  ...       V21       V22       V23       V24       V25  \\\n",
            "0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   \n",
            "1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   \n",
            "2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   \n",
            "3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   \n",
            "4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   \n",
            "\n",
            "        V26       V27       V28  Amount  Class  \n",
            "0 -0.189115  0.133558 -0.021053  149.62      0  \n",
            "1  0.125895 -0.008983  0.014724    2.69      1  \n",
            "2 -0.139097 -0.055353 -0.059752  378.66      0  \n",
            "3 -0.221929  0.062723  0.061458  123.50      0  \n",
            "4  0.502292  0.219422  0.215153   69.99      0  \n",
            "\n",
            "[5 rows x 31 columns]\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 772 entries, 0 to 771\n",
            "Data columns (total 31 columns):\n",
            " #   Column  Non-Null Count  Dtype  \n",
            "---  ------  --------------  -----  \n",
            " 0   Time    772 non-null    int64  \n",
            " 1   V1      772 non-null    float64\n",
            " 2   V2      772 non-null    float64\n",
            " 3   V3      772 non-null    float64\n",
            " 4   V4      772 non-null    float64\n",
            " 5   V5      772 non-null    float64\n",
            " 6   V6      772 non-null    float64\n",
            " 7   V7      772 non-null    float64\n",
            " 8   V8      772 non-null    float64\n",
            " 9   V9      772 non-null    float64\n",
            " 10  V10     772 non-null    float64\n",
            " 11  V11     772 non-null    float64\n",
            " 12  V12     772 non-null    float64\n",
            " 13  V13     772 non-null    float64\n",
            " 14  V14     772 non-null    float64\n",
            " 15  V15     772 non-null    float64\n",
            " 16  V16     772 non-null    float64\n",
            " 17  V17     772 non-null    float64\n",
            " 18  V18     772 non-null    float64\n",
            " 19  V19     772 non-null    float64\n",
            " 20  V20     772 non-null    float64\n",
            " 21  V21     772 non-null    float64\n",
            " 22  V22     772 non-null    float64\n",
            " 23  V23     772 non-null    float64\n",
            " 24  V24     772 non-null    float64\n",
            " 25  V25     772 non-null    float64\n",
            " 26  V26     772 non-null    float64\n",
            " 27  V27     772 non-null    float64\n",
            " 28  V28     772 non-null    float64\n",
            " 29  Amount  772 non-null    float64\n",
            " 30  Class   772 non-null    int64  \n",
            "dtypes: float64(29), int64(2)\n",
            "memory usage: 187.1 KB\n",
            "None\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = data.drop(\"Class\", axis=1)\n",
        "y = data[\"Class\"]"
      ],
      "metadata": {
        "id": "JvuR1_RwM_hR"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "smote = SMOTE(random_state=42)\n",
        "X_resampled, y_resampled = smote.fit_resample(X, y)"
      ],
      "metadata": {
        "id": "9VhqekJBM_ma"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Original class distribution:\")\n",
        "print(y.value_counts())\n",
        "print(\"Resampled class distribution:\")\n",
        "print(y_resampled.value_counts())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TVcNaRSZM_qK",
        "outputId": "de9fd171-1a58-4b79-a21e-690c8c4fb6f1"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Original class distribution:\n",
            "Class\n",
            "0    763\n",
            "1      9\n",
            "Name: count, dtype: int64\n",
            "Resampled class distribution:\n",
            "Class\n",
            "0    763\n",
            "1    763\n",
            "Name: count, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sample_size = 1000"
      ],
      "metadata": {
        "id": "4Z8bve_kM_tn"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "samples = {\n",
        "    \"Sampling1\": X_resampled.sample(n=sample_size, random_state=42),\n",
        "    \"Sampling2\": X_resampled.sample(n=sample_size, random_state=21),\n",
        "    \"Sampling3\": X_resampled.iloc[::len(X_resampled)//sample_size, :],\n",
        "    \"Sampling4\": X_resampled.sample(n=sample_size, random_state=56),\n",
        "    \"Sampling5\": X_resampled.sample(n=sample_size, random_state=99),\n",
        "}"
      ],
      "metadata": {
        "id": "wqC3nY26M_xW"
      },
      "execution_count": 39,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sample_datasets = {\n",
        "    name: (sample, y_resampled.loc[sample.index])\n",
        "    for name, sample in samples.items()\n",
        "}\n"
      ],
      "metadata": {
        "id": "yE4cvFu6M_1C"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "models = {\n",
        "    \"M1\": LogisticRegression(random_state=42),\n",
        "    \"M2\": DecisionTreeClassifier(random_state=42),\n",
        "    \"M3\": RandomForestClassifier(random_state=42),\n",
        "    \"M4\": SVC(random_state=42),\n",
        "    \"M5\": KNeighborsClassifier(),\n",
        "}\n",
        "\n",
        "results = {}"
      ],
      "metadata": {
        "id": "V3ilwaE3M_5B"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for sample_name, (X_sample, y_sample) in sample_datasets.items():\n",
        "    print(f\"Evaluating for {sample_name}...\")\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)\n",
        "\n",
        "    for model_name, model in models.items():\n",
        "        model.fit(X_train, y_train)\n",
        "        predictions = model.predict(X_test)\n",
        "        accuracy = accuracy_score(y_test, predictions)\n",
        "        if sample_name not in results:\n",
        "            results[sample_name] = {}\n",
        "        results[sample_name][model_name] = accuracy\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9VwW4J3vN23Z",
        "outputId": "d5cd0f55-c6ec-4051-ddb3-b0312f9ffc4a"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Evaluating for Sampling1...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Evaluating for Sampling2...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Evaluating for Sampling3...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Evaluating for Sampling4...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Evaluating for Sampling5...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "matrix_data = []\n",
        "for sample_name, accuracies in results.items():\n",
        "    row = [accuracies.get(model, None) for model in models.keys()]\n",
        "    matrix_data.append(row)\n",
        "\n",
        "results_matrix = pd.DataFrame(\n",
        "    matrix_data,\n",
        "    index=results.keys(),\n",
        "    columns=models.keys()\n",
        ")\n",
        "\n"
      ],
      "metadata": {
        "id": "WO0-Y6_lN28F"
      },
      "execution_count": 46,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Accuracy Matrix (Assignment Format):\")\n",
        "print(results_matrix)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7aDpFSHmQLFE",
        "outputId": "1b481854-8ea8-48d6-a9fc-0e1c1249254e"
      },
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy Matrix (Assignment Format):\n",
            "                 M1        M2        M3        M4        M5\n",
            "Sampling1  0.906667  0.983333  0.996667  0.676667  0.860000\n",
            "Sampling2  0.906667  0.963333  1.000000  0.696667  0.840000\n",
            "Sampling3  0.906114  0.978166  0.993450  0.676856  0.847162\n",
            "Sampling4  0.913333  0.960000  0.996667  0.663333  0.836667\n",
            "Sampling5  0.916667  0.980000  0.990000  0.666667  0.820000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "best_combinations = results_matrix.idxmax()\n",
        "print(\"Best Sampling Technique for Each Model:\")\n",
        "print(best_combinations)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tL02hDNzN29_",
        "outputId": "c15e420b-816b-4695-a8a9-06cdb3b766db"
      },
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Best Sampling Technique for Each Model:\n",
            "M1    Sampling5\n",
            "M2    Sampling1\n",
            "M3    Sampling2\n",
            "M4    Sampling2\n",
            "M5    Sampling1\n",
            "dtype: object\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "results_matrix.to_csv(\"results_matrix_assignment_exact.csv\")"
      ],
      "metadata": {
        "id": "kM28awFnUtvN"
      },
      "execution_count": 50,
      "outputs": []
    }
  ]
}
