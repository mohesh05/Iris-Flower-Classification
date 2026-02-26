import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model

def main():
    os.makedirs("plots", exist_ok=True)

    # Load dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = df['target'].map(dict(enumerate(iris.target_names)))
    df.drop('target', axis=1, inplace=True)

    # EDA plot
    sns.pairplot(df, hue='species')
    plt.savefig("plots/pairplot.png")
    plt.close()

    # Preprocess
    X, y = preprocess_data(df)

    # Train
    model, X_test, y_test = train_model(X, y)

    # Evaluate
    cm = evaluate_model(model, X_test, y_test)

    # Confusion matrix plot
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    print("\nPlots saved in /plots folder")

if __name__ == "__main__":
    main()
