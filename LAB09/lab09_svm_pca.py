"""Lab 9: SVM classification, PCA dimensionality reduction, and LDA.

Run from this directory with:
    python lab09_svm_pca.py

The script uses the two UCI data files supplied with the lab and writes all
tables, figures, and a concise result summary to the ``outputs`` directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

WDBC_BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
WDBC_FEATURES = [
    f"{feature}_{statistic}"
    for statistic in ("mean", "se", "worst")
    for feature in WDBC_BASE_FEATURES
]
WINE_FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]


def load_wdbc() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the local Wisconsin Diagnostic Breast Cancer data."""
    columns = ["id", "diagnosis", *WDBC_FEATURES]
    data = pd.read_csv(ROOT / "wdbc.data", header=None, names=columns)

    if data.shape != (569, 32):
        raise ValueError(f"Expected WDBC shape (569, 32), got {data.shape}")
    if data.isna().any().any():
        raise ValueError("WDBC data unexpectedly contains missing values")
    if set(data["diagnosis"]) != {"B", "M"}:
        raise ValueError("WDBC diagnosis values must be B and M")

    # ID is an identifier and must not be used as a predictive feature.
    return data[WDBC_FEATURES], data["diagnosis"]


def load_wine() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the local UCI Wine data."""
    columns = ["class", *WINE_FEATURES]
    data = pd.read_csv(ROOT / "wine.data", header=None, names=columns)

    if data.shape != (178, 14):
        raise ValueError(f"Expected Wine shape (178, 14), got {data.shape}")
    if data.isna().any().any():
        raise ValueError("Wine data unexpectedly contains missing values")
    if set(data["class"]) != {1, 2, 3}:
        raise ValueError("Wine class values must be 1, 2, and 3")

    return data[WINE_FEATURES], data["class"]


def run_svm_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune four SVM kernels and evaluate each on one untouched test set."""
    x, y = load_wdbc()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    grids = {
        "linear": {"svc__C": [0.01, 0.1, 1, 10, 100]},
        "rbf": {
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1],
        },
        "poly": {
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 0.01, 0.1],
            "svc__degree": [2, 3],
        },
        "sigmoid": {
            "svc__C": [0.01, 0.1, 1, 10],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1],
        },
    }
    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}

    for kernel, parameter_grid in grids.items():
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel=kernel)),
            ]
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="f1_macro",
            cv=cross_validation,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        predictions = search.predict(x_test)
        matrix = confusion_matrix(y_test, predictions, labels=["B", "M"])
        matrices[kernel] = matrix

        rows.append(
            {
                "kernel": kernel,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_malignant": precision_score(
                    y_test, predictions, pos_label="M"
                ),
                "recall_malignant": recall_score(y_test, predictions, pos_label="M"),
                "f1_malignant": f1_score(y_test, predictions, pos_label="M"),
                "true_benign": int(matrix[0, 0]),
                "false_malignant": int(matrix[0, 1]),
                "false_benign": int(matrix[1, 0]),
                "true_malignant": int(matrix[1, 1]),
            }
        )
        tuning_rows.append(
            {
                "kernel": kernel,
                "best_cv_macro_f1": search.best_score_,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(
        ["f1_malignant", "accuracy"], ascending=False
    )
    tuning = pd.DataFrame(tuning_rows).set_index("kernel").loc[metrics["kernel"]]
    metrics.to_csv(OUTPUT_DIR / "svm_kernel_metrics.csv", index=False)
    tuning.reset_index().to_csv(OUTPUT_DIR / "svm_tuning_results.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, kernel in zip(axes.flat, grids):
        sns.heatmap(
            matrices[kernel],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=axis,
        )
        axis.set_title(f"{kernel.capitalize()} kernel")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
    fig.suptitle("SVM confusion matrices on the 20% test set", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "svm_confusion_matrices.png", dpi=180)
    plt.close(fig)

    metric_columns = [
        "accuracy",
        "precision_malignant",
        "recall_malignant",
        "f1_malignant",
    ]
    plot_data = metrics.melt(
        id_vars="kernel", value_vars=metric_columns, var_name="metric", value_name="score"
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_data, x="kernel", y="score", hue="metric", ax=axis)
    axis.set_ylim(0.80, 1.01)
    axis.set_title("Test-set performance of tuned SVM kernels")
    axis.set_xlabel("Kernel")
    axis.set_ylabel("Score")
    axis.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "svm_kernel_comparison.png", dpi=180)
    plt.close(fig)

    return metrics.reset_index(drop=True), tuning.reset_index()


def class_separation_ratio(values: np.ndarray, labels: pd.Series) -> float:
    """Return between-class scatter divided by within-class scatter."""
    overall_mean = values.mean(axis=0)
    between = 0.0
    within = 0.0
    label_array = labels.to_numpy()
    for label in np.unique(label_array):
        group = values[label_array == label]
        group_mean = group.mean(axis=0)
        between += len(group) * float(np.sum((group_mean - overall_mean) ** 2))
        within += float(np.sum((group - group_mean) ** 2))
    return between / within


def run_pca_and_lda_analysis() -> dict[str, object]:
    """Perform PCA and extra-credit LDA on standardized Wine features."""
    x, y = load_wine()
    standardized = StandardScaler().fit_transform(x)

    pca = PCA().fit(standardized)
    transformed_all = pca.transform(standardized)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    components_for_95 = int(np.argmax(cumulative >= 0.95) + 1)
    first_two_retained = float(cumulative[1])
    pca_2d = transformed_all[:, :2]

    variance_table = pd.DataFrame(
        {
            "principal_component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )
    variance_table.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        {
            "feature": WINE_FEATURES,
            "PC1_loading": pca.components_[0],
            "PC2_loading": pca.components_[1],
        }
    )
    loadings["PC1_absolute_loading"] = loadings["PC1_loading"].abs()
    loadings["PC2_absolute_loading"] = loadings["PC2_loading"].abs()
    loadings.to_csv(OUTPUT_DIR / "pca_feature_loadings.csv", index=False)

    pca_coordinates = pd.DataFrame(pca_2d, columns=["PC1", "PC2"])
    pca_coordinates.insert(0, "class", y.to_numpy())
    pca_coordinates.to_csv(OUTPUT_DIR / "wine_pca_2d.csv", index=False)

    fig, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_coordinates,
        x="PC1",
        y="PC2",
        hue="class",
        palette="Set1",
        s=65,
        alpha=0.85,
        ax=axis,
    )
    axis.set_title(
        f"Wine data after PCA (first two PCs retain {first_two_retained:.1%})"
    )
    axis.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    axis.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    axis.legend(title="Wine class")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_wine_scatter.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    component_numbers = np.arange(1, len(explained) + 1)
    axis.bar(component_numbers, explained, alpha=0.65, label="Individual variance")
    axis.plot(
        component_numbers,
        cumulative,
        marker="o",
        color="darkred",
        label="Cumulative variance",
    )
    axis.axhline(0.95, color="black", linestyle="--", linewidth=1, label="95% target")
    axis.axvline(components_for_95, color="gray", linestyle=":", linewidth=1)
    axis.set_xticks(component_numbers)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Number of principal components")
    axis.set_ylabel("Explained variance ratio")
    axis.set_title("PCA explained variance for the Wine dataset")
    axis.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_explained_variance.png", dpi=180)
    plt.close(fig)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_2d = lda.fit_transform(standardized, y)
    lda_coordinates = pd.DataFrame(lda_2d, columns=["LD1", "LD2"])
    lda_coordinates.insert(0, "class", y.to_numpy())
    lda_coordinates.to_csv(OUTPUT_DIR / "wine_lda_2d.csv", index=False)

    fig, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=lda_coordinates,
        x="LD1",
        y="LD2",
        hue="class",
        palette="Set1",
        s=65,
        alpha=0.85,
        ax=axis,
    )
    axis.set_title("Wine data after supervised LDA")
    axis.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]:.1%} discriminative variance)")
    axis.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]:.1%} discriminative variance)")
    axis.legend(title="Wine class")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lda_wine_scatter.png", dpi=180)
    plt.close(fig)

    pca_top_pc1 = (
        loadings.nlargest(5, "PC1_absolute_loading")["feature"].tolist()
    )
    pca_top_pc2 = (
        loadings.nlargest(5, "PC2_absolute_loading")["feature"].tolist()
    )
    result = {
        "pca_pc1_explained_variance": float(explained[0]),
        "pca_pc2_explained_variance": float(explained[1]),
        "pca_first_two_cumulative_variance": first_two_retained,
        "pca_components_for_95_percent": components_for_95,
        "pca_variance_at_selected_components": float(cumulative[components_for_95 - 1]),
        "pca_top_absolute_loadings_pc1": pca_top_pc1,
        "pca_top_absolute_loadings_pc2": pca_top_pc2,
        "pca_silhouette_score_2d": float(silhouette_score(pca_2d, y)),
        "lda_silhouette_score_2d": float(silhouette_score(lda_2d, y)),
        "pca_class_separation_ratio_2d": class_separation_ratio(pca_2d, y),
        "lda_class_separation_ratio_2d": class_separation_ratio(lda_2d, y),
    }
    return result


def write_summary(
    svm_metrics: pd.DataFrame,
    svm_tuning: pd.DataFrame,
    dimension_results: dict[str, object],
) -> None:
    best_kernel = str(svm_metrics.iloc[0]["kernel"])
    best_tuning = svm_tuning.set_index("kernel").loc[best_kernel]
    summary = {
        "data": {
            "wdbc_samples": 569,
            "wdbc_predictive_features": 30,
            "wine_samples": 178,
            "wine_original_features": 13,
        },
        "svm": {
            "split": "80% train / 20% test, stratified, random_state=42",
            "selection": "5-fold stratified cross-validation maximizing macro F1",
            "best_test_kernel_by_malignant_f1": best_kernel,
            "best_cv_macro_f1": float(best_tuning["best_cv_macro_f1"]),
            "best_parameters": json.loads(str(best_tuning["best_parameters"])),
            "test_metrics": svm_metrics.to_dict(orient="records"),
        },
        "dimensionality_reduction": dimension_results,
    }
    (OUTPUT_DIR / "results_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    svm_metrics, svm_tuning = run_svm_analysis()
    dimension_results = run_pca_and_lda_analysis()
    write_summary(svm_metrics, svm_tuning, dimension_results)

    print("\nSVM test metrics")
    print(svm_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSVM cross-validation tuning")
    print(svm_tuning.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPCA and LDA summary")
    for key, value in dimension_results.items():
        print(f"{key}: {value}")
    print(f"\nSaved reproducible outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
