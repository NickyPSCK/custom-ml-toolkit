import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importances(
    feature_importance,
    feature_names,
    top_n=None,
    figsize=(7, 7)
):
    feature_name_col = 'feature_names'
    imp_score_col = 'score'
    percent_imp_score_col = f'%{imp_score_col}'
    cum_percent_imp_score_col = f'cumulative_{imp_score_col}'

    feature_importance = np.array(feature_importance)
    feature_names = np.array(feature_names)
    feat_imp_df = pd.DataFrame({
        feature_name_col: feature_names,
        imp_score_col: feature_importance
    })
    feat_imp_df = feat_imp_df\
        .sort_values(by=[imp_score_col], ascending=False)\
        .reset_index(drop=True)

    feat_imp_df[percent_imp_score_col] = (
        100
        * feat_imp_df[imp_score_col]
        / feat_imp_df[imp_score_col].sum()
    )

    feat_imp_df[cum_percent_imp_score_col] = feat_imp_df[percent_imp_score_col].cumsum()

    all_feat_imp_df = feat_imp_df.copy()
    if top_n is not None:
        feat_imp_df = feat_imp_df[:top_n]

    plt.figure(figsize=figsize)
    sns.barplot(
        x=feat_imp_df[percent_imp_score_col],
        y=feat_imp_df[feature_name_col]
    )
    plt.title('Feature Importances')
    plt.xlabel('%Score')
    plt.ylabel('Feature Names')
    plt.show()

    fig, ax1 = plt.subplots()
    fig.set_size_inches(*figsize)
    ax1.bar(
        feat_imp_df[feature_name_col],
        feat_imp_df[percent_imp_score_col],
        color='C0'
    )
    ax2 = ax1.twinx()
    ax2.plot(
        feat_imp_df[feature_name_col],
        feat_imp_df[cum_percent_imp_score_col],
        color='C1',
        marker='D',
        ms=7
    )

    ax1.tick_params(axis='y', colors='C0', labelrotation=0)
    ax2.tick_params(axis='y', colors='C1', labelrotation=0)
    ax1.tick_params(axis='x', labelrotation=90)

    plt.show()

    return all_feat_imp_df


def plot_shap_values(
    clf,
    X,
    label
):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values=shap_values,
        features=X,
        plot_type='bar',
        show=False
    )
    plt.title('Overall')
    plt.legend(label)
    plt.show()

    for i, label in enumerate(label):
        shap.summary_plot(
            shap_values=shap_values[i],
            features=X,
            plot_type='dot',
            show=False
        )
        plt.title(label)
        plt.show()