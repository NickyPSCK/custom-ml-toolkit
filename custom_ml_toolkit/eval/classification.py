import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def process_eval_dict(
    eval_dict: dict,
    augmented_cols: dict
) -> pd.DataFrame:
    eval_df = pd.DataFrame(eval_dict).reset_index()
    pivot_values = eval_df.columns
    eval_df['pivot_index'] = 'pivot_index'
    pivoted_eval_df = eval_df.pivot(index='pivot_index', columns=['index'], values=pivot_values)
    pivoted_eval_df = pivoted_eval_df.drop(['index'], axis=1)
    pivoted_eval_df = pivoted_eval_df.reset_index(drop=True)
    pivoted_eval_df.columns = ['_'.join(col) for col in pivoted_eval_df.columns]
    for col_name in augmented_cols:
        pivoted_eval_df[col_name] = augmented_cols[col_name]
    return pivoted_eval_df


def eval_performance(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    y_holdout_a=None,
    y_holdout_pred_a=None,
    y_holdout_b=None,
    y_holdout_pred_b=None,
    y_holdout_c=None,
    y_holdout_pred_c=None,
    labels=None
):
    print('--------------- Train Set Performance ---------------')
    print(
        classification_report(
            y_true=y_train,
            y_pred=y_train_pred,
            labels=labels,
            digits=4
        )
    )
    print('---------------  Test Set Performance ---------------')
    print(
        classification_report(
            y_true=y_test,
            y_pred=y_test_pred,
            labels=labels,
            digits=4
        )
    )
    if (y_holdout_a is not None) and (y_holdout_pred_a is not None):
        print('---------------  Hold Out Set A Performance ---------------')
        print(
            classification_report(
                y_true=y_holdout_a,
                y_pred=y_holdout_pred_a,
                labels=labels,
                digits=4
            )
        )
    if (y_holdout_b is not None) and (y_holdout_pred_b is not None):
        print('---------------  Hold Out Set B Performance ---------------')
        print(
            classification_report(
                y_true=y_holdout_b,
                y_pred=y_holdout_pred_b,
                labels=labels,
                digits=4
            )
        )
    if (y_holdout_c is not None) and (y_holdout_pred_c is not None):
        print('---------------  Hold Out Set C Performance ---------------')
        print(
            classification_report(
                y_true=y_holdout_c,
                y_pred=y_holdout_pred_c,
                labels=labels,
                digits=4
            )
        )


def plot_confusion_matrix(y_true, y_pred):
    # confusion matrix
    confusion_matrix_df = pd.DataFrame({'actual': y_true, 'prediction': y_pred})
    confusion_matrix_df['cnt'] = 1
    confusion_matrix_df = confusion_matrix_df.pivot_table(
        index='actual',
        columns='prediction',
        values='cnt',
        aggfunc='sum'
    )
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        confusion_matrix_df,
        cmap='viridis',
        fmt='.0f',
        annot=True,
        cbar=False
    )
