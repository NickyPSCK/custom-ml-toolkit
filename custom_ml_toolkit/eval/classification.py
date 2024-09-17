import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


def process_eval_dict(
    eval_dict: dict,
    augmented_cols: Optional[dict] = None
) -> pd.DataFrame:
    eval_df = pd.DataFrame(eval_dict).reset_index()
    pivot_values = eval_df.columns
    eval_df['pivot_index'] = 'pivot_index'
    pivoted_eval_df = eval_df.pivot(index='pivot_index', columns=['index'], values=pivot_values)
    pivoted_eval_df = pivoted_eval_df.drop(['index'], axis=1)
    pivoted_eval_df = pivoted_eval_df.reset_index(drop=True)
    pivoted_eval_df.columns = ['_'.join(col) for col in pivoted_eval_df.columns]
    if augmented_cols is not None:
        for col_name in augmented_cols:
            pivoted_eval_df[col_name] = augmented_cols[col_name]
    return pivoted_eval_df


def classification_reports(
    data: Dict[str, Tuple[int, int]],  # Set Name: (y_true, y_pred)
    labels: list = None,
    digits: int = 4
):
    for set_name in data:
        print(f'--------------- {set_name} Performance ---------------')
        print(
            classification_report(
                y_true=data[set_name][0],
                y_pred=data[set_name][1],
                labels=labels,
                digits=digits
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


def create_gain_and_lift_data(
    y_true: list,
    y_score: list,
    bin_size: int = None,
    q: int = None
):
    if (bin_size is None) and (q is None):
        raise ValueError('Either bin_size or q must be spacified.')
    elif (bin_size is not None) and (q is not None):
        raise ValueError('Cannot spacify both bin_siza or q at the same time.')
    else:
        pass

    gain_and_lift_raw_df = pd.DataFrame({
        'y_true': y_true.to_list(),
        'y_score': list(y_score)
    })\
        .sort_values('y_score', ascending=False)\
        .reset_index(drop=True)\
        .reset_index(drop=False)

    if q is not None:
        gain_and_lift_raw_df['bin'] = pd.qcut(
            gain_and_lift_raw_df['index'],
            q=q,
            labels=False
        ) + 1
    else:
        gain_and_lift_raw_df['bin'] = (
            (
                gain_and_lift_raw_df['index']
                // bin_size
            ) + 1
        ) * bin_size

    gain_and_lift_raw_df = gain_and_lift_raw_df.drop(
        columns='index'
    )

    gain_and_lift_curve_df = gain_and_lift_raw_df\
        .groupby('bin')\
        .agg({
            'y_score': ['max', 'min'],
            'y_true': ['sum', 'count']
        })
    gain_and_lift_curve_df.columns = [
        'score_max',
        'score_min',
        'converted_event',
        'total_event'
    ]
    gain_and_lift_curve_df = gain_and_lift_curve_df.reset_index(drop=False)
    gain_and_lift_curve_df['%_converted'] = (
        100
        * gain_and_lift_curve_df['converted_event']
        / gain_and_lift_curve_df['total_event']
    )
    gain_and_lift_curve_df['%_cum_converted'] = (
        100
        * gain_and_lift_curve_df['converted_event'].cumsum()
        / gain_and_lift_curve_df['converted_event'].sum()
    )

    return gain_and_lift_raw_df, gain_and_lift_curve_df


def plot_lift_curve(
    data: Dict[str, Tuple[int, int]],  # Line Name: (y_true, y_score)
    bin_size: int = None,
    q: int = None,
    baseline: Optional[float] = 50,  # %
    lift: bool = False,
    title: str = 'Lift Chart',
    xlim: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (10, 4)
):
    _, ax = plt.subplots(figsize=figsize)

    legend_list = list()
    raw_dict = dict()

    for pair_name in data:
        y_true = data[pair_name][0]
        y_score = data[pair_name][1]
        lift_raw_data_df, lift_curve_df = create_gain_and_lift_data(
            y_true=y_true,
            y_score=y_score,
            bin_size=bin_size,
            q=q
        )
        gini_score = (
            2
            * roc_auc_score(
                y_true=y_true,
                y_score=y_score
            )
            - 1)
        legend_list.append(f'{pair_name}, GINI {gini_score:.4f}')
        if lift:
            ax.plot(
                lift_curve_df['bin'],
                lift_curve_df['%_converted'] / baseline,
                'o',
                linestyle='-'
            )
        else:
            ax.plot(
                lift_curve_df['bin'],
                lift_curve_df['%_converted'],
                'o',
                linestyle='-'
            )
        raw_dict[pair_name] = {
            'raw_data': lift_raw_data_df,
            'curve_data': lift_curve_df
        }

    if baseline is not None:
        if lift:
            baseline = 1
            legend_list.append(f'Baseline: {baseline}')
        else:
            legend_list.append(f'Baseline: {baseline}%')

        ax.axhline(
            y=baseline,
            color='r',
            linestyle='--'
        )

    ax.legend(legend_list)
    ax.tick_params(axis='x', labelrotation=0)
    plt.title(title)
    plt.xlabel('Bucket')
    if lift:
        plt.ylabel('Lift')
    else:
        plt.ylabel('% of Conversion')
    plt.grid(
        visible=True,
        which='both',
        axis='both'
    )
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.show()
    return raw_dict


def plot_gain_curve(
    data: Dict[str, Tuple[int, int]],  # Line Name: (y_true, y_score)
    q: int = 10,
    bin_size: int = None,
    title: str = 'Gain Chart',
    slim_fit: bool = False,
    xlim: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (10, 4)
):
    _, ax = plt.subplots(figsize=figsize)

    legend_list = list()
    raw_dict = dict()

    max_x = 0
    for pair_name in data:
        y_true = data[pair_name][0]
        y_score = data[pair_name][1]
        lift_raw_data_df, lift_curve_df = create_gain_and_lift_data(
            y_true=y_true,
            y_score=y_score,
            bin_size=bin_size,
            q=q
        )
        gini_score = (
            2
            * roc_auc_score(
                y_true=y_true,
                y_score=y_score
            )
            - 1)
        legend_list.append(f'{pair_name}, GINI {gini_score:.4f}')

        X = [0] + lift_curve_df['bin'].to_list()
        y = [0] + lift_curve_df['%_cum_converted'].to_list()

        ax.plot(X, y, 'o', linestyle='-')
        raw_dict[pair_name] = {
            'raw_data': lift_raw_data_df,
            'curve_data': lift_curve_df
        }
        local_max_x = lift_curve_df['bin'].max()

        if q is None:
            ax.plot(
                [0, local_max_x],
                [0, 100],
                color='r',
                linestyle='--'
            )
            legend_list.append(f'Baseline {pair_name}')

        if max_x < local_max_x:
            max_x = local_max_x

    if q is not None:
        ax.plot([0, max_x], [0, 100], color='r', linestyle='--')
        legend_list.append('Baseline')

    ax.legend(legend_list)
    ax.tick_params(axis='x', labelrotation=0)
    plt.title(title)
    plt.xlabel('Bucket')
    plt.ylabel('% of Cumulative Conversion')
    plt.grid(
        visible=True,
        which='both',
        axis='both'
    )

    if slim_fit:
        plt.xlim(0, max_x)
        plt.ylim(0, 100)
    else:
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)

    plt.show()
    return raw_dict
