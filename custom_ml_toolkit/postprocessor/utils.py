import numpy as np
import pandas as pd

# def decode_predict_proba(
#     prediction_results: np.array,
#     classes: list,
#     only_label: bool = True,
#     top: int = None
# ):
#     decoded_prediction_results = list()
#     for prediction_result in prediction_results:
#         prediction_result = zip(list(classes),
#                                 list(prediction_result))
#         prediction_result = sorted(
#             prediction_result,
#             key=lambda i: i[1],
#             reverse=True)

#         if only_label:
#             prediction_result = [product[0] for product in prediction_result]
#         if top is None:
#             decoded_prediction_results.append(prediction_result)
#         else:
#             decoded_prediction_results.append(prediction_result[:top])

#     return decoded_prediction_results


def decode_predict_proba(
    prediction_results: np.array,
    classes: list,
    top: int = None
):

    if top is None:
        top = len(classes)

    decoded_prediction_results = list()
    for prediction_result in prediction_results:
        prediction_result = zip(
            list(classes),
            list(prediction_result)
        )
        prediction_result = sorted(
            prediction_result,
            key=lambda i: i[1],
            reverse=True
        )
        pred_class_result = (
            [product[0] for product in prediction_result]
        )
        pred_proba_result = (
            [product[1] for product in prediction_result]
        )
        decoded_prediction_results.append(
            pred_class_result[:top] +
            pred_proba_result[:top]
        )

    col_names = (
        [f'class_order_{i+1}' for i in range(len(classes))][:top]
        + [f'class_prob_order_{i+1}' for i in range(len(classes))][:top]
    )

    decoded_prediction_results_df = pd.DataFrame(
        data=decoded_prediction_results,
        columns=col_names
    )

    return decoded_prediction_results_df
