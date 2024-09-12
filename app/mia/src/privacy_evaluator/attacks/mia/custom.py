import pandas as pd
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

from target.utils import save_progress

"""
Metric and Classifier based attacks by https://github.com/inspire-group/membership-inference-evaluation
Integrated version into TF-Privacy:
    https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack
"""


def run_custom_attacks(attack_input):
    '''slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=True,
        by_percentiles=False,
        by_classification_correctness=True
    )'''

    slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=False,
        by_percentiles=False,
        by_classification_correctness=False
    )

    """
    You can add more attacks from
    https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/data_structures.py#L172
    """
    metric_attacks = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.THRESHOLD_ENTROPY_ATTACK
    ]
    trained_attacks = [
        AttackType.LOGISTIC_REGRESSION
    ]

    print("\nRunning Metric Attacks .....")
    save_progress({
        "status": "running",
        "message": "Running Metric Attacks"
    })
    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=metric_attacks)
    # print(attacks_result.summary(by_slices=True))
    pd.set_option("display.max_rows", 12, "display.max_columns", None)
    # print(attacks_result.calculate_pd_dataframe())
    result_df = attacks_result.calculate_pd_dataframe()[["slice feature", "attack type", "AUC"]]
    print(result_df)
    metric_attack_result = {}
    for i in range(len(metric_attacks)):
        metric_attack_result[result_df["attack type"][i]] = {
            "AUC": result_df["AUC"][i]
        }

    print("\nRunning Trained (Classifier) Attacks .....")
    save_progress({
        "status": "running",
        "message": "Running Trained (Classifier) Attacks"
    })
    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=trained_attacks)
    # print(attacks_result.summary(by_slices=True))
    pd.set_option("display.max_rows", 12, "display.max_columns", None)
    result_df = attacks_result.calculate_pd_dataframe()[["slice feature", "AUC"]]
    print(result_df)
    trained_attack_result = {
        "AUC": result_df["AUC"][0]
    }
    return {
        "metric_attack": metric_attack_result,
        "trained_attack": trained_attack_result
    }
