from privacy_meter.information_source import InformationSource
from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.constants import InferenceGame

from _utils.wrapper_model import WrapperTF, WrapperTorch
from _utils.helper import get_trg_ref_data
from attacks.config import priv_meter as pm

from _utils.helper import convert_result_list

def run_population_metric(tdata, model, num_class, is_torch):
    target_dataset, reference_dataset = get_trg_ref_data(
        tdata, num_class, population=True)
    if is_torch:
        target_model = WrapperTorch(
            model_obj=model, loss_fn=pm['torch_loss'])
    else:
        target_model = WrapperTF(
            model_obj=model, loss_fn=pm['tf_loss'])

    target_info_source = InformationSource(
        models=[target_model],
        datasets=[target_dataset]
    )
    reference_info_source = InformationSource(
        models=[target_model],
        datasets=[reference_dataset]
    )

    audit_obj = Audit(
        metrics=MetricEnum.POPULATION,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=pm['fpr_tolerance_list'],
        logs_directory_names=["/apprun"],
        save_logs=True
    )
    print("Preparing population metric attack....")
    audit_obj.prepare()

    print("Starting population metric attack....")
    audit_results = audit_obj.run()[0]
    print(audit_results[0])

    results = convert_result_list(audit_results)

    return results[0]
