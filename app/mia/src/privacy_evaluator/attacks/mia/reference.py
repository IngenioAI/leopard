from privacy_meter.information_source import InformationSource
from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.constants import InferenceGame
import torch
import os

from _utils.wrapper_model import WrapperTF, WrapperTorch
from _utils.helper import get_trg_ref_data
from attacks.config import priv_meter as pm
from attacks.config import device

from _utils.helper import convert_result_list


def run_reference_metric(tdata, model, num_class, is_torch):
    target_dataset = get_trg_ref_data(tdata, num_class=num_class)

    ref_path = os.listdir(pm['ref_models'])
    ref_models = []

    if is_torch:
        suff = '.pt'
        target_model = WrapperTorch(
            model_obj=model.to(device), loss_fn=pm['torch_loss'])
    else:
        suff = '.h5'
        target_model = WrapperTF(model_obj=model, loss_fn=pm['tf_loss'])

    for reference in ref_path:
        if reference.endswith(suff):
            fpath = pm['ref_models'] + reference
            if is_torch:
                model = torch.load(fpath)
                ref_models.append(WrapperTorch(
                    model_obj=model.to(device), loss_fn=pm['torch_loss']))
            else:
                model.load_weights(fpath)
                ref_models.append(
                    WrapperTF(model_obj=model, loss_fn=pm['tf_loss']))

    print('Number of reference models found:', len(ref_models))
    target_info_source = InformationSource(
        models=[target_model],
        datasets=[target_dataset]
    )

    reference_info_source = InformationSource(
        models=ref_models,
        datasets=[target_dataset]
    )
    audit_obj = Audit(
        metrics=MetricEnum.REFERENCE,
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=pm['fpr_tolerance_list'],
        logs_directory_names=["/apprun"],
        save_logs=True
    )
    print("Preparing reference metric attack....")
    audit_obj.prepare()

    print("Starting reference metric attack....")
    audit_results = audit_obj.run()[0]
    print(audit_results[0])

    results = convert_result_list(audit_results)
    return results[0]
