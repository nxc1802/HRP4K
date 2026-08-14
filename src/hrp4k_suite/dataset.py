from hrp4k.data.paths import SPLITS, image_path
from hrp4k.data.coco import SCALE_ORDER, scale_class, load_split, iter_master_rows, available_image_ids, filtered_coco
from hrp4k.data.audit import analyze_dataset, _summary, _rank, _spearman, _js_divergence, _ks_distance
import hrp4k.data.identity as _id_mod
import hrp4k.data.views as _views
from hrp4k.data.identity import verify_dataset_identity


def prepare_dataset_view(*args, **kwargs):
    # If verify_dataset_identity was patched in this module's namespace, reflect into identity module
    orig = _id_mod.verify_dataset_identity
    try:
        if verify_dataset_identity != orig:
            _id_mod.verify_dataset_identity = verify_dataset_identity
        return _views.prepare_dataset_view(*args, **kwargs)
    finally:
        _id_mod.verify_dataset_identity = orig


def prepare_smoke_dataset(*args, **kwargs):
    return prepare_dataset_view(*args, **kwargs)


dataset_completeness = _views.dataset_completeness
