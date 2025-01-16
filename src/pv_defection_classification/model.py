from detectron2 import model_zoo
from detectron2.config import get_cfg


def get_model(Num_Classes : int = 1) -> get_cfg:
    """
    this function creates FRCNN model from detectron2 ecosystem

    Args:
        Num_Classes: int, number of classes (no +1 needed for background)

    Returns:
        cfg: detectron2 model rcnn

    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pv_module_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = Num_Classes

    return cfg

