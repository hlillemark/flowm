from torchvision.datasets.video_utils import _collate_fn


def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return []
    return _collate_fn(batch)