from model.neck.adjust_size import AdjustAllSizesLayer
from model.neck.adjust_channels import AdjustAllChannelsLayer
from model.neck.identity import Identity


def get_neck(name, kwargs):
    if name == "AdjustAllLayer":
        return AdjustAllChannelsLayer(**kwargs)

    if name == "AdjustSize":
        return AdjustAllSizesLayer(**kwargs)

    if name == "Identity":
        return Identity()

    raise RuntimeError("No neck for {}".format(name))
