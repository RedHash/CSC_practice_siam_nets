from model.chest.identity import Identity
from model.chest.bifpn import BIFPN


def get_chest(name, kwargs):
    if name.startswith('Identity'):
        return Identity()

    if name.startswith('BIFPN'):
        return BIFPN(**kwargs)

    raise RuntimeError("No chest for {}".format(name))
