from model.head.multi_rpn import MultiRPN


def get_rpn_head(name, kwargs):
    if name == "MultiRPN":
        return MultiRPN(**kwargs)

    raise RuntimeError("No head for {}".format(name))
