from .nn_base import NNBase

__all__ = ['NNBase', 'mpl_imshow', 'tensorboard_model']


def mpl_imshow(img, one_channel=False):
    """helper function to show an image"""
    from matplotlib import pyplot as plt
    import numpy as np
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def tensorboard_model(model, log_path: str, model_input, **kwargs):
    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(log_path)
    writer.add_graph(model, model_input)
    writer.close()
