"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
import numpy as np
import ND2Reader
from dask import delayed
import dask.array as da
import toolz as tz
from napari_plugin_engine import napari_hook_implementation


def get_nd2_vol(nd2_data, c, frame):
    nd2_data.default_coords['c']=c
    nd2_data.bundle_axes = ('y', 'x', 'z')
    v = nd2_data.get_frame(frame)
    v = np.array(v)
    return v



@napari_hook_implementation
def napari_get_reader(path):
    """Returns reader if path is valid .nd2 file, otherwise None

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized nd2 file, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """

    # can only read nd2 files
    if not path.endswith(".nd2"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    nd2_data = ND2Reader('200519_IVMTR69_Inj4_dmso_exp3.nd2')
    object_channel = 2

    frame_0 = get_nd2_vol(nd2_data, object_channel, 70)

    nd2vol = tz.curry(get_nd2_vol)
    arr = da.stack(
    [da.from_delayed(delayed(nd2vol(nd2_data, 2))(i),
     shape=frame_0.shape,
     dtype=frame_0.dtype)
     for  i in range(193)]  # note hardcoded n-timepoints
     )

    add_kwargs = {
        "scale": [1, 1, 1, 4]
    }

    layer_type = "image"  # optional, default is "image"
    return [(arr, add_kwargs, layer_type)]
