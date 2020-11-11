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
from nd2reader import ND2Reader
from dask import delayed
import dask.array as da
import toolz as tz
from napari_plugin_engine import napari_hook_implementation
import dask
from vispy.color import Colormap

CHANNEL_COLORS = {
    "Alxa 647": (1.0, 0.5019607843137255, 1.0),
    "GaAsP Alexa 568": (1.0, 0.0, 0.0),
    "GaAsP Alexa 488": (0.0, 1.0, 0.0),
    "TD": (1, 1, 1)
}
VISIBLE = [
    "Alxa 647",
    "GaAsP Alexa 568",
    "GaAsP Alexa 488"
    ]


def get_nd2reader_nd2_vol(path, c, frame):
    with ND2Reader(path) as nd2_data:
        nd2_data.default_coords['c']=c
        nd2_data.bundle_axes = ('z', 'y', 'x')
        v = nd2_data.get_frame(frame)
        v = np.array(v)
    return v    


@napari_hook_implementation(specname="napari_get_reader")
def napari_get_nd2_reader(path):
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
    return nd2_reader


def nd2_reader(path):
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
    with ND2Reader(path) as nd2_data:
        channels = nd2_data.metadata['channels']
        n_timepoints = nd2_data.sizes['t']
        z_depth = nd2_data.sizes['z']
        frame_shape = (z_depth, *nd2_data.frame_shape)
        frame_dtype = nd2_data._dtype
        nd2vol = tz.curry(get_nd2reader_nd2_vol)
        layer_list = get_layer_list(channels, nd2vol, path, frame_shape, frame_dtype, n_timepoints)

    return layer_list


def get_layer_list(channels, nd2_func, nd2_data, frame_shape, frame_dtype, n_timepoints):
    channel_dict = dict(zip(channels, [[] for _ in range(len(channels))]))
    for i, channel in enumerate(channels):
        arr = da.stack(
            [da.from_delayed(delayed(nd2_func(nd2_data, i))(j),
            shape=frame_shape,
            dtype=frame_dtype)
            for  j in range(n_timepoints)]
            )
        channel_dict[channel] = dask.optimize(arr)[0]

    layer_list = []
    for channel_name, channel in channel_dict.items():
        visible = channel_name in VISIBLE
        blending = 'additive' if visible else 'translucent'
        channel_color = list(CHANNEL_COLORS[channel_name])
        color = Colormap([[0, 0, 0],channel_color])
        add_kwargs = {
            "scale": [1, 1, 1, 1],
            "name": channel_name,
            "visible": visible,
            "colormap": color,
            "blending": blending
        }
        layer_type = "image"
        layer_list.append(
            (
                channel,
                add_kwargs,
                layer_type
            )
        )       
    return layer_list