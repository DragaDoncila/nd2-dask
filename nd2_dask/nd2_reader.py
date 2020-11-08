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
from pims import ND2_Reader
from dask import delayed
import dask.array as da
import toolz as tz
from napari_plugin_engine import napari_hook_implementation
import dask


def get_pims_nd2_vol(nd2_data, c, frame):
    nd2_data.default_coords['c']=c
    nd2_data.bundle_axes = 'zyx'
    v = nd2_data[frame]
    v = np.array(v)
    return v


def get_nd2reader_nd2_vol(path, c, frame):
    with ND2Reader(path) as nd2_data:
        nd2_data.default_coords['c']=c
        nd2_data.bundle_axes = ('z', 'y', 'x')
        v = nd2_data.get_frame(frame)
        v = np.array(v)
    return v    

# @napari_hook_implementation(specname="napari_get_reader")
def napari_get_pims_reader(path):
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
    return pims_reader


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


def pims_reader(path, max_retries=30):
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

    tries = 0
    while tries < max_retries:
        try:
            nd2_data = ND2_Reader(path)
            channels = [nd2_data.metadata[f"plane_{i}"]["name"] for i in range(0, nd2_data.sizes['c'])]
            n_timepoints = nd2_data.sizes['t']
            frame_shape = nd2_data.frame_shape
            frame_dtype = nd2_data.pixel_type
            nd2vol = tz.curry(get_pims_nd2_vol)
            channel_dict = dict(zip(channels, [[] for _ in range(len(channels))]))
            layer_list = get_layer_list(channels, nd2vol, nd2_data, frame_shape, frame_dtype, n_timepoints)

            for (data, _, _) in layer_list:
                shape = np.asarray(data[0]).shape
            break
        except Exception as e:
            tries += 1
            if tries > max_retries:
                raise e

    return layer_list


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
        add_kwargs = {
            "scale": [1, 1, 1, 1],
            "name": channel_name,
            "visible": channel_name == "Alxa 647"
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