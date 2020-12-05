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


CHANNEL_METADATA_PATH = (
        b'SLxPictureMetadata',
        b'sPicturePlanes',
        b'sSampleSetting',
        b'a0',
        b'pCameraSetting',
        b'Metadata',
        b'Channels',  # contains dict of channels
        # b'Channel_0',  # channel key
        # b'Color',   # color, or b'Name' name
        )


def gettup(d : dict, k : tuple):
    """Get value inside a nested dictionary by passing list of keys."""
    first, *rest = k
    if len(rest) == 0:
        return d[first]
    else:
        return gettup(d[first], rest)


def decimal2rgba(dec):
    """Convert decimal color to RGBA float array.

    References
    ----------
    https://convertingcolors.com/
    """
    return np.array([dec], '<u4').view(np.uint8) / 255


def get_metadata(path):
    with ND2Reader(path) as image:
        meta = image.metadata
        raw_meta = image.parser._raw_metadata.image_metadata
        raw_meta_seq = image.parser._raw_metadata.image_metadata_sequence

        # Scale
        try:
            z_scale = (
                    raw_meta[b'SLxExperiment'][b'ppNextLevelEx']
                            [b''][b'uLoopPars'][b'dZStep']
            )
        except KeyError:
            z_scale = 4  # TODO(jni): not a good default in general, but needed
        x_scale = y_scale = meta['pixel_microns']
        # sampling interval is in ms, we convert to s
        t_scale = meta['experiment']['loops'][0]['sampling_interval'] / 1e3
        scale = [1, z_scale, -y_scale, -x_scale]

        # Translation
        centre_x = raw_meta_seq[b'SLxPictureMetadata'][b'dXPos']
        centre_y = raw_meta_seq[b'SLxPictureMetadata'][b'dYPos']
        # z appears to be interpreted as the origin by Imaris...
        # not sure this is correct...
        origin_z = raw_meta_seq[b'SLxPictureMetadata'][b'dZPos']
        size_x = image.sizes['x']
        size_y = image.sizes['y']
        origin_t = 0  # TODO(jni): find a good way to set timepoint offset
        origin = (
            origin_t,
            origin_z,
            centre_y + size_y / 2 * y_scale,
            centre_x + size_x / 2 * x_scale,
        )

        # Colors
        channels_dict = gettup(raw_meta_seq, CHANNEL_METADATA_PATH)
        channel_colors = {
            channels_dict[ch][b'Name'].decode(): \
                                decimal2rgba(channels_dict[ch][b'Color'])
            for ch in channels_dict
            }

    return {'scale': scale, 'translate': origin, 'channels': channel_colors}




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


def get_layer_list(channels, nd2_func, path, frame_shape, frame_dtype, n_timepoints):
    channel_dict = dict(zip(channels, [[] for _ in range(len(channels))]))
    for i, channel in enumerate(channels):
        arr = da.stack(
            [da.from_delayed(delayed(nd2_func(path, i))(j),
            shape=frame_shape,
            dtype=frame_dtype)
            for  j in range(n_timepoints)]
            )
        channel_dict[channel] = dask.optimize(arr)[0]

    layer_list = []
    for channel_name, channel in channel_dict.items():
        visible = True
        blending = 'additive' if visible else 'translucent'
        meta = get_metadata(path)
        channel_color = meta['channels'][channel_name]
        color = Colormap([[0, 0, 0], channel_color[:-1]])  # ignore alpha
        add_kwargs = {
            "name": channel_name,
            "visible": visible,
            "colormap": color,
            "blending": blending,
            "scale": meta['scale'],
            "translate": meta['translate'],
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
