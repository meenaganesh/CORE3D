import argparse
import os


def arg_exist_file(x):
    """
    for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def derived_path(path, append='', alt_ext=None, alt_dir=None):
    """Returns a derived path.

    Args:
        path: The source path to derive from.
        append (str): optional append to derived path (before extension).
        alt_ext (str): optional alternate extension for derived path; otherwise path extension will be used.
        alt_dir (str): optional alternate directory for derived path; otherwise directory of path will be used.

    Returns:
        The return the derived path.

    Raises:
        ValueError: If the supplied arguments end up producing the same file path as the src_path

    """
    base_name = os.path.splitext(os.path.basename(path))[0]

    dir_name = os.path.dirname(path)
    if alt_dir is not None:
        dir_name = alt_dir

    ext = os.path.splitext(os.path.basename(path))[1]
    if alt_ext is not None:
        ext = alt_ext

    result = os.path.join(dir_name, base_name + append + ext)
    if path == result:
        raise ValueError('derived path resolves to same path: {}'.format(path))
    return result




