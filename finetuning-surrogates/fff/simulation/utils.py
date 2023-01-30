"""Utilities related to using ASE and other simulation tools"""

from io import StringIO

from ase import Atoms
from ase import io


# TODO (wardlt): These feel like good additions to ASE
def write_to_string(atoms: Atoms, fmt: str) -> str:
    """Write an ASE atoms object to string

    Args:
        atoms: Structure to write
        fmt: Target format
    Returns:
        Structure written in target format
    """

    out = StringIO()
    atoms.write(out, fmt)
    return out.getvalue()


def read_from_string(atoms_msg: str, fmt: str) -> Atoms:
    """Read an ASE atoms object from a string

    Args:
        atoms_msg: String format of the object to read
        fmt: Format (cannot be autodetected)
    Returns:
        Parsed atoms object
    """

    out = StringIO(str(atoms_msg))  # str() ensures that Proxies are resolved
    return io.read(out, format=fmt)
