"""
Extended XYZ support

Reference:
        https://gitlab.com/ase/ase/-/blob/3.22.1/ase/io/extxyz.py
"""


from itertools import islice
import re
import warnings
from io import StringIO, UnsupportedOperation
import json

import numpy as np
import numbers

from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.constraints import FixAtoms, FixCartesian
from ase.io.formats import index2range
from ase.utils import reader

__all__ = ['read_xyz', 'write_xyz', 'iread_xyz']

PROPERTY_NAME_MAP = {'positions': 'pos',
                     'numbers': 'Z',
                     'charges': 'charge',
                     'symbols': 'species'}

REV_PROPERTY_NAME_MAP = dict(zip(PROPERTY_NAME_MAP.values(),
                                 PROPERTY_NAME_MAP.keys()))

KEY_QUOTED_VALUE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_-]*)'
                              + r'\s*=\s*["\{\}]([^"\{\}]+)["\{\}]\s*')
KEY_VALUE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_]*)\s*='
                       + r'\s*([^\s]+)\s*')
KEY_RE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_-]*)\s*')

UNPROCESSED_KEYS = ['uid']

SPECIAL_3_3_KEYS = ['Lattice', 'virial', 'stress']

# partition ase.calculators.calculator.all_properties into two lists:
#  'per-atom' and 'per-config'
per_atom_properties = ['forces', 'stresses', 'charges', 'magmoms', 'energies']
per_config_properties = ['energy', 'stress', 'dipole', 'magmom', 'free_energy']


def key_val_str_to_dict(string, sep=None):
    """
    Parse an xyz properties string in a key=value and return a dict with
    various values parsed to native types.

    Accepts brackets or quotes to delimit values. Parses integers, floats
    booleans and arrays thereof. Arrays with 9 values whose name is listed
    in SPECIAL_3_3_KEYS are converted to 3x3 arrays with Fortran ordering.

    If sep is None, string will split on whitespace, otherwise will split
    key value pairs with the given separator.

    """
    # store the closing delimiters to match opening ones
    delimiters = {
        "'": "'",
        '"': '"',
        '{': '}',
        '[': ']'
    }

    # Make pairs and process afterwards
    kv_pairs = [
        [[]]]  # List of characters for each entry, add a new list for new value
    cur_delimiter = None  # push and pop closing delimiters
    escaped = False  # add escaped sequences verbatim

    # parse character-by-character unless someone can do nested brackets
    # and escape sequences in a regex
    for char in string.strip():
        if escaped:  # bypass everything if escaped
            kv_pairs[-1][-1].append(char)
            escaped = False
        elif char == '\\':  # escape the next thing
            escaped = True
        elif cur_delimiter:  # inside brackets
            if char == cur_delimiter:  # found matching delimiter
                cur_delimiter = None
            else:
                kv_pairs[-1][-1].append(char)  # inside quotes, add verbatim
        elif char in delimiters:
            cur_delimiter = delimiters[char]  # brackets or quotes
        elif (sep is None and char.isspace()) or char == sep:
            if kv_pairs == [[[]]]:  # empty, beginning of string
                continue
            elif kv_pairs[-1][-1] == []:
                continue
            else:
                kv_pairs.append([[]])
        elif char == '=':
            if kv_pairs[-1] == [[]]:
                del kv_pairs[-1]
            kv_pairs[-1].append([])  # value
        else:
            kv_pairs[-1][-1].append(char)

    kv_dict = {}

    for kv_pair in kv_pairs:
        if len(kv_pair) == 0:  # empty line
            continue
        elif len(kv_pair) == 1:  # default to True
            key, value = ''.join(kv_pair[0]), 'T'
        else:  # Smush anything else with kv-splitter '=' between them
            key, value = ''.join(kv_pair[0]), '='.join(
                ''.join(x) for x in kv_pair[1:])

        if key.lower() not in UNPROCESSED_KEYS:
            # Try to convert to (arrays of) floats, ints
            split_value = re.findall(r'[^\s,]+', value)
            try:
                try:
                    numvalue = np.array(split_value, dtype=int)
                except (ValueError, OverflowError):
                    # don't catch errors here so it falls through to bool
                    numvalue = np.array(split_value, dtype=float)
                if len(numvalue) == 1:
                    numvalue = numvalue[0]  # Only one number
                value = numvalue
            except (ValueError, OverflowError):
                pass  # value is unchanged

            # convert special 3x3 matrices
            if key in SPECIAL_3_3_KEYS:
                if not isinstance(value, np.ndarray) or value.shape != (9,):
                    raise ValueError("Got info item {}, expecting special 3x3 "
                                     "matrix, but value is not in the form of "
                                     "a 9-long numerical vector".format(key))
                value = np.array(value).reshape((3, 3), order='F')

            # parse special strings as boolean or JSON
            if isinstance(value, str):
                # Parse boolean values: 'T' -> True, 'F' -> False,
                #                       'T T F' -> [True, True, False]
                str_to_bool = {'T': True, 'F': False}

                try:
                    boolvalue = [str_to_bool[vpart] for vpart in
                                 re.findall(r'[^\s,]+', value)]
                    if len(boolvalue) == 1:
                        value = boolvalue[0]
                    else:
                        value = boolvalue
                except KeyError:
                    # parse JSON
                    if value.startswith("_JSON "):
                        d = json.loads(value.replace("_JSON ", "", 1))
                        value = np.array(d)
                        if value.dtype.kind not in ['i', 'f', 'b']:
                            value = d

        kv_dict[key] = value

    return kv_dict


def key_val_str_to_dict_regex(s):
    """
    Parse strings in the form 'key1=value1 key2="quoted value"'
    """
    d = {}
    s = s.strip()
    while True:
        # Match quoted string first, then fall through to plain key=value
        m = KEY_QUOTED_VALUE.match(s)
        if m is None:
            m = KEY_VALUE.match(s)
            if m is not None:
                s = KEY_VALUE.sub('', s, 1)
            else:
                # Just a key with no value
                m = KEY_RE.match(s)
                if m is not None:
                    s = KEY_RE.sub('', s, 1)
        else:
            s = KEY_QUOTED_VALUE.sub('', s, 1)

        if m is None:
            break        # No more matches

        key = m.group(1)
        try:
            value = m.group(2)
        except IndexError:
            # default value is 'T' (True)
            value = 'T'

        if key.lower() not in UNPROCESSED_KEYS:
            # Try to convert to (arrays of) floats, ints
            try:
                numvalue = []
                for x in value.split():
                    if x.find('.') == -1:
                        numvalue.append(int(float(x)))
                    else:
                        numvalue.append(float(x))
                if len(numvalue) == 1:
                    numvalue = numvalue[0]         # Only one number
                elif len(numvalue) == 9:
                    # special case: 3x3 matrix, fortran ordering
                    numvalue = np.array(numvalue).reshape((3, 3), order='F')
                else:
                    numvalue = np.array(numvalue)  # vector
                value = numvalue
            except (ValueError, OverflowError):
                pass

            # Parse boolean values: 'T' -> True, 'F' -> False,
            #                       'T T F' -> [True, True, False]
            if isinstance(value, str):
                str_to_bool = {'T': True, 'F': False}

                if len(value.split()) > 1:
                    if all([x in str_to_bool.keys() for x in value.split()]):
                        value = [str_to_bool[x] for x in value.split()]
                elif value in str_to_bool:
                    value = str_to_bool[value]

        d[key] = value

    return d


def escape(string):
    if (' ' in string or
            '"' in string or "'" in string or
            '{' in string or '}' in string or
            '[' in string or ']' in string):
        string = string.replace('"', '\\"')
        string = '"%s"' % string
    return string


def key_val_dict_to_str(dct, sep=' '):
    """
    Convert atoms.info dictionary to extended XYZ string representation
    """

    def array_to_string(key, val):
        # some ndarrays are special (special 3x3 keys, and scalars/vectors of
        # numbers or bools), handle them here
        if key in SPECIAL_3_3_KEYS:
            # special 3x3 matrix, flatten in Fortran order
            val = val.reshape(val.size, order='F')
        if val.dtype.kind in ['i', 'f', 'b']:
            # numerical or bool scalars/vectors are special, for backwards
            # compat.
            if len(val.shape) == 0:
                # scalar
                val = str(known_types_to_str(val))
            elif len(val.shape) == 1:
                # vector
                val = ' '.join(str(known_types_to_str(v)) for v in val)
        return val

    def known_types_to_str(val):
        if isinstance(val, bool) or isinstance(val, np.bool_):
            return 'T' if val else 'F'
        elif isinstance(val, numbers.Real):
            return '{}'.format(val)
        elif isinstance(val, Spacegroup):
            return val.symbol
        else:
            return val

    if len(dct) == 0:
        return ''

    string = ''
    for key in dct:
        val = dct[key]

        if isinstance(val, np.ndarray):
            val = array_to_string(key, val)
        else:
            # convert any known types to string
            val = known_types_to_str(val)

        if val is not None and not isinstance(val, str):
            # what's left is an object, try using JSON
            if isinstance(val, np.ndarray):
                val = val.tolist()
            try:
                val = '_JSON ' + json.dumps(val)
                # if this fails, let give up
            except TypeError:
                warnings.warn('Skipping unhashable information '
                              '{0}'.format(key))
                continue

        key = escape(key)  # escape and quote key
        eq = "="
        # Should this really be setting empty value that's going to be
        # interpreted as bool True?
        if val is None:
            val = ""
            eq = ""
        val = escape(val)  # escape and quote val

        string += '%s%s%s%s' % (key, eq, val, sep)

    return string.strip()


def parse_properties(prop_str):
    """
    Parse extended XYZ properties format string

    Format is "[NAME:TYPE:NCOLS]...]", e.g. "species:S:1:pos:R:3".
    NAME is the name of the property.
    TYPE is one of R, I, S, L for real, integer, string and logical.
    NCOLS is number of columns for that property.
    """

    properties = {}
    properties_list = []
    dtypes = []
    converters = []

    fields = prop_str.split(':')

    def parse_bool(x):
        """
        Parse bool to string
        """
        return {'T': True, 'F': False,
                'True': True, 'False': False}.get(x)

    fmt_map = {'R': ('d', float),
               'I': ('i', int),
               'S': (object, str),
               'L': ('bool', parse_bool)}

    for name, ptype, cols in zip(fields[::3],
                                 fields[1::3],
                                 [int(x) for x in fields[2::3]]):
        if ptype not in ('R', 'I', 'S', 'L'):
            raise ValueError('Unknown property type: ' + ptype)
        ase_name = REV_PROPERTY_NAME_MAP.get(name, name)

        dtype, converter = fmt_map[ptype]
        if cols == 1:
            dtypes.append((name, dtype))
            converters.append(converter)
        else:
            for c in range(cols):
                dtypes.append((name + str(c), dtype))
                converters.append(converter)

        properties[name] = (ase_name, cols)
        properties_list.append(name)

    dtype = np.dtype(dtypes)
    return properties, properties_list, dtype, converters


def _read_xyz_frame(lines, natoms, properties_parser=key_val_str_to_dict,
                    nvec=0):
    # comment line
    line = next(lines).strip()
    if nvec > 0:
        info = {'comment': line}
    else:
        info = properties_parser(line) if line else {}

    pbc = None
    if 'pbc' in info:
        pbc = info['pbc']
        del info['pbc']
    elif 'Lattice' in info:
        # default pbc for extxyz file containing Lattice
        # is True in all directions
        pbc = [True, True, True]
    elif nvec > 0:
        # cell information given as pseudo-Atoms
        pbc = [False, False, False]

    cell = None
    if 'Lattice' in info:
        # NB: ASE cell is transpose of extended XYZ lattice
        cell = info['Lattice'].T
        del info['Lattice']
    elif nvec > 0:
        # cell information given as pseudo-Atoms
        cell = np.zeros((3, 3))

    if 'Properties' not in info:
        # Default set of properties is atomic symbols and positions only
        info['Properties'] = 'species:S:1:pos:R:3'
    properties, names, dtype, convs = parse_properties(info['Properties'])
    del info['Properties']

    data = []
    for ln in range(natoms):
        try:
            line = next(lines)
        except StopIteration:
            raise XYZError('ase.io.extxyz: Frame has {} atoms, expected {}'
                           .format(len(data), natoms))
        vals = line.split()
        row = tuple([conv(val) for conv, val in zip(convs, vals)])
        data.append(row)

    try:
        data = np.array(data, dtype)
    except TypeError:
        raise XYZError('Badly formatted data '
                       'or end of file reached before end of frame')

    # Read VEC entries if present
    if nvec > 0:
        for ln in range(nvec):
            try:
                line = next(lines)
            except StopIteration:
                raise XYZError('ase.io.adfxyz: Frame has {} cell vectors, '
                               'expected {}'.format(len(cell), nvec))
            entry = line.split()

            if not entry[0].startswith('VEC'):
                raise XYZError('Expected cell vector, got {}'.format(entry[0]))

            try:
                n = int(entry[0][3:])
            except ValueError as e:
                raise XYZError('Expected VEC{}, got VEC{}'
                               .format(ln + 1, entry[0][3:])) from e
            if n != ln + 1:
                raise XYZError('Expected VEC{}, got VEC{}'
                               .format(ln + 1, n))

            cell[ln] = np.array([float(x) for x in entry[1:]])
            pbc[ln] = True
        if nvec != pbc.count(True):
            raise XYZError('Problem with number of cell vectors')
        pbc = tuple(pbc)

    arrays = {}
    for name in names:
        ase_name, cols = properties[name]
        if cols == 1:
            value = data[name]
        else:
            value = np.vstack([data[name + str(c)]
                               for c in range(cols)]).T
        arrays[ase_name] = value

    symbols = None
    if 'symbols' in arrays:
        symbols = [s.capitalize() for s in arrays['symbols']]
        del arrays['symbols']

    numbers = None
    duplicate_numbers = None
    if 'numbers' in arrays:
        if symbols is None:
            numbers = arrays['numbers']
        else:
            duplicate_numbers = arrays['numbers']
        del arrays['numbers']

    initial_charges = None
    if 'initial_charges' in arrays:
        initial_charges = arrays['initial_charges']
        del arrays['initial_charges']


    positions = None
    if 'positions' in arrays:
        positions = arrays['positions']
        del arrays['positions']

    atoms = Atoms(symbols=symbols,
                  positions=positions,
                  numbers=numbers,
                  charges=initial_charges,
                  cell=cell,
                  pbc=pbc,
                  info=info)

    # Read and set constraints
    if 'move_mask' in arrays:
        if properties['move_mask'][1] == 3:
            cons = []
            for a in range(natoms):
                cons.append(FixCartesian(a, mask=~arrays['move_mask'][a, :]))
            atoms.set_constraint(cons)
        elif properties['move_mask'][1] == 1:
            atoms.set_constraint(FixAtoms(mask=~arrays['move_mask']))
        else:
            raise XYZError('Not implemented constraint')
        del arrays['move_mask']

    for name, array in arrays.items():
        atoms.new_array(name, array)

    if duplicate_numbers is not None:
        atoms.set_atomic_numbers(duplicate_numbers)

    # Load results of previous calculations into SinglePointCalculator
    results = {}
    for key in list(atoms.info.keys()):
        if key in per_config_properties:
            results[key] = atoms.info[key]
            # special case for stress- convert to Voigt 6-element form
            if key == 'stress' and results[key].shape == (3, 3):
                stress = results[key]
                stress = np.array([stress[0, 0],
                                   stress[1, 1],
                                   stress[2, 2],
                                   stress[1, 2],
                                   stress[0, 2],
                                   stress[0, 1]])
                results[key] = stress
    for key in list(atoms.arrays.keys()):
        if (key in per_atom_properties and len(value.shape) >= 1
            and value.shape[0] == len(atoms)):
            results[key] = atoms.arrays[key]
    if results != {}:
        calculator = SinglePointCalculator(atoms, **results)
        atoms.calc = calculator
    return atoms


class XYZError(IOError):
    pass


class XYZChunk:
    def __init__(self, lines, natoms):
        self.lines = lines
        self.natoms = natoms

    def build(self):
        """Convert unprocessed chunk into Atoms."""
        return _read_xyz_frame(iter(self.lines), self.natoms)


def ixyzchunks(fd):
    """Yield unprocessed chunks (header, lines) for each xyz image."""
    while True:
        line = next(fd).strip()  # Raises StopIteration on empty file
        try:
            natoms = int(line)
        except ValueError:
            raise XYZError('Expected integer, found "{0}"'.format(line))
        try:
            lines = [next(fd) for _ in range(1 + natoms)]
        except StopIteration:
            raise XYZError('Incomplete XYZ chunk')
        yield XYZChunk(lines, natoms)


class ImageIterator:
    """"""

    def __init__(self, ichunks):
        self.ichunks = ichunks

    def __call__(self, fd, indices=-1):
        if not hasattr(indices, 'start'):
            if indices < 0:
                indices = slice(indices - 1, indices)
            else:
                indices = slice(indices, indices + 1)

        for chunk in self._getslice(fd, indices):
            yield chunk.build()

    def _getslice(self, fd, indices):
        try:
            iterator = islice(self.ichunks(fd), indices.start, indices.stop,
                              indices.step)
        except ValueError:
            # Negative indices.  Go through the whole thing to get the length,
            # which allows us to evaluate the slice, and then read it again
            startpos = fd.tell()
            nchunks = 0
            for chunk in self.ichunks(fd):
                nchunks += 1
            fd.seek(startpos)
            indices_tuple = indices.indices(nchunks)
            iterator = islice(self.ichunks(fd), *indices_tuple)
        return iterator


iread_xyz = ImageIterator(ixyzchunks)


@reader
def read_xyz(fileobj, index=-1, properties_parser=key_val_str_to_dict):
    r"""
    Read from a file in Extended XYZ format

    index is the frame to read, default is last frame (index=-1).
    properties_parser is the parse to use when converting the properties line
    to a dictionary, ``extxyz.key_val_str_to_dict`` is the default and can
    deal with most use cases, ``extxyz.key_val_str_to_dict_regex`` is slightly
    faster but has fewer features.

    Extended XYZ format is an enhanced version of the `basic XYZ format
    <http://en.wikipedia.org/wiki/XYZ_file_format>`_ that allows extra
    columns to be present in the file for additonal per-atom properties as
    well as standardising the format of the comment line to include the
    cell lattice and other per-frame parameters.

    It's easiest to describe the format with an example.  Here is a
    standard XYZ file containing a bulk cubic 8 atom silicon cell ::

        8
        Cubic bulk silicon cell
        Si          0.00000000      0.00000000      0.00000000
        Si        1.36000000      1.36000000      1.36000000
        Si        2.72000000      2.72000000      0.00000000
        Si        4.08000000      4.08000000      1.36000000
        Si        2.72000000      0.00000000      2.72000000
        Si        4.08000000      1.36000000      4.08000000
        Si        0.00000000      2.72000000      2.72000000
        Si        1.36000000      4.08000000      4.08000000

    The first line is the number of atoms, followed by a comment and
    then one line per atom, giving the element symbol and cartesian
    x y, and z coordinates in Angstroms.

    Here's the same configuration in extended XYZ format ::

        8
        Lattice="5.44 0.0 0.0 0.0 5.44 0.0 0.0 0.0 5.44" Properties=species:S:1:pos:R:3 Time=0.0
        Si        0.00000000      0.00000000      0.00000000
        Si        1.36000000      1.36000000      1.36000000
        Si        2.72000000      2.72000000      0.00000000
        Si        4.08000000      4.08000000      1.36000000
        Si        2.72000000      0.00000000      2.72000000
        Si        4.08000000      1.36000000      4.08000000
        Si        0.00000000      2.72000000      2.72000000
        Si        1.36000000      4.08000000      4.08000000

    In extended XYZ format, the comment line is replaced by a series of
    key/value pairs.  The keys should be strings and values can be
    integers, reals, logicals (denoted by `T` and `F` for true and false)
    or strings. Quotes are required if a value contains any spaces (like
    `Lattice` above).  There are two mandatory parameters that any
    extended XYZ: `Lattice` and `Properties`. Other parameters --
    e.g. `Time` in the example above --- can be added to the parameter line
    as needed.

    `Lattice` is a Cartesian 3x3 matrix representation of the cell
    vectors, with each vector stored as a column and the 9 values listed in
    Fortran column-major order, i.e. in the form ::

      Lattice="R1x R1y R1z R2x R2y R2z R3x R3y R3z"

    where `R1x R1y R1z` are the Cartesian x-, y- and z-components of the
    first lattice vector (:math:`\mathbf{a}`), `R2x R2y R2z` those of the second
    lattice vector (:math:`\mathbf{b}`) and `R3x R3y R3z` those of the
    third lattice vector (:math:`\mathbf{c}`).

    The list of properties in the file is described by the `Properties`
    parameter, which should take the form of a series of colon separated
    triplets giving the name, format (`R` for real, `I` for integer) and
    number of columns of each property. For example::

      Properties="species:S:1:pos:R:3:vel:R:3:select:I:1"

    indicates the first column represents atomic species, the next three
    columns represent atomic positions, the next three velcoities, and the
    last is an single integer called `select`. With this property
    definition, the line ::

      Si        4.08000000      4.08000000      1.36000000   0.00000000      0.00000000      0.00000000       1

    would describe a silicon atom at position (4.08,4.08,1.36) with zero
    velocity and the `select` property set to 1.

    The property names `pos`, `Z`, `mass`, and `charge` map to ASE
    :attr:`ase.atoms.Atoms.arrays` entries named
    `positions`, `numbers`, `masses` and `charges` respectively.

    Additional key-value pairs in the comment line are parsed into the
    :attr:`ase.Atoms.atoms.info` dictionary, with the following conventions

     - Values can be quoted with `""`, `''`, `[]` or `{}` (the latter are
       included to ease command-line usage as the `{}` are not treated
       specially by the shell)
     - Quotes within keys or values can be escaped with `\"`.
     - Keys with special names `stress` or `virial` are treated as 3x3 matrices
       in Fortran order, as for `Lattice` above.
     - Otherwise, values with multiple elements are treated as 1D arrays, first
       assuming integer format and falling back to float if conversion is
       unsuccessful.
     - A missing value defaults to `True`, e.g. the comment line
       `"cutoff=3.4 have_energy"` leads to
       `{'cutoff': 3.4, 'have_energy': True}` in `atoms.info`.
     - Value strings starting with `"_JSON"` are interpreted as JSON content;
       similarly, when writing, anything which does not match the criteria above
       is serialised as JSON.

    The extended XYZ format is also supported by the
    the `Ovito <http://www.ovito.org>`_ visualisation tool
    (from `v2.4 beta
    <http://www.ovito.org/index.php/component/content/article?id=25>`_
    onwards).
    """  # noqa: E501

    if not isinstance(index, int) and not isinstance(index, slice):
        raise TypeError('Index argument is neither slice nor integer!')

    # If possible, build a partial index up to the last frame required
    last_frame = None
    if isinstance(index, int) and index >= 0:
        last_frame = index
    elif isinstance(index, slice):
        if index.stop is not None and index.stop >= 0:
            last_frame = index.stop

    # scan through file to find where the frames start
    try:
        fileobj.seek(0)
    except UnsupportedOperation:
        fileobj = StringIO(fileobj.read())
        fileobj.seek(0)
    frames = []
    while True:
        frame_pos = fileobj.tell()
        line = fileobj.readline()
        if line.strip() == '':
            break
        try:
            natoms = int(line)
        except ValueError as err:
            raise XYZError('ase.io.extxyz: Expected xyz header but got: {}'
                           .format(err))
        fileobj.readline()  # read comment line
        for i in range(natoms):
            fileobj.readline()
        # check for VEC
        nvec = 0
        while True:
            lastPos = fileobj.tell()
            line = fileobj.readline()
            if line.lstrip().startswith('VEC'):
                nvec += 1
                if nvec > 3:
                    raise XYZError('ase.io.extxyz: More than 3 VECX entries')
            else:
                fileobj.seek(lastPos)
                break
        frames.append((frame_pos, natoms, nvec))
        if last_frame is not None and len(frames) > last_frame:
            break

    trbl = index2range(index, len(frames))

    for index in trbl:
        frame_pos, natoms, nvec = frames[index]
        fileobj.seek(frame_pos)
        # check for consistency with frame index table
        assert int(fileobj.readline()) == natoms
        yield _read_xyz_frame(fileobj, natoms, properties_parser, nvec)


def output_column_format(atoms, columns, arrays,
                         write_info=True, results=None):
    """
    Helper function to build extended XYZ comment line
    """
    fmt_map = {'d': ('R', '%16.8f'),
               'f': ('R', '%16.8f'),
               'i': ('I', '%8d'),
               'O': ('S', '%s'),
               'S': ('S', '%s'),
               'U': ('S', '%-2s'),
               'b': ('L', ' %.1s')}

    # NB: Lattice is stored as tranpose of ASE cell,
    # with Fortran array ordering
    lattice_str = ('Lattice="'
                   + ' '.join([str(x) for x in np.reshape(atoms.cell.T,
                                                          9, order='F')]) +
                   '"')

    property_names = []
    property_types = []
    property_ncols = []
    dtypes = []
    formats = []

    for column in columns:
        array = arrays[column]
        dtype = array.dtype

        property_name = PROPERTY_NAME_MAP.get(column, column)
        property_type, fmt = fmt_map[dtype.kind]
        property_names.append(property_name)
        property_types.append(property_type)

        if (len(array.shape) == 1
                or (len(array.shape) == 2 and array.shape[1] == 1)):
            ncol = 1
            dtypes.append((column, dtype))
        else:
            ncol = array.shape[1]
            for c in range(ncol):
                dtypes.append((column + str(c), dtype))

        formats.extend([fmt] * ncol)
        property_ncols.append(ncol)

    props_str = ':'.join([':'.join(x) for x in
                          zip(property_names,
                              property_types,
                              [str(nc) for nc in property_ncols])])

    comment_str = ''
    if atoms.cell.any():
        comment_str += lattice_str + ' '
    comment_str += 'Properties={}'.format(props_str)

    info = {}
    if write_info:
        info.update(atoms.info)
    if results is not None:
        info.update(results)
    info['pbc'] = atoms.get_pbc()  # always save periodic boundary conditions
    comment_str += ' ' + key_val_dict_to_str(info)

    dtype = np.dtype(dtypes)
    fmt = ' '.join(formats) + '\n'

    return comment_str, property_ncols, dtype, fmt


def write_xyz(fileobj, images, comment='', columns=None,
              write_info=True,
              write_results=True, plain=False, vec_cell=False,
              append=False):
    """
    Write output in extended XYZ format

    Optionally, specify which columns (arrays) to include in output,
    whether to write the contents of the `atoms.info` dict to the
    XYZ comment line (default is True), the results of any
    calculator attached to this Atoms. The `plain` argument
    can be used to write a simple XYZ file with no additional information.
    `vec_cell` can be used to write the cell vectors as additional
    pseudo-atoms. If `append` is set to True, the file is for append (mode `a`),
    otherwise it is overwritten (mode `w`).

    See documentation for :func:`read_xyz()` for further details of the extended
    XYZ file format.
    """
    if isinstance(fileobj, str):
        mode = 'w'
        if append:
            mode = 'a'
        fileobj = paropen(fileobj, mode)

    if hasattr(images, 'get_positions'):
        images = [images]

    for atoms in images:
        natoms = len(atoms)

        if columns is None:
            fr_cols = None
        else:
            fr_cols = columns[:]

        if fr_cols is None:
            fr_cols = (['symbols', 'positions']
                       + [key for key in atoms.arrays.keys() if
                          key not in ['symbols', 'positions', 'numbers',
                                      'species', 'pos']])

        if vec_cell:
            plain = True

        if plain:
            fr_cols = ['symbols', 'positions']
            write_info = False
            write_results = False

        per_frame_results = {}
        per_atom_results = {}
        if write_results:
            calculator = atoms.calc
            if (calculator is not None
                    and isinstance(calculator, Calculator)):
                for key in all_properties:
                    value = calculator.results.get(key, None)
                    if value is None:
                        # skip missing calculator results
                        continue
                    if (key in per_atom_properties and len(value.shape) >= 1
                        and value.shape[0] == len(atoms)):
                        # per-atom quantities (forces, energies, stresses)
                        per_atom_results[key] = value
                    elif key in per_config_properties:
                        # per-frame quantities (energy, stress)
                        # special case for stress, which should be converted
                        # to 3x3 matrices before writing
                        if key == 'stress':
                            xx, yy, zz, yz, xz, xy = value
                            value = np.array(
                                [(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
                        per_frame_results[key] = value

        # Move symbols and positions to first two properties
        if 'symbols' in fr_cols:
            i = fr_cols.index('symbols')
            fr_cols[0], fr_cols[i] = fr_cols[i], fr_cols[0]

        if 'positions' in fr_cols:
            i = fr_cols.index('positions')
            fr_cols[1], fr_cols[i] = fr_cols[i], fr_cols[1]

        # Check first column "looks like" atomic symbols
        if fr_cols[0] in atoms.arrays:
            symbols = atoms.arrays[fr_cols[0]]
        else:
            symbols = atoms.get_chemical_symbols()

        if natoms > 0 and not isinstance(symbols[0], str):
            raise ValueError('First column must be symbols-like')

        # Check second column "looks like" atomic positions
        pos = atoms.arrays[fr_cols[1]]
        if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
            raise ValueError('Second column must be position-like')

        # if vec_cell add cell information as pseudo-atoms
        if vec_cell:
            pbc = list(atoms.get_pbc())
            cell = atoms.get_cell()

            if True in pbc:
                nPBC = 0
                for i, b in enumerate(pbc):
                    if b:
                        nPBC += 1
                        symbols.append('VEC' + str(nPBC))
                        pos = np.vstack((pos, cell[i]))
                # add to natoms
                natoms += nPBC
                if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
                    raise ValueError(
                        'Pseudo Atoms containing cell have bad coords')

        # Move mask
        if 'move_mask' in fr_cols:
            cnstr = images[0]._get_constraints()
            if len(cnstr) > 0:
                c0 = cnstr[0]
                if isinstance(c0, FixAtoms):
                    cnstr = np.ones((natoms,), dtype=bool)
                    for idx in c0.index:
                        cnstr[idx] = False
                elif isinstance(c0, FixCartesian):
                    masks = np.ones((natoms, 3), dtype=bool)
                    for i in range(len(cnstr)):
                        idx = cnstr[i].a
                        masks[idx] = cnstr[i].mask
                    cnstr = masks
            else:
                fr_cols.remove('move_mask')

        # Collect data to be written out
        arrays = {}
        for column in fr_cols:
            if column == 'positions':
                arrays[column] = pos
            elif column in atoms.arrays:
                arrays[column] = atoms.arrays[column]
            elif column == 'symbols':
                arrays[column] = np.array(symbols)
            elif column == 'move_mask':
                arrays[column] = cnstr
            else:
                raise ValueError('Missing array "%s"' % column)

        if write_results:
            for key in per_atom_results:
                if key not in fr_cols:
                    fr_cols += [key]
                else:
                    warnings.warn('write_xyz() overwriting array "{0}" present '
                                  'in atoms.arrays with stored results '
                                  'from calculator'.format(key))
            arrays.update(per_atom_results)

        comm, ncols, dtype, fmt = output_column_format(atoms,
                                                       fr_cols,
                                                       arrays,
                                                       write_info,
                                                       per_frame_results)

        if plain or comment != '':
            # override key/value pairs with user-speficied comment string
            comm = comment.rstrip()
            if '\n' in comm:
                raise ValueError('Comment line should not have line breaks.')

        # Pack fr_cols into record array
        data = np.zeros(natoms, dtype)
        for column, ncol in zip(fr_cols, ncols):
            value = arrays[column]
            if ncol == 1:
                data[column] = np.squeeze(value)
            else:
                for c in range(ncol):
                    data[column + str(c)] = value[:, c]

        nat = natoms
        if vec_cell:
            nat -= nPBC
        # Write the output
        fileobj.write('%d\n' % nat)
        fileobj.write('%s\n' % comm)
        for i in range(natoms):
            fileobj.write(fmt % tuple(data[i]))


# create aliases for read/write functions
read_extxyz = read_xyz
write_extxyz = write_xyz
