"""Functions to export information to data files."""

from uuid import uuid4 as _uuid
from ruamel.yaml import YAML as _yaml


def layout2yaml(layout, site, name, filename):
    """Write layout for a given siye to a YAML file

    layout
        layout from a wflopg.Owflop object
    site : str
        site name
    name : str
        layout name
    filename : str
        file to write to

    """
    output = {}
    output['name'] = name
    output['uuid'] = str(_uuid())
    output['site'] = site
    output['layout'] = layout.values.tolist()
    with open(filename, 'w') as f:
        _yaml(typ='safe').dump(output, f)
