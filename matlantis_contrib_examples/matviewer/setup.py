import os
from typing import List, Dict

from setuptools import find_packages, setup  # NOQA

setup_requires: List[str] = []
install_requires: List[str] = [
    "pfcc-extras",
    "torch",
    "cython",
    "scikit-learn",
    "autode",
]

__version__: str
here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
exec(open(os.path.join(here, "matviewer", "_version.py")).read())

package_data = {"matviewer": ["data/*"]}

setup(
    name="matviewer",
    version=__version__,  # NOQA
    description="Viewer for Matlantis that can edit and calculate with GUI",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=install_requires,
    include_package_data=True,
    package_data=package_data,
)
