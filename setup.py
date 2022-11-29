import re
from pathlib import Path
from setuptools import setup, find_packages


def stream_requirements(fd):
    """For a given requirements file descriptor, generate lines of
    distribution requirements, ignoring comments and chained requirement
    files.
    """
    for line in fd:
        cleaned = re.sub(r'#.*$', '', line).strip()
        if cleaned and not cleaned.startswith('-r'):
            yield cleaned


def load_requirements(txt_path):
    """Short helper for loading requirements from a .txt file.

    Parameters
    ----------
    txt_path : Path or str
        Path to the requirements file.

    Returns
    -------
    list
        List of requirements, one list element per line in the text file.
    """
    with Path(txt_path).open() as requirements_file:
        return list(stream_requirements(requirements_file))


# ---------------------------------------------------------------------------- #
#                                   Requirements                               #
# ---------------------------------------------------------------------------- #

ROOT_PATH = Path(__file__).parent
README_PATH = ROOT_PATH / 'README.md'

REQUIREMENTS_PATH = ROOT_PATH / 'requirements' / 'main.txt'
REQUIREMENTS_TEST_PATH = ROOT_PATH / 'requirements' / 'test.txt'
REQUIREMENTS_PLOTTING_PATH = ROOT_PATH / 'requirements' / 'plotting.txt'

requirements = load_requirements(REQUIREMENTS_PATH)
requirements_test = load_requirements(REQUIREMENTS_TEST_PATH)
requirements_plotting = load_requirements(REQUIREMENTS_PLOTTING_PATH)


# ---------------------------------------------------------------------------- #
#                                   Version                                    #
# ---------------------------------------------------------------------------- #
SRC_PATH = ROOT_PATH / 'src' / 'hpt'
VERSION_PATH = SRC_PATH / 'version.py'

with VERSION_PATH.open('rb') as version_file:
    exec(version_file.read())


# ---------------------------------------------------------------------------- #
#                                   SETUP                                      #
# ---------------------------------------------------------------------------- #
setup(
    name="hyperparameter-tuning",
    version=__version__,
    description="A minimal framework for running hyperparameter tuning",
    keywords=["ml", "optimization", "hyperparameter", "tuning", "fairness"],

    long_description=(README_PATH).read_text(),
    long_description_content_type="text/markdown",

    package_dir={'': 'src'},
    packages=find_packages('src', exclude=['tests', 'tests.*']),
    package_data={
        '': ['*.yaml, *.yml'],
    },
    include_package_data=True,

    python_requires='>=3.8',

    install_requires=requirements,

    extras_require={
        'plotting': requirements_plotting,
        'testing': requirements_test,
        'all': requirements_plotting + requirements_test,
        # NOTE: remember to add extra requirements to 'all' as well.
    },

    zip_safe=False,

    test_suite='tests',     # TODO: check this parameter
    tests_require=requirements_test,

    author="AndreFCruz",
    url="https://github.com/AndreFCruz/hpt",

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
