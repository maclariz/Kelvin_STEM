from setuptools import setup, find_packages
from distutils.util import convert_path

with open("README.md", "r") as f:
    long_description = f.read()

version_ns = {}
vpath = convert_path("Kelvin/version.py")
with open(vpath) as version_file:
    exec(version_file.read(), version_ns)

setup(
    name="Kelvin_STEM",
    version=version_ns["__version__"],
    packages=find_packages(),
    description="Tools for 4D STEM data processing from the Kelvin Nanocharacterisation Centre, University of Glasgow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maclariz/Kelvin_STEM/",
    author="Ian MacLarem",
    author_email="ian.maclaren@glasgow.ac.uk",
    license="MIT",
    keywords="STEM,4DSTEM",
    python_requires=">=3.8",
    install_requires=[
        "numpy >= 1.19",
        "scipy >= 1.5.2",
        "h5py >= 3.2.0",
        "hdf5plugin >= 4.1.3",
        "matplotlib >= 3.2.2",
        "tqdm >= 4.46.1",
    ],
)
