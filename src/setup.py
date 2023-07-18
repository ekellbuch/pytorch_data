import os

import setuptools

loc = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
    name="pytorch_data",
    description="pytorch_data",
    long_description="",
    long_description_content_type="test/markdown",
    url="https://github.com/ekellbuch/pytorch_data",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={},
    classifiers=["License :: OSI Approved :: MIT License"],
    python_requires=">=3.7",
)
