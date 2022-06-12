import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calibration",
    version="1.0",
    author="Saurabh Garg",
    author_email="garg.saurabh.2014@gmail.com",
    description="Utilities to calibrate model and measure calibration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saurabhgarg1996/calibration",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'torch', 'parameterized'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
