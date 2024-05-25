from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()


setup(
    name="mambular",  # Replace with your package's name
    version="0.1.2",  # The current version of your package
    author="Anton Thielmann",  # Replace with your name
    author_email="anton.thielmann@tu-clausthal.de",  # Replace with your email
    description="A python package for tabular deep learning with mamba blocks",  # A short description of your package
    long_description_content_type="text/markdown",
    url="https://github.com/AnFreTh/mamba-tabular",  # Replace with the URL to your package's repository
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6, <3.11",
    install_requires=read_requirements(),
)
