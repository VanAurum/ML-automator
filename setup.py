import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mlautomator",
    version="1.0.6",
    author="Kevin Vecmanis",
    author_email="kevinvecmanis@gmail.com",
    description="A fast, simple way to train machine learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VanAurum/ML-automator",
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
        'hyperopt',
        'xgboost',
    ]
)
