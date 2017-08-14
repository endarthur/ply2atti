from setuptools import setup

setup(
    name="ply2atti",
    version="0.1.0",
    py_modules=['ply2atti'],
    entry_points={
        'console_scripts': [
            'ply2atti = ply2atti:main',
        ]
    },

    install_requires=["numpy", "networkx"],

    # metadata for upload to PyPI
    author="Arthur Endlein",
    author_email="endarthur@gmail.com",
    description="Script for extraction of attitudes from 3d models using meshlab's mesh painting tool",
    license="MIT",
    keywords="geology attitudes meshlab",
    url="https://github.com/endarthur/ply2atti",
    dowload_url= "https://github.com/endarthur/ply2atti/archive/0.1.0.tar.gz",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)
