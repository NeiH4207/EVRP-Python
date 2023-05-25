from importlib.metadata import entry_points
from setuptools import setup
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup (
    name = 'VRP Project',
    version = '0.1',
    description = 'VRP stands for Vehicle Routing Problem, which is a well-known optimization problem in the field of operations research and logistics. ',
    long_description = long_description,
    long_description_content_type="text/markdown", 
    author = 'Vu Quoc Hien',
    author_email = 'hienvq.2000@gmail.com',
    url = 'https://github.com/NeiH4207/VRP-Project',
    packages=[],
    keywords='',
    install_requires=[
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
        ]
    },
    
)