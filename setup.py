from importlib.metadata import entry_points
from setuptools import setup
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

def get_requirements(path: str):
    return [l.strip() for l in open(path)]

setup (
    name = 'EVRP Project',
    version = '0.1',
    description = 'Electric Vehicle Routing Problem',
    long_description = long_description,
    long_description_content_type="text/markdown", 
    author = 'Vu Quoc Hien',
    author_email = 'hienvq23@gmail.com',
    url = 'https://github.com/NeiH4207/EVRP-Python',
    packages=find_packages(),
    keywords='',
    install_requires=get_requirements("requirements.txt"),
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
        ]
    },
    
)