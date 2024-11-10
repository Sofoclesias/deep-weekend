"""
https://github.com/Sofoclesias/deep-weekend
"""

from setuptools import setup
import codecs

with codecs.open('README.md','r',encoding='utf-8') as f:
    readme = f.read()

setup(
    name='deep-weekend',
    version="0.0.1",
    description="Hola, profesor.",
    long_description=readme,
    packages=['codes']
)