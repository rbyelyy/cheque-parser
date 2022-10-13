from setuptools import setup

setup(
    name='parse-bills',
    version='1.0',
    packages=['post_bill', 'crack_captcha'],
    url='',
    license='MIT',
    author='Roman Byelyy',
    author_email='rbyelyy@gmail.com',
    description='Post bill tool', install_requires=['selenium', 'mysql', 'Pillow', 'numpy', 'mysql-connector']
)
