from setuptools import setup, find_packages
import setuptools
import warnings
def read_requirements():
    try:
        with open('requirements.txt', encoding='utf-8-sig') as req:
            return [line.strip() for line in req if line.strip() and not line.startswith('#')]
    except UnicodeDecodeError:
        with open('requirements.txt', encoding='utf-16') as req:
            return [line.strip() for line in req if line.strip() and not line.startswith('#')]
setup(
    name='nwb4fp',
    version='0.8.0.2',
    url='https://github.com/sachuriga/QuattrocoloLab-nwb4fp',
    author='sachuriga',
    author_email='sachuriga.sachuriga@ntnu.no',
    packages=find_packages(where="src"),  # Automatically find packages in src/
    package_dir={"": "src"},    # Tell setuptools to look in src/
    include_package_data=True,
    description='Description of my package',
    #packages=find_packages(./src/),    
    install_requires=read_requirements(),
)

if __name__ == "__main__":
    ## pip list --format=freeze > requirements.txt
    setuptools.setup()

# python setup.py sdist bdist_wheel
# twine upload dist/*