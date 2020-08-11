from setuptools import setup

setup(name='fets',
      version='0.0.0',
      packages=['fets',
                'fets.models',
                'fets.models.pytorch',
                'fets.models.pytorch.pt_3dresunet',
                'fets.models.pytorch.brainmage',
                'fets.data',
                'fets.data.pytorch'],
      install_requires=['torch==1.6.0', 'protobuf', 'pyyaml', 'grpcio', 'tqdm', 'coloredlogs', 'nibabel', 'sklearn', 'SimpleITK', 'pandas']
)
