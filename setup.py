from setuptools import setup

setup(name='fets',
      version='0.0.2',
      packages=['fets',
                'fets.models',
                'fets.models.pytorch',
                'fets.models.pytorch.pt_3dresunet',
                'fets.models.pytorch.brainmage',
                'fets.models.pytorch.deepscan', 
                'fets.models.pytorch.nnunet',
                'fets.data',
                'fets.data.pytorch'],
      install_requires=['protobuf', 'pyyaml', 'grpcio', 'tqdm', 'coloredlogs', 'nibabel', 'sklearn', 'SimpleITK', 'pandas', 'nnUNet==1.6.6', 'batchgenerators==0.21', 'opencv-python']
)
