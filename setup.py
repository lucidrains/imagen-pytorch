from setuptools import setup, find_packages
exec(open('imagen_pytorch/version.py').read())

setup(
  name = 'imagen-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  entry_points={
    'console_scripts': [
      'imagen_pytorch = imagen_pytorch.cli:main',
      'imagen = imagen_pytorch.cli:imagen'
    ],
  },
  version = __version__,
  license='MIT',
  description = 'Imagen - unprecedented photorealism × deep level of language understanding',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/imagen-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'text-to-image',
    'denoising-diffusion'
  ],
  install_requires=[
    'accelerate>=0.23.0',
    'beartype',
    'click',
    'datasets',
    'einops>=0.7.0',
    'ema-pytorch>=0.0.3',
    'fsspec',
    'kornia',
    'numpy',
    'packaging',
    'pillow',
    'pydantic>=2',
    'pytorch-warmup',
    'sentencepiece',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
