from setuptools import setup, find_packages

setup(
  name = 'imagen-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.5',
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
    'einops>=0.4',
    'einops-exts',
    'kornia',
    'resize-right',
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
