from setuptools import setup, find_packages
exec(open('imagen_pytorch/version.py').read())

setup(
  name = 'imagen-pytorch',
  packages = find_packages(exclude=[]),
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
    'click',
    'einops>=0.4',
    'einops-exts',
    'numpy',
    'pillow',
    'pydantic',
    'pytorch-warmup',
    'resize-right',
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
