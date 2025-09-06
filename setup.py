#!/usr/bin/env python
"""
EEG2025 Challenge Setup
=======================

Setup script for the EEG Foundation Model Challenge 2025.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_file = os.path.join(os.path.dirname(__file__), 'src', '__init__.py')
    if os.path.exists(init_file):
        with open(init_file) as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

# Read README for long description
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def get_requirements():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='eeg2025',
    version=get_version(),
    description='EEG Foundation Model for 2025 Challenge',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='EEG2025 Team',
    author_email='team@eeg2025.challenge',
    url='https://github.com/eeg2025/foundation-model',

    # Package configuration
    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    # Include non-Python files
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.txt', '*.md'],
    },

    # Dependencies
    install_requires=get_requirements(),

    # Optional dependencies
    extras_require={
        'gpu': [
            'triton>=2.0.0',
            'cupy-cuda11x>=11.0.0',
        ],
        'demo': [
            'fastapi>=0.68.0',
            'uvicorn[standard]>=0.15.0',
            'pydantic>=1.8.0',
            'requests>=2.25.0',
        ],
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.12',
            'black>=21.0',
            'isort>=5.9',
            'flake8>=3.9',
            'mypy>=0.910',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
            'myst-parser>=0.15',
        ],
        'all': [
            'triton>=2.0.0',
            'cupy-cuda11x>=11.0.0',
            'fastapi>=0.68.0',
            'uvicorn[standard]>=0.15.0',
            'pydantic>=1.8.0',
            'requests>=2.25.0',
            'pytest>=6.0',
            'pytest-cov>=2.12',
            'black>=21.0',
            'isort>=5.9',
            'flake8>=3.9',
            'mypy>=0.910',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
            'myst-parser>=0.15',
        ]
    },

    # Entry points
    entry_points={
        'console_scripts': [
            'eeg2025-train=scripts.train:main',
            'eeg2025-inference=scripts.inference:main',
            'eeg2025-benchmark=scripts.bench_inference:main',
            'eeg2025-validate=scripts.validate_repository:main',
            'eeg2025-health-check=scripts.health_check:main',
        ],
    },

    # Metadata
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    # Requirements
    python_requires='>=3.8',

    # Project URLs
    project_urls={
        'Documentation': 'https://eeg2025.readthedocs.io/',
        'Source': 'https://github.com/eeg2025/foundation-model',
        'Tracker': 'https://github.com/eeg2025/foundation-model/issues',
    },

    # Keywords
    keywords=[
        'eeg', 'electroencephalography', 'deep-learning', 'pytorch',
        'foundation-model', 'self-supervised-learning', 'domain-adaptation',
        'neuroscience', 'brain-computer-interface', 'signal-processing'
    ],
)
