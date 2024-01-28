from setuptools import setup, find_packages

setup(
    name='SpeechStew',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A speech recognition package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/SpeechStew',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio',
        'numpy',
        'librosa',
        'Levenshtein',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
