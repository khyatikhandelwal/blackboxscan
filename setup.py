from setuptools import setup, find_packages

setup(
    name='blackboxscan',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'your-command=your_library_name.your_module:main',
        ],
    },
    author='Khyati Khandelwal',
    author_email='connect@khyatikhandelwal.com',
    description='A library to easily analyse the outputs of HuggingFace LLMs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/khyatikhandelwal/blackboxscan',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)