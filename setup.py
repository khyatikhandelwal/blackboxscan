from setuptools import setup, find_packages

setup(
    name="blackboxscan",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "blackboxscan=blackboxscan.scanner:main",
        ],
    },
    author="Khyati Khandelwal",
    author_email="connect@khyatikhandelwal.com",
    description="A library to easily analyse the outputs of HuggingFace LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/khyatikhandelwal/blackboxscan",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
