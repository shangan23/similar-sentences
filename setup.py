import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="similar-sentences",
    version="1.4",
    author="Shankar Ganesh Jayaraman",
    author_email="shangan.23@gmail.com",
    description="Similar sentence prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shangan23/similar-sentences",
    packages=setuptools.find_packages(),
    install_requires=[
        "sentence-transformers"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
