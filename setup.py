from setuptools import find_packages, setup


with open("README.md","r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

with open("requirements.txt", "r") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name="chinese_keybert",
    version="0.1.0",
    author="Jaackson Cakes",
    author_email="jacksonkek257@gmail.com",
    description="Chinese keyword extraction using transformer-based language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JacksonCakes/chinese_keybert",
    packages=find_packages(exclude=["docs", "tests","examples","chinese_keywords_extraction"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
)