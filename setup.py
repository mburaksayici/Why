from setuptools import setup

setup(
    name="Why",
    version="0.1.0",
    description="A example Python package",
    url="https://github.com/mburaksayici",
    author="Mehmet Burak Sayıcı",
    author_email="burak@gesund.ai",
    license="BSD 2-clause",
    packages=["why"],
    install_requires=["tensorflow", "numpy", "torch", "torchvision", "opencv-python"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
