from setuptools import setup, find_packages


packages = find_packages()


packages.append("why")


setup(
    name="Why-XAI",
    version="0.1.0",
    description="Framework-agnostic CNN XAI Library",
    url="https://github.com/mburaksayici",
    packages=packages,
    author="Mehmet Burak Sayıcı",
    author_email="mburaksayici@gmail.com",
    license="BSD 2-clause",
    install_requires=["tensorflow", "numpy", "torch", "torchvision", "opencv-python", "matplotlib"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
