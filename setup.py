import setuptools
import sizif

REQUIRED_PACKAGES = [
    'numpy >= 1.12',
    'Keras >= 2.2'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sizif",
    version=pyagender.VERSION,
    author="Michael Butlitsky",
    author_email="aristofun@yandex.ru",
    description="Deep learning Keras models lifecycle management backup/restore nano framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aristofun/sizif",
    packages=setuptools.find_packages(),
    classifiers=(
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        "Environment :: Console",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    zip_safe=False,
    install_requires=REQUIRED_PACKAGES,
    # entry_points={'console_scripts': CONSOLE_SCRIPTS}
)
