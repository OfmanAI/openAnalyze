# setup.py
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf8") if (ROOT/"README.md").exists() else ""

setup(
    name="openAnalyze",
    version="0.1.0",
    description="CLI that glues DeepMedia detector CSV → full analytics bundle",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "scipy",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "openAnalyse = openAnalyze.openAnalyze:main",
            "openAnalyze-gui = openAnalyze.openAnalyze_gui:launch"


        ]
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

