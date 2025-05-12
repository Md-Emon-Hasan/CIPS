from setuptools import setup, find_packages

setup(
    name="CIPS", 
    version="0.1.0",
    description="Cricket IPL Prediction System",
    author="Md. Emon Hasan",
    author_email="iconicemon01@gmail.com",
    packages=find_packages(),
    install_requires=[
        "requests",
        "scikit-learn",
        "flask",
        "gunicorn",
        "pandas"
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: Flask",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
