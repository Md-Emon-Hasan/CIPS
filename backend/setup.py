from setuptools import setup, find_packages

setup(
    name="cips-backend",
    version="0.1.0",
    description="Cricket IPL Prediction System Backend",
    author="Md Emon Hasan",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.115.8",
        "uvicorn==0.34.0",
        "sqlmodel==0.0.24",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "xgboost==3.0.0",
        "requests==2.32.5",
        "python-multipart==0.0.20",
        "pydantic-settings==2.10.1",
    ],
    url="https://github.com/Md-Emon-Hasan/CIPS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
