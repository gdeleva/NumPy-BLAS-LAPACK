FROM ubuntu:20.10

# We install numpy supported by the Netlib implementation of blas and lapack libraries 

## Essentials
RUN : \
    && apt-get update \
    && apt-get install -y sudo \
    && apt-get install -y git \
    && sudo apt-get install -y libc6 gcc \
    && sudo apt-get install -y gfortran \
    && sudo apt-get install -y python-is-python3 python3-dev python3-pip cython3 \
    && :

# Standard Blas and Lapack libraries 
RUN sudo apt-get install -y libblas-dev
RUN sudo apt-get install -y liblapack-dev


# Download NumPy library
WORKDIR /app/downloads
RUN git clone https://github.com/numpy/numpy

WORKDIR /app/downloads/numpy
RUN NPY_BLAS_ORDER=BLAS NPY_LAPACK_ORDER=LAPACK python setup.py build
RUN python setup.py install

# Install scikit-learn
RUN pip install scikit-learn

WORKDIR /app/scripts
