FROM ubuntu:20.10

# We install numpy supported by OpenBLAS library 

## Essentials
RUN : \
    && apt-get update \
    && apt-get install -y sudo \
    && apt-get install -y git \
    && sudo apt-get install -y libc6 gcc \
    && sudo apt-get install -y gfortran \
    && sudo apt-get install -y python-is-python3 python3-dev python3-pip cython3 \
    && :

# OpenBLAS library 
RUN sudo apt-get install -y libopenblas-dev
ENV OPENBLAS_NUM_THREADS=4

# Download NumPy library
WORKDIR /app/downloads
RUN git clone https://github.com/numpy/numpy

# Install NumPy
WORKDIR /app/downloads/numpy
RUN NPY_BLAS_ORDER=openblas NPY_LAPACK_ORDER=openblas python setup.py build
RUN python setup.py install

# Install scikit-learn
RUN pip install scikit-learn

WORKDIR /app/scripts
