# use the official Python image from Docker Hub
FROM python:3.10-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    make \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libcdd-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# clone acados
RUN git clone https://github.com/acados/acados.git /acados && \
    cd /acados && \
    git submodule update --recursive --init

# build acados
RUN cd /acados && \
    mkdir build && \
    cd build && \
    cmake .. -DACADOS_WITH_C_INTERFACE=ON -DACADOS_INSTALL_PYTHON=ON && \
    make install

# double check acados_template
RUN cd /acados/interfaces/acados_template && \
    pip install .

# environment variables
ENV ACADOS_SOURCE_DIR=/acados
ENV ACADOS_INSTALL_DIR=/acados
ENV LD_LIBRARY_PATH=/acados/lib
ENV PYTHONPATH=/acados/interfaces/acados_template/python

# set the working directory
WORKDIR /app

# copy the current directory contents into the container at /app
COPY . /app

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-csv jupyterlab cvxopt scipy pycddlib==2.1.0 pytope matplotlib seaborn torch

# expose the port for jupyterlab
EXPOSE 8888

# entrypoint
CMD ["bash"]
