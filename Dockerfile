# use the official Python image from Docker Hub (slim in local, deepnote/python in deepnote)
#FROM python:3.8-slim
FROM deepnote/python:3.10

# install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    curl \
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

# ---- Patch to remove future_fstrings encoding header ----
RUN python - <<'PY'
import re, importlib.util
from pathlib import Path
pat = re.compile(r'^\s*#\s*-\*-\s*coding:\s*future_fstrings\s*-\*-\s*$', re.I)
def patch_tree(root: Path):
    if not root.exists(): return 0
    n=0
    for p in root.rglob("*.py"):
        try:
            L = p.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        except Exception: continue
        ch=False
        for i in (0,1):
            if i < len(L) and pat.match(L[i]):
                L[i] = "# -*- coding: utf-8 -*-\n"; ch=True
        if ch:
            p.write_text("".join(L), encoding="utf-8"); n+=1
    print("patched", n, "files under", root)
spec = importlib.util.find_spec("acados_template")
if spec and spec.origin:
    patch_tree(Path(spec.origin).parent)
patch_tree(Path("/acados"))
PY
# --------------------------------------------------

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

# entrypoint (bash in local, jupyter in Deepnote)
#CMD ["bash"]
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

