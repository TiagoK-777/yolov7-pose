# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && \
    apt-get install -y libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /demo

COPY requirements.txt .

RUN python -m venv /opt/yolov7-env && \
    . /opt/yolov7-env/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair && \
    echo 'source /opt/yolov7-env/bin/activate' >> /root/.bashrc

# RUN pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair

COPY . .

WORKDIR /demo/src/
