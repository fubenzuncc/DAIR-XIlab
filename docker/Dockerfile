# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# version: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2020
# Torch 1.9 Torchvision 0.9 python 3.8.8
FROM nvcr.io/nvidia/pytorch:21.03-py3
LABEL maintainer = "yangdecheng@supremind.com"

RUN export WORKSPACE_ROOT=/workspace && mkdir -p ${WORKSPACE_ROOT}

# Install linux packages
RUN chmod 777 /tmp
RUN apt update && apt install -y zip htop psmisc

RUN pip install seaborn==0.11.1 pytorch_ignite==0.4.4 tensorboardX==2.1 opencv-python==4.5.5.64 Python-Deprecated torchcontrib pandas yacs matplotlib numpy Pillow pretrainedmodels protobuf pycocotools rfconv scikit_learn nuscenes-devkit pyquaternion fvcore sklearn omegaconf -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果要使用 JupyterLab, 在 Dockerfile 中添加
RUN pip --no-cache-dir install --upgrade jupyterlab --index-url http://mirrors.aliyun.com/pypi/simple

# 如需使用 SSH 访问训练容器, 在 Dockerfile 中添加
RUN apt-get update && apt-get install -y libgl1-mesa-glx openssh-server && mkdir -p /var/run/sshd

# git 证书更新
RUN apt-get install --reinstall ca-certificates
RUN mkdir /usr/share/ca-certificates/cacert.org
RUN wget -P /usr/share/ca-certificates/cacert.org http://www.cacert.org/certs/root.crt http://www.cacert.org/certs/class3.crt
RUN sed -i 's:mozilla/DST_Root_CA_X3.crt:!mozilla/DST_Root_CA_X3.crt:' /etc/ca-certificates.conf
RUN echo 'cacert.org/class3.crt' >> /etc/ca-certificates.conf
RUN echo 'cacert.org/root.crt' >> /etc/ca-certificates.conf
RUN update-ca-certificates
RUN git config --global http.sslCAinfo /etc/ssl/certs/ca-certificates.crt 

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# (Optional) 使用 tini 作为 entypoint 防止僵尸进程
ADD https://github.com/krallin/tini/releases/download/v0.18.0/tini /bin/tini
RUN chmod +x /bin/tini
ENTRYPOINT ["/bin/tini", "--"]

ENV LC_ALL=C.UTF-8
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

LABEL com.supremind.os = "ubuntu-20.04"
LABEL com.supremind.type = "Suprevision"

