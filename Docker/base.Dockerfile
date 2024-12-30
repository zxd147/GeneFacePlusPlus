# 使用现有的基础镜像
FROM base/python:ubuntu22.04-cuda11.8-python3.9

# 设置代理 (确保网络环境)
ENV all_proxy=http://192.168.0.64:7890
# 设置 GPU 使用
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
# 设置访问密钥
ENV VITS2-SECRET-KEY=sk-gebeface

# 设置工作目录
WORKDIR /opt/install/

# 复制脚本文件
COPY ./requirements.txt ./requirements.txt
COPY ./install_ext.sh ./install_ext.sh
# 复制本地文件
COPY ./cu118-py39-2.0.1/ ./cu118-py39-2.0.1/
COPY ./dlib /usr/local/lib/python3.10/site-packages/dlib
COPY ./dlib-19.24.6.dist-info /usr/local/lib/python3.10/site-packages/dlib-19.24.6.dist-info
COPY ./freqencoder ./freqencoder
COPY ./gridencoder ./gridencoder
COPY ./raymarching ./raymarching
COPY ./shencoder ./shencoder

# 更新并安装必要的依赖
ARG APTPKGS="zsh wget tmux tldr nvtop vim neovim curl rsync net-tools less iputils-ping 7zip zip unzip"
RUN apt-get update \
    # 在这里安装你需要的依赖，比如 git、python 等
    && apt-get install -y --no-install-recommends git git-lfs ffmpeg openssl ca-certificates openssh-server openssh-client \
    && apt-get install -y --no-install-recommends libasound2-dev portaudio19-dev libgnutls30 libgl1 libssl-dev \
    && apt-get install -y --no-install-recommends $APTPKGS \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global credential.helper store && git lfs install \
    && pip install ./cu118-py39-2.0.1/* \
    && pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    && pip install chardet cython openmim==0.3.9 \
    && mim install mmcv==2.1.0 \
    && pip install -r ./requirements.txt \
    && bash ./install_ext.sh  \
    && pip install ./cu118-py39-2.0.1/* \
    && rm -r ./cu118-py39-2.0.1/ \
    && pip cache purge

# 映射端口
EXPOSE 8041

# 容器启动时默认执行的命令
CMD ["/opt/nvidia/nvidia_entrypoint.sh"]






