FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04


SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y git wget ssh
RUN apt-get install -y libegl1 libxext6
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y x11vnc fluxbox xvfb
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute,display




RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vulkan-tools \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    xserver-xorg-video-dummy \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# RUN conda install ffmpeg -y && conda clean -ya

ARG SSH_PRIVATE_KEY
RUN mkdir ~/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > ~/.ssh/id_rsa_git
RUN chmod 600 ~/.ssh/id_rsa_git
RUN ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
# RUN echo "Host github.com\n\tStrictHostKeyChecking no\n\tHostName github.com\n\tIdentityFile ~/.ssh/id_rsa_git" >> ~/.ssh/config
# RUN ssh-keyscan /github.com/ >> ~/.ssh/known_hosts

# RUN cat ~/.ssh/config

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh
RUN chmod +x ~/miniconda.sh
RUN ~/miniconda.sh -b -u -p /opt/conda
RUN rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python=3.10 && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN pip install --no-cache ninja cython 
RUN pip install --no-cache torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


# RUN git clone https://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@github.com/haosulab/pyrl.git
# RUN cat setup.py

RUN pip install --no-cache wandb
RUN pip install --no-cache yapf sorcery tensorboardX pynvml lmdb
RUN pip install https://storage1.ucsd.edu/wheels/sapien-dev/sapien-2.0.0.dev20230405-cp310-cp310-manylinux2014_x86_64.whl


# RUN mkdir -p /root/.mujoco \
#     && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#     && tar -xf mujoco.tar.gz -C /root/.mujoco \
#     && rm mujoco.tar.gz

# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN pip install mujoco_py
RUN pip install tensorboardX
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install gym==0.23.1
RUN pip install yacs==0.1.8
RUN pip install matplotlib
RUN pip install transforms3d
RUN pip install h5py
RUN pip install moviepy

RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"



# RUN pip install mujoco

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
# RUN python3 -c "import gym; gym.make('HalfCheetah-v3')"

#WORKDIR /root
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# RUN chmod +x  "cuda_11.8.0_520.61.05_linux.run"-
# RUN ./cuda_11.8.0_520.61.05_linux.run -s


WORKDIR /root
RUN git clone https://github.com/novnc/noVNC.git
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y net-tools

RUN pip install trimesh
COPY ./xorg.conf xorg.conf
#COPY nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
#COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json


# Install cmake version 3.15
RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2-Linux-x86_64.tar.gz && \
    tar -xvf cmake-3.15.2-Linux-x86_64.tar.gz && cp cmake-3.15.2-Linux-x86_64/bin/cmake /usr/local/bin && \
    mkdir -p /usr/local/share/cmake-3.15 && \
    cp -r cmake-3.15.2-Linux-x86_64/share/cmake-3.15/* /usr/local/share/cmake-3.15 && \
    rm -rf cmake-3.15.2-Linux-x86_64*

# # Download and compile vulkan components
# RUN ln -s /usr/bin/python3 /usr/bin/python && \
#     git clone https://github.com/KhronosGroup/Vulkan-ValidationLayers.git /opt/vulkan && \
#     cd /opt/vulkan && git checkout $(git describe --tags `git rev-list --tags --max-count=1`) && \
#     mkdir build && cd build && ../scripts/update_deps.py && \
#     cmake -C helper.cmake -DCMAKE_BUILD_TYPE=Release .. && \
#     cmake --build . && make install && ldconfig && \
#     mkdir -p /usr/local/lib && cp -a Vulkan-Loader/build/install/lib/* /usr/local/lib && \
#     mkdir -p /usr/local/include/vulkan && cp -r Vulkan-Headers/build/install/include/vulkan/* /usr/local/include/vulkan && \
#     mkdir -p /usr/local/share/vulkan/registry && \
#     cp -r Vulkan-Headers/build/install/share/vulkan/registry/* /usr/local/share/vulkan/registry && \
#     rm -rf /opt/vulkan

RUN GIT_SSH_COMMAND="ssh -i /root/.ssh/id_rsa_git -o IdentitiesOnly=yes" git clone -b hza https://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@github.com/hzaskywalker/TaskAnnotator.git ~/TaskAnnotator
WORKDIR /root/TaskAnnotator/
RUN python setup.py develop

RUN nvcc -O3 -Xptxas -O3,-v -ccbin=g++ --compiler-options -fPIC -shared /root/TaskAnnotator/mpm/csrc/integrator.cu -o /root/TaskAnnotator/mpm/libmaniskill_mpm.so

RUN cp docker/start.sh start.sh
RUN pip install omegaconf
RUN pip install stable_baselines3[extra]
RUN pip install geomloss
RUN pip install pykeops