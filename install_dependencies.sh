pip install --no-cache ninja cython 
pip install --no-cache torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


# RUN git clone https://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@github.com/haosulab/pyrl.git
# RUN cat setup.py

pip install --no-cache wandb
pip install --no-cache yapf sorcery tensorboardX pynvml lmdb
pip install https://storage1.ucsd.edu/wheels/sapien-dev/sapien-2.0.0.dev20230310-cp39-cp39-manylinux2014_x86_64.whl


# RUN mkdir -p /root/.mujoco \
#     && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#     && tar -xf mujoco.tar.gz -C /root/.mujoco \
#     && rm mujoco.tar.gz

# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
pip install mujoco_py
pip install tensorboardX
pip install opencv-python
pip install tqdm
pip install gym==0.23.1
pip install yacs==0.1.8
pip install matplotlib
pip install transforms3d
pip install h5py
pip install moviepy


pip install pytorch3d --upgrade --no-cache -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html



# RUN pip install mujoco

curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
# RUN python3 -c "import gym; gym.make('HalfCheetah-v3')"

#WORKDIR /root
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# RUN chmod +x  "cuda_11.8.0_520.61.05_linux.run"-
# RUN ./cuda_11.8.0_520.61.05_linux.run -s


pip install trimesh

# nvcc -O3 -Xptxas -O3,-v -ccbin=g++ --compiler-options -fPIC -shared /root/TaskAnnotator/mpm/csrc/integrator.cu -o /root/TaskAnnotator/mpm/libmaniskill_mpm.so

pip install omegaconf
pip install -e .