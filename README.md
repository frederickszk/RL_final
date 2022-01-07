# RL_final

The final project of reinforcement learning (CS7309)

### TODOs
- [x] Prepare the gym-atari environment
- [x] Implement the DQN algorithms
    - [x] Re-implement the DQN network with PyTorch
    - [x] Implement the DDQN (Double DQN)
- [x] Prepare the MuJoCo environment
- [ ] Implement the A2C algorithm



# Environment Preparation

## For atari environment (ALE)
- Gym with atari
```shell script
pip install gym[atari]
pip install gym[accept-rom-license]
```

- PyTorch
```shell
conda install pytorch torchvision torchaudio cudatoolkit=xxx -c pytorch
```

- OpenCV
```shell
pip install opencv-python
```

- Matplotlib
```shell
conda install matplotlib
```

## For MuJoCo
Refer to the instruction of [mujoco-py](https://github.com/openai/mujoco-py).

My procedures:
- Download the binaries and extract to the assigned directory following the instructions.\
- Prepare the dependencies
  - Run the following command
    ```shell
    sudo apt install libosmesa6-dev libgl1-mesa-glx ligl1-mesa-dev libglfw3 libglew-dev 
    ```
  - Add lines to `~/.bashrc`. Then `source ~/.bashrc`.
    ```shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/(username)/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so
    ```
    Tips: The first line import the mujoco path as the same to the directory in step1. While the third line
    is used to avoid the `ERROR: GLEW initalization error: Missing GL version` when rendering the environment in `gym`.
    The `libGL.so` may be in `usr/lib/nvidia-xxx/` according to other tutorials. 
    However, I could not find that directory. Therefore, I search it in the right place and link to it.
- Run `pip3 install -U 'mujoco-py<2.2,>=2.1'`
- Run `python` and `import mujoco_py`, it starts to automatically build.

# Utils

- Check the action list (may be useful for testing the game)

```python
env.unwrapped.get_action_meanings()
```

- Resolve the package import error in the same folder (Pycharm)

> Right click the folder -> Mark Directory as -> Source Root
> 
> However, relative import would fail if not sys.path.append(folder). Therefore, we use the absolute import. 

- From observations to the network input
Each observation generate a state:`LazyFrames`, including a list of 4 x [1, 84, 84] numpy arrays.
  Use the `np.asarray(state)` can easily convert it to [4, 84, 84] array for further use.
  

# Reference
- [Gsurma's repo](https://github.com/gsurma/atari)
- [Pytorch's RL tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
