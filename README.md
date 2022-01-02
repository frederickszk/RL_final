# RL_final

The final project of reinforcement learning (CS7309)

### TODOs
- [x] Prepare the gym-atari environment
- [ ] Implement the DQN algorithm
    - [ ] Re-implement the network with PyTorch
- [ ] Implement the DDQN (Double DQN)


# Environment Preparation

- Gym with atari
```shell script
pip install gym[atari]
pip install gym[accept-rom-license]
```

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
