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

Check the action list (may be useful for testing the game)

```python
env.unwrapped.get_action_meanings()
```


# Reference
- [Gsurma's repo](https://github.com/gsurma/atari)
- [Pytorch's RL tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
