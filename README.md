# MADDPG

This is a pytorch implementation of MADDPG on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of MADDPG is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

## Requirements

- python=3.6.5
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0

## Quick Start

```shell
$ python main.py --scenario-name=simple_tag --evaluate-episodes=10
```

Directly run the main.py, then the algrithm will be tested on scenario 'simple_tag' for 10 episodes, using the pretrained model.

## Changes

+ add gumbel_softmax to actor network

```
        ...
        actions = self.action_out(x)
        actions_one_hot = F.gumbel_softmax(actions, tau=1, hard=True)
        ...
```

+ define the actor network for evaluation


```
        class e_Actor(nn.Module):
        ...
```

+ discrete action in multiagent-particle-envs-master/multiagent/environment.py like this:

```
        ...
        action_choice_list = [[0, 0], [0, 0.2], [0, -0.2], [0.2, 0], [0.2, 0.2], [0.2, -0.2], [-0.2, 0],
                                  [-0.2, 0.2], [-0.2, -0.2]]
        which_action_to_choice = 0
        for one_pos in range(9):
            if action[0][one_pos]:
                which_action_to_choice = one_pos
                    break
         agent.action.u[0] += action_choice_list[which_action_to_choice][0]
         agent.action.u[1] += action_choice_list[which_action_to_choice][1]
        ...
```
