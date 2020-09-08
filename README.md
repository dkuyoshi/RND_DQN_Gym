# RND + DQN

## CartPoleなどのGym環境用です
[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)

## Requirement

chainerrl==0.8.0

chainer==7.4.0

[chainerのgithub](https://github.com/chainer)



## Usage
`python train_dqn_easy.py`

- DQNでのTraining

- --gpu : デフォルトは-1(gpu使いたかったら0にする)

`python train_easy_rnd.py`

- DQN + RND でのTraining

- --gpu : デフォルトは-1(gpu使いたかったら0にする)



