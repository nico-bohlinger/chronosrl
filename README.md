# ChronoSRL: Temporal Geometry for Self-Supervised Reinforcement Learning

│ [Website](https://nico-bohlinger.github.io/chronosrl_website) │ [Paper]() │

Code for ChronoSRL and the baselines CRL, AC-CRL and SRL on the seven JaxGCRL locomotion and navigation tasks and the three Unitree Go2 tasks of the paper.


## Installation
```
git clone git@github.com:nico-bohlinger/chronosrl.git
cd chronosrl
```
The JaxGCRL tasks and the Go2 tasks use the JAX, Brax and MuJoCo versions of the paper runs, so they need separate environments.

JaxGCRL tasks:
```
conda create -n chronosrl python=3.10
conda activate chronosrl
pip install -e ".[benchmark]"
pip install "jax[cuda12_pip]==0.4.23" "nvidia-cudnn-cu12==8.9.7.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Go2 tasks (MuJoCo Warp, needs an NVIDIA GPU):
```
conda create -n chronosrl_go2 python=3.11
conda activate chronosrl_go2
pip install -e ".[go2]"
pip install "jax[cuda12]==0.7.2"
```


## Experiments
```
cd experiments
python experiment.py --algorithm.name=chronosrl --environment.name=ant_u4_maze --algorithm.depth=8 --environment.seed=1000
XLA_FLAGS=--xla_gpu_enable_command_buffer= python experiment.py --algorithm.name=chronosrl --environment.name=go2_velocity --algorithm.depth=8 --environment.seed=1000
```
- `--algorithm.name`: `chronosrl`, `crl`, `accrl` or `srl`
- `--environment.name`:
    - JaxGCRL: `ant_u4_maze`, `ant_big_maze`, `ant_hardest_maze`, `ant_u5_maze`, `humanoid`, `humanoid_u_maze` or `humanoid_big_maze`
    - Go2: `go2_velocity`, `go2_position` or `go2_box`
- `--algorithm.depth`: network depth, 1 to 64 in the paper
- `--environment.seed`: 1000 to 1003 in the paper

The Go2 runs disable XLA's command buffers as in the paper, which avoids rare crashes of MuJoCo Warp's CUDA graph capture.

The defaults are the settings of the paper for every method and task, including the goal sequences and larger batches of ChronoSRL and SRL on the humanoid tasks and the training regime of the Go2 tasks.
All settings can be changed in the same way, e.g. `--algorithm.total_timesteps=50000000`.

The evaluation metrics are printed before every epoch and after the last one.
`eval/time_at_goal` is the Time at Goal of the paper, velocity tracking adds `eval/forward_time_at_goal` and `eval/forward_tracking_error` for the forward command, and box climbing logs the box height of the training curriculum as `train/box_height` after every epoch.
`--runner.track_wandb=True --runner.wandb_entity=<entity>` logs to Weights & Biases and `--runner.save_model=True` saves the networks to `runs/`.
`slurm_experiment.sh` runs the four seeds of one setting as a Slurm array.


## Acknowledgements
The JaxGCRL environments are adapted from [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) in the version of [scaling-crl](https://github.com/wang-kevin3290/scaling-crl) (Apache-2.0).
The survival critic and the relabeling follow [SRL](https://github.com/Simple-Robotics/survival-reinforcement-learning) and the action chunks follow [AC-CRL](https://github.com/M-Korniak/action-chunked-contrastive-rl).
The Go2 environment is the MuJoCo Warp locomotion environment of [RL-X](https://github.com/nico-bohlinger/RL-X) with the Go2 model of [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) (BSD-3-Clause).
