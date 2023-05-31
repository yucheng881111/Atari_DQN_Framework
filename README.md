# Atari DQN Framework

Implementation of DQN, QRDQN, IQN, FQF. Codes modified from https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch.

## Setup

```
(git clone https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch)

conda create -n fqf python=3.8 -y
conda activate fqf
(install torch cuda version https://pytorch.org/)
pip install -r requirements.txt
conda install -c conda-forge atari_py
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar .
python -m atari_py.import_roms ROMS
conda install -c anaconda protobuf
```

## Run Train

```
(env CUDA_VISIBLE_DEVICES=0) python train_qrdqn.py --cuda --env_id BreakoutNoFrameskip-v4 --config config/qrdqn.yaml
```
## Run Evaluate and Render
Install this for render:
```
conda install -c conda-forge libstdcxx-ng
```
run:
```
python eval.py --cuda --env_id BreakoutNoFrameskip-v4 --config config/qrdqn_dueling.yaml --agent qrdqn
```
then input the number of model path.

Note that your agent, config, and model must match.

