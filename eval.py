import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import QRDQNAgent
from fqf_iqn_qrdqn.agent import DQNAgent
from fqf_iqn_qrdqn.agent import FQFAgent
from fqf_iqn_qrdqn.agent import IQNAgent

def run(args):
    dirs = os.listdir(os.path.join('logs', args.env_id))
    cnt = 0
    model_list = []
    for d in dirs:
        cnt += 1
        print(str(cnt) + ': logs/' + args.env_id + '/' + d + '/model/best')
        model_list.append('logs/' + args.env_id + '/' + d + '/model/best')
        cnt += 1
        print(str(cnt) + ': logs/' + args.env_id + '/' + d + '/model/final')
        model_list.append('logs/' + args.env_id + '/' + d + '/model/final')

    model_dir = int(input('Select model directory path: '))

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join('logs', 'dummy')

    # Create the agent and run.
    if args.agent == 'qrdqn':
        agent = QRDQNAgent(
            env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
            cuda=args.cuda, **config)
        agent.load_models(model_list[model_dir-1])
        agent.eval_and_render()

    elif args.agent == 'dqn':
        agent = DQNAgent(
            env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
            cuda=args.cuda, **config)
        agent.load_models(model_list[model_dir-1])
        agent.eval_and_render()

    elif args.agent == 'iqn':
        agent = IQNAgent(
            env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
            cuda=args.cuda, **config)
        agent.load_models(model_list[model_dir-1])
        agent.eval_and_render()

    elif args.agent == 'fqf':
        agent = FQFAgent(
            env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
            cuda=args.cuda, **config)
        agent.load_models(model_list[model_dir-1])
        agent.eval_and_render()

    else:
        print('Agent does not exist.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='qrdqn')
    parser.add_argument('--config', type=str, default=os.path.join('config', 'qrdqn.yaml'))
    #parser.add_argument('--model_dir', type=str)
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
