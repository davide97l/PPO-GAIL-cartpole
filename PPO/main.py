from agent import Agent
import argparse
from collect import collect


def main(args):
    agent = Agent()
    if args.load:
        agent.load_weights()
    if args.train:
        agent.train()
    if args.save:
        agent.save_weights()
    if args.eval:
        agent.eval(10)
    if args.collect:
        collect(args)


# python PPO/main.py --train --save
# python PPO/main.py --load --eval
# python PPO/main.py --collect
# tensorboard --logdir=runs
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--collect', action='store_true', default=False)

    parser.add_argument('--min_reward', type=int, default=195,
                        help='Collect a trajectory only if its cumulative reward is >= this threshold')
    parser.add_argument('--n_traj', type=int, default=100,
                        help='Number of trajectories to store')
    parser.add_argument('--traj_path', type=str, default='PPO/trajectories/cart_pole.pickle',
                        help='Path where to store collected trajectories')
    args = parser.parse_args()
    main(args)
