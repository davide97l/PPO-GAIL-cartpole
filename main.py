import argparse
import torch
from environments import CartPoleEnv
from evaluation import evaluate_agent
from models import ActorCritic
from train_GAIL import train


def get_args():
    parser = argparse.ArgumentParser(description='IL')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--steps', type=int, default=200000, metavar='T', help='Number of environment steps')
    parser.add_argument('--hidden-size', type=int, default=32, metavar='H', help='Hidden size')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount')
    parser.add_argument('--trace-decay', type=float, default=0.95, metavar='λ', help='GAE trace decay')
    parser.add_argument('--ppo-clip', type=float, default=0.2, metavar='ε', help='PPO clip ratio')
    parser.add_argument('--ppo-epochs', type=int, default=4, metavar='K', help='PPO epochs')
    parser.add_argument('--value-loss-coeff', type=float, default=0.5, metavar='c1', help='Value loss coefficient')
    parser.add_argument('--entropy-loss-coeff', type=float, default=0, metavar='c2',
                        help='Entropy regularisation coefficient')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='η', help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='B', help='Minibatch size')
    parser.add_argument('--max-grad-norm', type=float, default=1, metavar='N', help='Maximum gradient L2 norm')
    parser.add_argument('--state-only', action='store_true', default=False, help='State-only imitation learning')
    parser.add_argument('--absorbing', action='store_true', default=False, help='Indicate absorbing states')
    parser.add_argument('--imitation-epochs', type=int, default=5, metavar='IE', help='Imitation learning epochs')
    parser.add_argument('--imitation-batch-size', type=int, default=128, metavar='IB',
                        help='Imitation learning minibatch size')
    parser.add_argument('--imitation-replay-size', type=int, default=4, metavar='IRS',
                        help='Imitation learning trajectory replay size')
    parser.add_argument('--r1-reg-coeff', type=float, default=1, metavar='γ', help='R1 gradient regularisation coefficient')
    parser.add_argument('--weights_path', type=str, default='models/GAIL/cart_pole.pth')
    parser.add_argument('--evaluation_episodes', type=int, default=100)

    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    return parser.parse_args()


# python main.py --train --save
# python main.py --load --eval
if __name__ == "__main__":
    args = get_args()
    # Set up environment
    env = CartPoleEnv()
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    # Set up actor-critic model
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.hidden_size)

    if args.load:
        agent = torch.load(args.weights_path)
        agent.eval()
    if args.train:
        agent = train(args, agent, env)
    if args.eval:
        evaluate_agent(agent, env, args.evaluation_episodes)
    if args.save:
        assert args.weights_path is not None
        torch.save(agent, args.weights_path)
        print("model saved to", args.weights_path)
    env.close()


