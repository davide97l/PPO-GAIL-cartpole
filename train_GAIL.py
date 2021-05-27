from collections import deque
import torch
from torch import optim
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
from models import GAILDiscriminator
from training_utils import TransitionDataset, adversarial_imitation_update, compute_advantages, \
    indicate_absorbing, ppo_update, flatten_list_dicts


def train(args, agent, env):

    # Load expert trajectories
    with open(args.expert_trajectories_path, 'rb') as handle:
        expert_trajectories = pickle.load(handle)
    expert_trajectories = TransitionDataset(expert_trajectories)

    # Set up actor-critic model optimiser
    agent_optimiser = optim.RMSprop(agent.parameters(), lr=args.learning_rate)

    # Set up discriminator
    discriminator = GAILDiscriminator(env.observation_space.shape[0] + (1 if args.absorbing else 0),
                                      env.action_space.n, args.hidden_size, state_only=args.state_only)
    discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=args.learning_rate)

    # Metrics
    writer = SummaryWriter()

    # Init
    state = env.reset()
    episode_return = 0
    episodes = 0
    trajectories = []
    policy_trajectory_replay_buffer = deque(maxlen=args.imitation_replay_size)

    # Start training
    for step in range(args.steps):
        # Collect set of trajectories by running policy π in the environment
        policy, value = agent(state)
        action = policy.sample()
        log_prob_action, entropy = policy.log_prob(action), policy.entropy()
        next_state, reward, terminal = env.step(action)
        episode_return += reward
        trajectories.append(dict(states=state, actions=action, rewards=torch.tensor([reward], dtype=torch.float32),
                                 terminals=torch.tensor([terminal], dtype=torch.float32),
                                 log_prob_actions=log_prob_action, old_log_prob_actions=log_prob_action.detach(),
                                 values=value, entropies=entropy))
        state = next_state

        # If end episode
        if terminal:
            # Store metrics
            writer.add_scalar("Reward", episode_return, episodes)
            print('episode: {}, total step: {}, last_episode_reward: {}'.format(episodes+1, step+1, episode_return))

            # Reset the environment
            state, episode_return = env.reset(), 0

            if len(trajectories) >= args.batch_size:
                policy_trajectories = flatten_list_dicts(trajectories)
                trajectories = []  # Clear the set of trajectories

                # Use a replay buffer of previous trajectories to prevent overfitting to current policy
                policy_trajectory_replay_buffer.append(policy_trajectories)
                policy_trajectory_replays = flatten_list_dicts(policy_trajectory_replay_buffer)
                # Train discriminator and predict rewards
                for _ in tqdm(range(args.imitation_epochs), leave=False):
                    adversarial_imitation_update(discriminator, expert_trajectories,
                                                 TransitionDataset(policy_trajectory_replays),
                                                 discriminator_optimiser, args.imitation_batch_size,
                                                 args.absorbing, args.r1_reg_coeff)
                states = policy_trajectories['states']
                actions = policy_trajectories['actions']
                next_states = torch.cat([policy_trajectories['states'][1:], next_state])
                terminals = policy_trajectories['terminals']

                if args.absorbing:
                    states, actions, next_states = indicate_absorbing(states, actions,
                                                                      policy_trajectories['terminals'], next_states)
                with torch.no_grad():
                    policy_trajectories['rewards'] = discriminator.predict_reward(states, actions)

                # Compute rewards-to-go R and generalised advantage estimates ψ based on the current value function V
                compute_advantages(policy_trajectories, agent(state)[1], args.discount, args.trace_decay)
                # Normalise advantages
                policy_trajectories['advantages'] = (policy_trajectories['advantages'] - policy_trajectories[
                    'advantages'].mean()) / (policy_trajectories['advantages'].std() + 1e-8)

                # Perform PPO updates using the rewards given by the discriminator
                for epoch in tqdm(range(args.ppo_epochs), leave=False):
                    ppo_update(agent, policy_trajectories, agent_optimiser, args.ppo_clip, epoch, args.value_loss_coeff,
                               args.entropy_loss_coeff)
            episodes += 1

    writer.flush()
    writer.close()

    return agent
