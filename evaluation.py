import torch


# Evaluate agent with deterministic policy π
def evaluate_agent(agent, env, episodes, return_trajectories=False, seed=1):
    env.seed(seed)
    returns, trajectories = [], []
    steps = 0

    for _ in range(episodes):
        states, actions, rewards = [], [], []
        state, terminal = env.reset(), False
        while not terminal:
            with torch.no_grad():
                policy, _ = agent(state)
                action = policy.logits.argmax(dim=-1)  # Pick action greedily
                state, reward, terminal = env.step(action)
                steps += 1

                if return_trajectories:
                    states.append(state)
                    actions.append(action)
                rewards.append(reward)
        returns.append(sum(rewards))
        if return_trajectories:
            # Collect trajectory data (including terminal signal, which may be needed for offline learning)
            terminals = torch.cat([torch.zeros(len(rewards) - 1), torch.ones(1)])
            trajectories.append(dict(states=torch.cat(states), actions=torch.cat(actions),
                                     rewards=torch.tensor(rewards, dtype=torch.float32), terminals=terminals))

    print("avg rew: {}, avg step: {}, max rew: {}, min reward: {}".format(sum(returns)/episodes, steps/episodes,
                                                                          max(returns), min(returns)))
    return (returns, trajectories) if return_trajectories else returns
