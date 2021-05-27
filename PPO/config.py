class AgentConfig:
    # Learning
    gamma = 0.99
    plot_every = 10
    update_freq = 1
    k_epoch = 3
    learning_rate = 0.02
    lmbda = 0.95
    eps_clip = 0.2
    v_coef = 1
    entropy_coef = 0.01
    max_episodes = 1000
    weights_path = "PPO/models/cart_pole.pth"
    env_name = 'CartPole-v0'

    # Memory
    memory_size = 400
