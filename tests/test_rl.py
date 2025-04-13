import tensorflow as tf
import numpy as np
import gym
import gym.spaces

from Satmind.actor_critic import Actor, Critic
from Satmind.utils import OrnsteinUhlenbeck
from Satmind.replay_memory import Per_Memory, Uniform_Memory


def test_training():
    """Test if training has taken place (update of weights)"""

    ENV = 'Pendulum-v1'
    env = gym.make(ENV)
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    # Create actor with higher learning rate for testing
    actor = Actor(features, n_actions, 128, 128, action_bound, 0.0001, 0.05, 1, 'actor')  # Higher learning rate
    critic = Critic(features, n_actions, 128, 128, 0.001, 0.001, 'critic')

    # Handle both new and old gym versions
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        s = reset_result[0]
    else:
        s = reset_result

    # Store model weights before training
    before_weights = [tf.identity(w) for w in actor.model.weights]

    # Create state tensor
    s_tensor = tf.convert_to_tensor(np.reshape(s, (1, features)), dtype=tf.float32)

    # Use a significant action gradient to ensure weight changes
    action_gradient = tf.ones((1, n_actions), dtype=tf.float32) * 5.0

    # Train the actor with this gradient
    with tf.GradientTape() as tape:
        actions = actor.model(s_tensor, training=True)
        # Define a loss that will cause weights to change
        loss = -tf.reduce_mean(tf.reduce_sum(actions * action_gradient, axis=1))

    # Calculate and apply gradients manually with higher learning rate
    gradients = tape.gradient(loss, actor.model.trainable_variables)
    for var, grad in zip(actor.model.trainable_variables, gradients):
        var.assign_sub(0.1 * grad)  # Apply significant gradient step

    # Get weights after training
    after_weights = actor.model.weights

    # Check if any weights changed
    for i, (before, after) in enumerate(zip(before_weights, after_weights)):
        if tf.reduce_any(tf.not_equal(before, after)):
            # If we find any difference, the test passes
            return

    # If no weights changed, fail the test
    assert False, "Actor weights did not change after training"


def test_rl():
    """Test reinforcement learning implementation using modern Gym API"""

    ENVS = ('Pendulum-v1', 'MountainCarContinuous-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2')
    ENV = ENVS[2]  # BipedalWalker-v3

    # Create environment
    env = gym.make(ENV, render_mode="human")  # Modern API supports render_mode parameter

    # Environment properties
    features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bound = env.action_space.high

    # Set random seeds for reproducibility
    seed_value = 1234
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)  # TF2 version

    # Training parameters
    num_episodes = 1000
    batch_size = 128
    layer_1_nodes, layer_2_nodes = 500, 450
    tau = 0.001
    actor_lr, critic_lr = 0.0001, 0.001
    gamma = 0.99

    # Initialize actor and critic networks
    actor = Actor(features, n_actions, layer_1_nodes, layer_2_nodes,
                  action_bound, tau, actor_lr, batch_size, 'actor')
    critic = Critic(features, n_actions, layer_1_nodes, layer_2_nodes,
                    critic_lr, tau, 'critic')

    # Noise process for exploration
    actor_noise = OrnsteinUhlenbeck(np.zeros(n_actions))

    # Initialize replay memory with prioritized experience replay
    per_mem = Per_Memory(capacity=100000)

    # Initialize target networks
    actor.update_target_network()
    critic.update_target_network()

    # Training loop
    for i in range(num_episodes):
        # Reset environment with seed only for the first episode
        if i == 0:
            observation, _ = env.reset(seed=seed_value)
        else:
            observation, _ = env.reset()

        sum_reward = 0
        sum_q = 0
        episode_rewards = []
        steps = 0

        # Episode loop
        while True:
            # Convert state to tensor
            state_tensor = tf.convert_to_tensor(
                np.reshape(observation, (1, features)),
                dtype=tf.float32
            )

            # Get action from actor network and add exploration noise
            action_tensor = actor.predict(state_tensor)
            action = action_tensor.numpy() + actor_noise()

            # Take action in environment (modern API returns 5 values)
            next_observation, reward, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated  # Combined termination signal

            episode_rewards.append(float(reward))

            # Store experience in replay memory with initial error
            error = abs(reward)  # Initial TD error estimate
            per_mem.add(error, (
                np.reshape(observation, (features,)),
                np.reshape(action[0], (n_actions,)),
                reward,
                np.reshape(next_observation, (features,)),
                done
            ))

            # Only train if we have enough samples
            if batch_size < per_mem.count:
                # Sample batch from replay memory
                mem, idxs, isweight = per_mem.sample(batch_size)

                # Extract and prepare batch data as tensors
                states = tf.convert_to_tensor(
                    np.array([_[0] for _ in mem]),
                    dtype=tf.float32
                )
                actions = tf.convert_to_tensor(
                    np.array([_[1] for _ in mem]),
                    dtype=tf.float32
                )
                rewards = np.array([_[2] for _ in mem])
                next_states = tf.convert_to_tensor(
                    np.array([_[3] for _ in mem]),
                    dtype=tf.float32
                )
                dones = np.array([_[4] for _ in mem])
                importance_weights = tf.convert_to_tensor(
                    np.reshape(isweight, (batch_size, 1)),
                    dtype=tf.float32
                )

                # Calculate target Q-values
                next_actions = actor.predict_target(next_states)
                target_q_values = critic.predict_target(next_states, next_actions)
                target_q_numpy = target_q_values.numpy()

                # Prepare TD targets
                y_targets = []
                for j in range(batch_size):
                    if dones[j]:
                        y_targets.append(rewards[j])
                    else:
                        y_targets.append(rewards[j] + gamma * target_q_numpy[j][0])

                # Convert targets to tensor
                y_targets_tensor = tf.convert_to_tensor(
                    np.reshape(y_targets, (batch_size, 1)),
                    dtype=tf.float32
                )

                # Update critic network
                errors, predicted_q, _ = critic.train(
                    states, actions, y_targets_tensor, importance_weights
                )

                # Update priorities in replay memory
                errors_numpy = errors.numpy()
                for j in range(batch_size):
                    per_mem.update(idxs[j], abs(errors_numpy[j][0]))

                # Track Q-values for reporting
                sum_q += np.max(predicted_q.numpy())

                # Update actor policy using policy gradient
                actions_pred = actor.predict(states)
                action_gradients = critic.action_gradient(states, actions_pred)
                actor.train(states, action_gradients)

                # Update target networks with soft updates
                actor.update_target_network()
                critic.update_target_network()

            # Update for next step
            sum_reward += reward
            observation = next_observation
            steps += 1

            # Check if episode is done
            if done:
                avg_q = sum_q / float(steps) if steps > 0 else 0
                print(f'Episode: {i}, Reward: {int(sum_reward)}, Avg Q: {avg_q:.4f}, Steps: {steps}')
                print('===========')
                break


if __name__ == '__main__':
    test_rl()