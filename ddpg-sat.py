import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
import pickle
import argparse
import datetime
import json
from math import degrees
import Satmind.actor_critic as models
from Satmind.env_orekit import OrekitEnv
import Satmind.utils
from Satmind.replay_memory import Uniform_Memory, Per_Memory

stepT = 500.0


def orekit_setup():
    """Set up the Orekit environment with parameters from input file"""
    mission_type = ['inclination_change', 'Orbit_Raising', 'sma_change', 'meo_geo']

    input_file = 'input.json'
    with open(input_file) as input:
        data = json.load(input)
        mission = data[mission_type[1]]
        state = list(mission['initial_orbit'].values())
        state_targ = list(mission['target_orbit'].values())
        date = list(mission['initial_date'].values())
        dry_mass = mission['spacecraft_parameters']['dry_mass']
        fuel_mass = mission['spacecraft_parameters']['fuel_mass']
        duration = mission['duration']
    mass = [dry_mass, fuel_mass]
    duration = 24.0 * 60.0 ** 2 * duration

    env = OrekitEnv(state, state_targ, date, duration, mass, stepT)
    return env, duration, mission_type[1]


def main(args):
    ENVS = ('OrekitEnv-orbit-raising', 'OrekitEnv-incl', 'OrekitEnv-sma', 'meo_geo')
    ENV = ENVS[2]

    env, duration, mission = orekit_setup()
    iter_per_episode = int(duration / stepT)
    ENV = mission

    # Network inputs and outputs
    features = env.observation_space
    n_actions = env.action_space
    action_bound = env.action_bound

    # Set random seeds
    seed_value = 1234
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)  # TF2 version

    # Training parameters
    num_episodes = 2000
    batch_size = 128
    layer_1_nodes, layer_2_nodes = 512, 450
    tau = 0.01
    actor_lr, critic_lr = 0.001, 0.0001
    GAMMA = 0.99

    # Initialize actor and critic networks (TF2 version)
    actor = models.Actor(features, n_actions, layer_1_nodes, layer_2_nodes,
                         action_bound, tau, actor_lr, batch_size, 'actor')
    actor_noise = Satmind.utils.OrnsteinUhlenbeck(np.zeros(n_actions))
    critic = models.Critic(features, n_actions, layer_1_nodes, layer_2_nodes,
                           critic_lr, tau, 'critic')

    # Replay memory buffer
    per_mem = Per_Memory(capacity=10000000)

    # Save model directory and parameters
    LOAD, TRAIN, checkpoint_path, rewards, save_fig, show = model_saving(ENV, args)

    # Save the model parameters (for reproducibility)
    params = checkpoint_path + '/model_params.txt'
    with open(params, 'w+') as text_file:
        text_file.write("environment params:\n")
        text_file.write("environment: " + ENV + "\n")
        text_file.write("episodes: {}, iterations per episode {}\n".format(num_episodes, iter_per_episode))
        text_file.write("model parameters:\n")
        text_file.write(actor.__str__())
        text_file.write(critic.__str__() + "\n")

    # Create TF2 checkpoint manager for saving/loading models
    checkpoint = tf.train.Checkpoint(actor_model=actor.model,
                                     actor_target=actor.target_model,
                                     critic_model=critic.model,
                                     critic_target=critic.target_model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

    # Render target trajectory
    env.render_target()
    env.randomize = True

    if TRAIN:
        # Initialize target networks
        actor.update_target_network()
        critic.update_target_network()

        if LOAD and manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print(f"Restored from {manager.latest_checkpoint}")

        noise_decay = 1

        for i in range(1, num_episodes):
            s = env.reset()
            sum_reward = 0
            sum_q = 0
            actions = []
            env.target_hit = False
            noise_decay = np.clip(noise_decay - 0.0001, 0.01, 1)

            for j in range(iter_per_episode):
                # Convert state to tensor
                s_tensor = tf.convert_to_tensor(np.reshape(s, (1, features)), dtype=tf.float32)

                # Select action and add noise
                a = actor.predict(s_tensor).numpy()
                a = np.clip(a + actor_noise(), -action_bound, action_bound)

                # Observe state and reward
                s1, r, done = env.step(a[0])

                actions.append(a[0])

                # Store in replay memory with priority based on reward
                error = abs(r)
                per_mem.add(error, (np.reshape(s, (features,)),
                                    np.reshape(a[0], (n_actions,)),
                                    r,
                                    np.reshape(s1, (features,)),
                                    done))

                # Sample from replay memory if we have enough samples
                if batch_size < per_mem.count:
                    mem, idxs, isweight = per_mem.sample(batch_size)

                    # Prepare batch data as tensors
                    s_rep = tf.convert_to_tensor(np.array([_[0] for _ in mem]), dtype=tf.float32)
                    a_rep = tf.convert_to_tensor(np.array([_[1] for _ in mem]), dtype=tf.float32)
                    r_rep = np.array([_[2] for _ in mem])
                    s1_rep = tf.convert_to_tensor(np.array([_[3] for _ in mem]), dtype=tf.float32)
                    d_rep = np.array([_[4] for _ in mem])
                    isweight_tensor = tf.convert_to_tensor(np.reshape(isweight, (batch_size, 1)), dtype=tf.float32)

                    # Get target actions and Q-values using target networks
                    target_actions = actor.predict_target(s1_rep)
                    target_q = critic.predict_target(s1_rep, target_actions).numpy()

                    # Compute TD targets
                    y_i = []
                    for x in range(batch_size):
                        if d_rep[x]:
                            y_i.append(r_rep[x])
                        else:
                            y_i.append(r_rep[x] + GAMMA * target_q[x][0])

                    y_i_tensor = tf.convert_to_tensor(np.reshape(y_i, (batch_size, 1)), dtype=tf.float32)

                    # Update critic network
                    with tf.GradientTape() as tape:
                        q_predicted = critic.model([s_rep, a_rep], training=True)
                        errors = q_predicted - y_i_tensor
                        critic_loss = tf.reduce_mean(tf.multiply(tf.square(errors), isweight_tensor))

                    critic_gradients = tape.gradient(critic_loss, critic.model.trainable_variables)
                    critic.optimizer.apply_gradients(zip(critic_gradients, critic.model.trainable_variables))

                    # Update priorities in replay memory
                    errors_np = errors.numpy()
                    for n in range(batch_size):
                        idx = idxs[n]
                        per_mem.update(idx, abs(errors_np[n][0]))

                    sum_q += np.amax(q_predicted.numpy())

                    # Update actor policy using policy gradient
                    with tf.GradientTape() as tape:
                        actions_pred = actor.model(s_rep, training=True)
                        actor_loss = -tf.reduce_mean(critic.model([s_rep, actions_pred]))

                    actor_gradients = tape.gradient(actor_loss, actor.model.trainable_variables)
                    actor.optimizer.apply_gradients(zip(actor_gradients, actor.model.trainable_variables))

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()

                sum_reward += r
                s = s1

                if done or j >= iter_per_episode - 1:
                    rewards.append(sum_reward)
                    print(f'I: {degrees(env._currentOrbit.getI())}')
                    print('Episode: {}, reward: {}, Q_max: {}'.format(i, int(sum_reward),
                                                                      sum_q / float(j) if j > 0 else 0))
                    print(f'diff:   a: {(env.r_target_state[0] - env._currentOrbit.getA()) / 1e3},\n'
                          f'ex: {env.r_target_state[1] - env._currentOrbit.getEquinoctialEx()},\t'
                          f'ey: {env.r_target_state[2] - env._currentOrbit.getEquinoctialEy()},\n'
                          f'hx: {env.r_target_state[3] - env._currentOrbit.getHx()},\t'
                          f'hy: {env.r_target_state[4] - env._currentOrbit.getHy()}\n'
                          f'Fuel Mass: {env.cuf_fuel_mass}\n'
                          f'Final Orbit:{env._currentOrbit}\n'
                          f'Initial Orbit:{env._orbit}')
                    print('=========================')

                    if save_fig:
                        np.save('results/rewards.npy', np.array(rewards))

                    # Save model checkpoint
                    manager.save()

                    if env.target_hit:
                        n = range(j + 1)
                        save_fig = True
                        env.render_target()

                        if 0 <= i < 10:
                            episode = '00' + str(i)
                        elif 10 <= i < 100:
                            episode = '0' + str(i)
                        elif i >= 100:
                            episode = str(i)

                        env.render_plots(i, save=save_fig, show=show)
                        plot_thrust(actions, episode, n, save_fig, show)
                        plot_reward(episode, rewards, save_fig, show)

                    break

            # Periodically render and save plots
            if i % 10 == 0:
                n = range(j + 1)
                save_fig = True if i % 10 == 0 and save_fig else False
                show = True if i % 50 == 0 and show else False

                env.render_target()
                env.render_plots(i, save=save_fig, show=show)

                if 0 <= i < 10:
                    episode = '00' + str(i)
                elif 10 <= i < 100:
                    episode = '0' + str(i)
                elif i >= 100:
                    episode = str(i)

                plot_thrust(actions, episode, n, save_fig, show)
                plot_reward(episode, rewards, save_fig, show)

    else:
        # Test mode
        if args['model'] is not None:
            if manager.latest_checkpoint:
                checkpoint.restore(manager.latest_checkpoint)
                print(f"Restored from {manager.latest_checkpoint}")

                env.render_target()

                for i in range(num_episodes):
                    s = env.reset()
                    sum_reward = 0
                    actions = []

                    for j in range(iter_per_episode):
                        # Convert state to tensor and predict action
                        s_tensor = tf.convert_to_tensor(np.reshape(s, (1, features)), dtype=tf.float32)
                        a = actor.predict(s_tensor).numpy()

                        s1, r, done = env.step(a[0])
                        s = s1
                        sum_reward += r
                        actions.append(a[0])

                        if done or j >= iter_per_episode - 1:
                            print(f'Episode: {i}, reward: {int(sum_reward)}')
                            n = range(j + 1)
                            env.render_plots()
                            plot_thrust(actions=actions, episode='00', n=n, save_fig=save_fig, show=show)
                            break
            else:
                print('No checkpoint found at {}'.format(checkpoint_path))
                exit(-1)
        else:
            print('Cannot run non-existent model', file=sys.stderr)
            exit(-1)


def model_saving(ENV, args):
    """Configure model saving/loading directories and parameters"""
    LOAD = False
    if args['model'] is not None:
        checkpoint_path = args['model']
        if args['test']:
            TRAIN = False
        else:
            TRAIN = True
            LOAD = True
    else:
        TRAIN = True
        today = datetime.date.today()
        path = 'models/'
        checkpoint_path = path + str(today) + '-' + ENV
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f'Model will be saved in: {checkpoint_path}')

    if args['savefig']:
        save_fig = True
        if os.path.exists('results/rewards.npy'):
            load_reward = np.load('results/rewards.npy')
            rewards = np.ndarray.tolist(load_reward)
        else:
            rewards = []
    else:
        save_fig = False
        rewards = []

    if args['showfig']:
        show = True
    else:
        show = False

    return LOAD, TRAIN, checkpoint_path, rewards, save_fig, show


def plot_reward(episode, rewards, save_fig, show):
    """Plot rewards over episodes"""
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    if save_fig:
        plt.savefig('results/' + episode + '/Rewards.pdf')
    if show:
        plt.show()


def plot_thrust(actions, episode, n, save_fig, show):
    """Plot thrust magnitude and components"""
    thrust_mag = np.linalg.norm(np.asarray(actions), axis=1)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(thrust_mag)
    plt.title('Thrust Magnitude (N)')
    plt.subplot(2, 2, 2)
    plt.plot(n, np.asarray(actions)[:, 0])
    plt.title('Thrust Magnitude (R)')
    plt.subplot(2, 2, 3)
    plt.plot(n, np.asarray(actions)[:, 1])
    plt.title('Thrust Magnitude (S)')
    plt.xlabel('Mission Step ' + str(stepT) + ' sec per step')
    plt.subplot(2, 2, 4)
    plt.plot(n, np.asarray(actions)[:, 2])
    plt.title('Thrust Magnitude (W)')
    plt.xlabel('Mission Step ' + str(stepT) + ' sec per step')
    plt.tight_layout()
    if save_fig:
        plt.savefig('results/' + episode + '/thrust.pdf')
        np.save('results/' + episode + '/' + 'thrust.npy', np.asarray(actions))
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="path of a trained tensorlfow model (str: path)", type=str)
    parser.add_argument('--test', help="pass if testing a model", action='store_true')
    parser.add_argument('--savefig', help="Save figures to file", action='store_true')
    parser.add_argument('--showfig', help='Display plotted figures', action='store_true')
    args = vars(parser.parse_args())
    main(args)