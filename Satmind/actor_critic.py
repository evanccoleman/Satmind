import numpy as np
import tensorflow as tf


class Actor:
    def __init__(self, features, n_actions, layer_1_nodes, layer_2_nodes, action_bound, tau, learning_rate, batch_size,
                 name):
        self.tau = tau
        self.action_bound = action_bound
        self.features = features
        self.n_actions = n_actions
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.name = name

        # Create main and target networks
        self.model = self._build_network(name)
        self.target_model = self._build_network(name + "_target")
        # Initialize target network with same weights
        self.target_model.set_weights(self.model.get_weights())

        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def _build_network(self, name):
        inputs = tf.keras.layers.Input(shape=(self.features,))

        # First layer
        x = tf.keras.layers.Dense(
            self.layer_1_nodes,
            kernel_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(self.layer_1_nodes),
                                                                   1 / np.sqrt(self.layer_1_nodes)),
            bias_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(self.layer_1_nodes),
                                                                 1 / np.sqrt(self.layer_1_nodes))
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Second layer
        x = tf.keras.layers.Dense(
            self.layer_2_nodes,
            kernel_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(self.layer_2_nodes),
                                                                   1 / np.sqrt(self.layer_2_nodes)),
            bias_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(self.layer_2_nodes),
                                                                 1 / np.sqrt(self.layer_2_nodes))
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Third layer (128 nodes)
        x = tf.keras.layers.Dense(
            128,
            kernel_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(128), 1 / np.sqrt(128)),
            bias_initializer=tf.keras.initializers.RandomUniform(-1 / np.sqrt(128), 1 / np.sqrt(128))
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Output layer
        output = tf.keras.layers.Dense(
            self.n_actions,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.RandomUniform(-0.003, 0.003)
        )(x)

        # Scale output to action bound
        scaled_output = tf.keras.layers.Lambda(
            lambda x: x * self.action_bound
        )(output)

        model = tf.keras.Model(inputs=inputs, outputs=scaled_output, name=name)
        return model

    def predict(self, state):
        """Predict actions using the main network"""
        return self.model(state)

    def predict_target(self, state):
        """Predict actions using the target network"""
        return self.target_model(state)

    def train(self, state, action_gradient):
        """Train the actor network using policy gradient"""
        with tf.GradientTape() as tape:
            actions = self.model(state, training=True)
            # Multiply by negative action gradient for policy gradient 
            actor_loss = -tf.reduce_mean(tf.reduce_sum(actions * action_gradient, axis=1))

        # Calculate and apply gradients
        actor_gradients = tape.gradient(actor_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_gradients, self.model.trainable_variables))

    def update_target_network(self):
        """Soft update target network weights: θ' ← τ*θ + (1-τ)*θ'"""
        for target_weight, weight in zip(self.target_model.weights, self.model.weights):
            target_weight.assign(self.tau * weight + (1 - self.tau) * target_weight)

    def __str__(self):
        return (f'Actor neural Network:\n'
                f'Inputs: {self.features} \t Actions: {self.n_actions} \t Action bound: {self.action_bound}\n'
                f'Layer 1 nodes: {self.layer_1_nodes} \t layer 2 nodes: {self.layer_2_nodes}\n'
                f'learning rate: {self.learning_rate} \t target network update (tau): {self.tau}\n')

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.features!r}, {self.n_actions!r},{self.layer_1_nodes!r}, {self.layer_2_nodes!r},'
                f'{self.action_bound!r}, {self.tau!r}, {self.learning_rate!r}, {self.batch_size!r}, {self.name!r})')


class Critic:
    def __init__(self, n_features, n_actions, layer_1_nodes, layer_2_nodes, learning_rate, tau, name,
                 actor_trainable_variables=None):
        self.tau = tau
        self.n_features = n_features
        self.n_actions = n_actions
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        self.learning_rate = learning_rate
        self.name = name
        # Note: actor_trainable_variables is no longer needed in TF2 but kept for compatibility

        # Create main and target networks
        self.model = self._build_network(name)
        self.target_model = self._build_network(name + "_target")
        # Initialize target network with same weights
        self.target_model.set_weights(self.model.get_weights())

        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def _build_network(self, name):
        state_input = tf.keras.layers.Input(shape=(self.n_features,))
        action_input = tf.keras.layers.Input(shape=(self.n_actions,))

        # Process state input
        x = tf.keras.layers.Dense(
            self.layer_1_nodes,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(state_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Concatenate state and action
        x = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(x), action_input])

        # Second layer after concatenation
        x = tf.keras.layers.Dense(
            self.layer_2_nodes,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)

        # Output layer (Q-value)
        output = tf.keras.layers.Dense(
            1,
            activation='linear',
            kernel_initializer=tf.keras.initializers.RandomUniform(-0.003, 0.003)
        )(x)

        model = tf.keras.Model(inputs=[state_input, action_input], outputs=output, name=name)
        return model

    def predict(self, state, action):
        """Predict Q-values using the main network"""
        return self.model([state, action])

    def predict_target(self, state, action):
        """Predict Q-values using the target network"""
        return self.target_model([state, action])

    def train(self, state, action, q_value, importance):
        """Train the critic network"""
        with tf.GradientTape() as tape:
            q_predicted = self.model([state, action], training=True)
            error = q_predicted - q_value
            # Calculate loss with importance sampling
            loss = tf.reduce_mean(tf.multiply(tf.square(error), importance))

        # Calculate and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return error, q_predicted, loss

    def action_gradient(self, state, action):
        """Calculate the gradient of Q-value with respect to actions"""
        with tf.GradientTape() as tape:
            tape.watch(action)
            q_value = self.model([state, action])

        return tape.gradient(q_value, action)

    def update_target_network(self):
        """Soft update target network weights: θ' ← τ*θ + (1-τ)*θ'"""
        for target_weight, weight in zip(self.target_model.weights, self.model.weights):
            target_weight.assign(self.tau * weight + (1 - self.tau) * target_weight)

    def __str__(self):
        return (f'Critic Neural Network:\n'
                f'Inputs: {self.n_features} \t Actions: {self.n_actions}\n'
                f'Layer 1 nodes: {self.layer_1_nodes} \t layer 2 nodes: {self.layer_2_nodes}\n'
                f'learning rate: {self.learning_rate} \t target network update (tau): {self.tau}')

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.n_features!r}, {self.n_actions!r},{self.layer_1_nodes!r}, {self.layer_2_nodes!r},'
                f'{self.learning_rate!r}, {self.tau!r}, {self.name!r})')