import numpy as np
import tensorflow as tf
from keras import layers, initializers
import random, sys, getopt, Globals
import math, csv, os, json
from collections import namedtuple

def copy_weights(source_model, target_model):

    if not target_model.built:
        dummy_input = tf.zeros((1, source_model.input_shape[-1]))  # Create a dummy input with correct shape
        target_model(dummy_input)  # Call the target model to build it

    target_model.set_weights(source_model.get_weights())


class NeuralNetModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions, initializer, activation_function='sigmoid'):
        super(NeuralNetModel, self).__init__()
        self.hidden_layers = [layers.Dense(h, activation=activation_function,
                                           kernel_initializer=initializer, bias_initializer=initializers.Zeros()) 
                              for h in hidden_units]
        self.output_layer = layers.Dense(num_actions, activation='linear', kernel_initializer=initializer,
                                         bias_initializer=initializers.Zeros())
    # Network forward pass
    @tf.function
    def call(self, inputs, return_activations=False, **kwargs):
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, 0)  # Add batch dimension
        x = inputs
        activations = []  # To store activations if requested
        for layer in self.hidden_layers:
            x = layer(x)
            if return_activations:
                activations.append(x)  # Store each layer's output if requested
        final_output = self.output_layer(x)
        if return_activations:
            activations.append(final_output)
            return activations  # Return all activations
        return final_output

    # Debugging method to log weights and activations
    def log_weights_and_activations(self, inputs, log_file="weights_activations.log"):
        """
        Append per-layer weight stats and activations for a single forward pass (debugging scale/saturation).
        """
        with open(log_file, "a") as f:  # Open the log file in append mode
            f.write("\nLogging Weights and Activations:\n")
            x = inputs
            for i, layer in enumerate(self.hidden_layers):
                weights, biases = layer.get_weights()
                f.write(f"Layer {i + 1}:\n")
                f.write(f"  Weights: Min={weights.min()}, Max={weights.max()}, Mean={weights.mean()}\n")
                f.write(f"  Biases: Min={biases.min()}, Max={biases.max()}, Mean={biases.mean()}\n")
                x = layer(x)
                f.write(f"  Activations: Min={x.numpy().min()}, Max={x.numpy().max()}, Mean={x.numpy().mean()}\n")
            final_output = self.output_layer(x)
            f.write(f"Output Layer Activations: Min={final_output.numpy().min()}, Max={final_output.numpy().max()}, Mean={final_output.numpy().mean()}\n")

# Replay Memory for Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)  # Returns the length of the memory list

# Epsilon Greedy Strategy
class EpsilonGreedyStrategy:
    def __init__(self):
        self.learning_start = Globals.learning_start
        self.beta = Globals.beta

    def get_exploration_rate(self, current_step):
        return math.exp((-1) * self.beta * max(0.0, current_step - self.learning_start))
    

# Build a simple feedforward policy model
def build_policy_model(input_dim=4, output_dim=21, hidden=[128,128]):
    inputs = tf.keras.Input((input_dim,))
    x = inputs
    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation=None)(x)
    return tf.keras.Model(inputs, outputs, name="policy_net")
