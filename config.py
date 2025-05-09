# config.py

# Reward settings
REWARD_SCALE      = 75     # Scaling factor for distance-based reward
GOAL_REWARD       = 700    # Reward when reaching goal
ALIGN_REWARD      = 3.0    # Bonus when moving in right direction
REPEAT_PENALTY    = 5.0    # Penalty for visiting same node
STEP_PENALTY      = 0.05   # Penalty per step to encourage shorter path
DEAD_END_PENALTY  = 30.0   # Penalty for reaching dead-end
MAX_REWARD        = 300
MIN_REWARD        = -300

# Training settings
MAX_STEPS_PER_EPISODE = 2000
TRAIN_INTERVAL         = 5
EVALUATE_AFTER_EPISODE = 15

# Agent settings
STATE_SIZE = 11
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 256
HIDDEN_SIZE3 = 128

# Memory
MEMORY_SIZE = 10000
BATCH_SIZE  = 64

MAX_ACTIONS = 12  # Max number of actions allowed per step

START_GOAL_MIN_DIST = 500
START_GOAL_MAX_DIST = 10000
