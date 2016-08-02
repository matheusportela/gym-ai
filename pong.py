import gym
import matplotlib.pylab as plt
import numpy as np

plt.ion()


class ObjectLocator(object):
    """Locates an object center of mass using its RGB color."""
    def __init__(self, color, min_y=34, max_y=193):
        """Parameters:
        color -- RGB color to locate, in numpy array format.
        min_y -- Minimum Y coordinate to consider when locating the object.
        max_y -- Maximum Y coordinate to consider when locating the object.
        """
        self.color = color
        self.min_y = min_y
        self.max_y = max_y

    def locate(self, image):
        """Locate object's center of mass in RGB image.

        Parameters:
        image -- Numpy matrix with RGB channels.

        Return:
        Numpy array with (Y, X) coordinates.
        """
        cropped_image = self._crop(image)
        morphology = self._extract_morphology(cropped_image)

        try:
            top_left, bottom_right = self._find_extremes(morphology)
            return self._calculate_center(top_left, bottom_right)
        except ValueError:
            return None

    def _crop(self, image):
        return image[self.min_y:self.max_y, :, :]

    def _extract_morphology(self, image):
        morphology = None

        for i, color in enumerate(self.color):
            channel_image = image[:, :, i]
            mask = channel_image == color

            if morphology is None:
                morphology = mask
            else:
                morphology *= mask

        return morphology

    def _find_extremes(self, morphology):
        indices_x = np.array(range(morphology.shape[1])*morphology.shape[0]).reshape(morphology.shape)
        indices_y = np.array([[i]*morphology.shape[1] for i in range(morphology.shape[0])])

        min_x = np.min(indices_x[morphology != 0])
        max_x = np.max(indices_x[morphology != 0])
        min_y = np.min(indices_y[morphology != 0])
        max_y = np.max(indices_y[morphology != 0])

        return np.array([min_y, min_x]), np.array([max_y, max_x])

    def _calculate_center(self, top_left, bottom_right):
        center = (top_left + bottom_right)/2.0
        return center.astype(int)


class LearningAlgorithm(object):
    def learn(self, state, action, reward):
        """Learn from experience.

        Learn by applying the reward received when transitioning from the
        current state to the new one by executing an action.

        Parameters:
        state -- Agent state after executing the action.
        action -- Executed action.
        reward -- Reward received after executing the action.
        """
        raise (NotImplementedError,
            '%s does not implement "learn" method' % str(type(self)))

    def act(self, state):
        """Select an action for the given state.

        By exploiting learned model, the algorithm selects the best action to be
        executed by the agent.

        Parameters:
        state -- Agent state to select an action.
        """
        raise (NotImplementedError,
            '%s does not implement "act" method' % str(type(self)))


class QLearning(LearningAlgorithm):
    """Q-learning algorithm implementation.

    Q-learning is a model free reinforcement learning algorithm that tries and
    learning state values and chooses actions that maximize the expected
    discounted reward for the current state.
    """

    def __init__(self, initial_state=0, learning_rate=1, discount_factor=1,
        actions=None):
        """Constructor.

        Parameters:
        initial_state -- State where the algorithm begins.
        learning_rate -- Value in [0, 1] interval that determines how much of
            the new information overrides the previous value. Deterministic
            scenarios may have optimal results with learning rate of 1, which
            means the new information completely replaces the old one.
        discount_factor -- Value in [0, 1) interval that determines the
            importance of future rewards. 0 makes the agent myopic and greedy,
            trying to achieve higher rewards in the next step. Closer to 1
            makes the agent maximize long-term rewards. Although values of 1
            and higher are possible, it may make the expected discounted reward
            infinite or divergent.
        actions -- List of (hashable) actions available in the current
            environment.
        """
        super(QLearning, self).__init__()
        self.previous_state = initial_state
        self.q_values = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        if actions:
            self.actions = actions
        else:
            self.actions = []

    def __str__(self):
        """Generates Q-values string representation."""
        results = []
        results.append('Q-values\n')
        for state in self.q_values:
            results.append(str(state))
            for action in self.q_values[state]:
                results.append(str(self.q_values[state][action]))
                results.append('\t')
            results.append('\n')
        return ''.join(results)

    def update_state(self, state):
        """Update Q Learning current state.

        Parameters:
        state -- State to which the learning algorithm is going.
        """
        self.previous_state = state

    def initialize_unknown_state(self, state):
        """Initialize Q-values for states that were not previously seen.

        Parameters:
        state -- Environment state.
        """
        if state not in self.q_values:
            self.q_values[state] = {}
            for action_ in self.actions:
                self.q_values[state][action_] = 0.0

    def get_q_value(self, state, action):
        """Get the current estimated value for the state-action pair.

        Parameters:
        state -- Environment state.
        action -- Agent action.
        """
        self.initialize_unknown_state(state)
        return self.q_values[state][action]

    def set_q_value(self, state, action, value):
        """Set a new estimated value for the state-action pair.

        Parameters:
        state -- Environment state.
        action -- Agent action.
        value -- New estimated value.
        """
        self.q_values[state][action] = value

    def _get_max_action_from_list(self, state, action_list):
        """Get the action with maximum estimated value from the given list of
        actions.

        state -- Environment state.
        action_list -- Actions to be evaluated.
        """
        self.initialize_unknown_state(state)
        actions = filter(lambda a: a in action_list, self.q_values[state])
        values = [self.q_values[state][action] for action in actions]
        max_value = max(values)
        max_actions = [action
                       for action in actions
                       if self.q_values[state][action] == max_value]

        return np.random.choice(max_actions)

    def get_max_action(self, state):
        """Get the action with maximum estimated value.

        Parameters:
        state -- Environment state.
        """
        self.initialize_unknown_state(state)
        return self._get_max_action_from_list(state, self.actions)

    def get_max_q_value(self, state):
        max_action = self.get_max_action(state)
        return self.q_values[state][max_action]

    def learn(self, state, action, reward):
        """Learn by updating the (state, action) reward.

        Learn by applying the reward received when transitioning from the
        current state to the new one by executing an action.

        Parameters:
        state -- Agent state after executing the action.
        action -- Executed action.
        reward -- Reward received after executing the action.
        """
        old_value = self.get_q_value(self.previous_state, action)
        next_expected_value = self.get_max_q_value(state)
        new_value = (old_value + self.learning_rate*(reward + self.discount_factor*next_expected_value - old_value))
        self.set_q_value(self.previous_state, action, new_value)
        self.update_state(state)

    def act(self, state, legal_actions=None):
        """Select the best legal action for the given state.

        Parameters:
        state -- Agent state to select an action.
        legal_actions -- Actions allowed in the current state.

        Return:
        Action number.
        """
        if legal_actions:
            return self._get_max_action_from_list(state, legal_actions)
        return self._get_max_action_from_list(state, self.actions)


class StateEstimator(object):
    def __init__(self):
        self.right_player_locator = ObjectLocator(np.array([92, 186, 92]))
        self.previous_right_player_location = np.zeros(2)
        self.left_player_locator = ObjectLocator(np.array([213, 130, 74]))
        self.previous_left_player_location = np.zeros(2)
        self.ball_locator = ObjectLocator(np.array([236, 236, 236]))
        self.previous_ball_location = np.zeros(2)

    def estimate(self, observation):
        right_player_location = self.right_player_locator.locate(observation)

        if right_player_location is not None:
            right_player_velocity = right_player_location - self.previous_right_player_location
            self.previous_right_player_location = right_player_location
        else:
            right_player_velocity = None
            self.previous_right_player_location = np.zeros(2)

        left_player_location = self.left_player_locator.locate(observation)

        if left_player_location is not None:
            left_player_velocity = left_player_location - self.previous_left_player_location
            self.previous_left_player_location = left_player_location
        else:
            left_player_velocity = None
            self.previous_left_player_location = np.zeros(2)

        ball_location = self.ball_locator.locate(observation)

        if ball_location is not None:
            ball_velocity = ball_location - self.previous_ball_location
            self.previous_ball_location = ball_location
        else:
            ball_velocity = None
            self.previous_ball_location = np.zeros(2)

        # Convert to string because it's hashable
        state = str([
            right_player_location,
            right_player_velocity,
            left_player_location,
            left_player_velocity,
            ball_location,
            ball_velocity,
        ])

        return state


class Agent(object):
    def __init__(self, num_actions, exploration_rate=0.05):
        self.actions = range(num_actions)
        self.q_learner = QLearning(
            actions=self.actions,
            learning_rate=0.5,
            discount_factor=0.5,
        )
        self.exploration_rate = exploration_rate

    def learn(self, state, action, reward):
        self.q_learner.learn(state, action, reward)

    def act(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.actions)
        return self.q_learner.act(state)


def main():
    env = gym.make('Pong-v0')
    print env.action_space

    agent = Agent(env.action_space.n)
    state_estimator = StateEstimator()

    observation = env.reset()
    state = state_estimator.estimate(observation)

    game_number = 1
    print 'Game #{}'.format(game_number)

    episode_reward = 0

    is_rendering = False
    reward_history = []

    while True:
        if is_rendering:
            env.render()

        action = agent.act(state)
        observation, reward, done, info = env.step(action)
        state = state_estimator.estimate(observation)
        agent.learn(state, action, reward)

        episode_reward += reward

        if done:
            reward_history.append(episode_reward)
            plt.clf()
            plt.plot(reward_history, '-o')
            plt.show()
            plt.pause(0.005)

            print 'Episode reward:', episode_reward
            episode_reward = 0

            game_number += 1
            print 'Game #{}'.format(game_number)

            if game_number % 5 == 0:
                is_rendering = True
            else:
                is_rendering = False

            observation = env.reset()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print 'Exited'
