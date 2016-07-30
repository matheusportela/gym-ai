import gym
import numpy as np


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


def print_locations(right_player_location, left_player_location,
                    ball_location):
    print 'Right player:', right_player_location
    print 'Left player:', left_player_location
    print 'Ball:', ball_location
    print


def main():
    env = gym.make('Pong-v0')

    right_player_locator = ObjectLocator(np.array([92, 186, 92]))
    left_player_locator = ObjectLocator(np.array([213, 130, 74]))
    ball_locator = ObjectLocator(np.array([236, 236, 236]))

    observation = env.reset()
    print_locations(right_player_locator.locate(observation),
                    left_player_locator.locate(observation),
                    ball_locator.locate(observation))

    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        print_locations(right_player_locator.locate(observation),
                        left_player_locator.locate(observation),
                        ball_locator.locate(observation))

        if done:
            print 'Finished'
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print 'Exited'
