import gym
import numpy as np
import theano
import theano.tensor as T


class ObjectLocator(object):
    def __init__(self, color):
        self.color = color

    def locate(self, image):
        mask = self.generate_mask(image)
        morphology = self.extract_morphology(image, mask)
        top_left, bottom_right = self.find_extremes(morphology)
        return self.calculate_center(top_left, bottom_right)

    def generate_mask(self, image):
        dim_size = image.shape[0] * image.shape[1]
        mask_values = []

        for i in range(len(self.color)):
            mask_values += [self.color[i]] * dim_size

        mask = np.array(mask_values).reshape(image.shape)

        return mask

    def extract_morphology(self, image, mask):
        print image == mask
        return (image == mask).astype(int)[0]

    def find_extremes(self, morphology):
        indices_x = np.array(range(3)*3).reshape([3, 3])
        indices_y = indices_x.T

        min_x = np.min(indices_x[morphology != 0])
        max_x = np.max(indices_x[morphology != 0])
        min_y = np.min(indices_y[morphology != 0])
        max_y = np.max(indices_y[morphology != 0])

        return np.array([min_y, min_x]), np.array([max_y, max_x])

    def calculate_center(self, top_left, bottom_right):
        center = (top_left + bottom_right)/2.0
        return center.astype(int)


def main():
    env = gym.make('Pong-v0')
    print env.action_space
    print env.observation_space

    ball_locator = ObjectLocator(np.array([236, 236, 236]))

    observation = env.reset()
    print observation.shape
    print ball_locator.locate(observation)

    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())

        if done:
            print 'Finished'
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print 'Exited'
