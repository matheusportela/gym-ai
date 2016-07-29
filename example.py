import gym


def main():
    env = gym.make('Pong-v0')
    print env.action_space
    print env.observation_space

    observation = env.reset()

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
