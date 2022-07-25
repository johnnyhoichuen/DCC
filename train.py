import os
import random
import time
import torch
import numpy as np
from worker import GlobalBuffer, Learner, Actor
import config

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(num_actors=config.num_actors, log_interval=config.log_interval):
    buffer = GlobalBuffer()
    learner = Learner(buffer)
    time.sleep(1)
    actors = [Actor(i, 0.4**(1+(i/(num_actors-1))*7), learner, buffer) for i in range(num_actors)]

    print(f'actors\' epsilons')
    for i in range(num_actors):
        print(f'0.4**(1+(i/(num_actors-1))*7): {0.4**(1+(i/(num_actors-1))*7)}')

    for actor in actors:
        actor.run()

    while not buffer.ready(): # when buffer length == config.learning_starts
        time.sleep(5)
        learner.stats(5)
        buffer.stats(5)

    while not buffer.ready(): # when buffer length == config.learning_starts
        time.sleep(5)
        learner.stats(5)
        buffer.stats(5)

    print('start training')
    buffer.run()
    learner.run()

    done = False
    while not done:
        time.sleep(log_interval)
        done = learner.stats(log_interval)
        buffer.stats(log_interval)
        print()

if __name__ == '__main__':
    main()
