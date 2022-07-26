import os
import sys
import random
import time
import torch
import numpy as np
import ray
from worker import GlobalBuffer, Learner, Actor, ModelSaver
import config

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(num_actors=config.num_actors, log_interval=config.log_interval):

    ray.init()
    # ray.init(address=os.environ["ip_head"])

    buffer = GlobalBuffer.remote()
    model_saver = ModelSaver.remote()
    learner = Learner.remote(buffer, model_saver)
    time.sleep(1)
    actors = [Actor.remote(i, 0.4**(1+(i/(num_actors-1))*7), learner, buffer) for i in range(num_cpus)]

    print(f'testing actor input params')
    for i in range(num_actors):
        print(f'0.4**(1+(i/(num_actors-1))*7): {0.4**(1+(i/(num_actors-1))*7)}')

    for actor in actors:
        actor.run.remote()

    while not ray.get(buffer.ready.remote()): # when buffer length == config.learning_starts
        time.sleep(5)
        ray.get(learner.stats.remote(5))
        ray.get(buffer.stats.remote(5))

    print('start training')
    buffer.run.remote()
    ray.get(learner.run.remote())

    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        ray.get(buffer.stats.remote(log_interval))
        print()

if __name__ == '__main__':
    num_cpus = int(sys.argv[1])
    # address = sys.argv[2]
    print(f'train.py args: {sys.argv}')

    config.num_actors = num_cpus
    config.training_steps = round(2400000/config.num_actors)
    # config.save_interval = round(config.training_steps * 0.05)
    config.save_interval = 300

    print(f'updated save_interval: {config.save_interval}, training_steps: {config.training_steps}')

    main(num_actors=config.num_actors-8)
