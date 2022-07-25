import os
import sys
import random
import time
import torch
import numpy as np
import ray
from worker import GlobalBuffer, Learner, Actor
import config

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(num_actors=config.num_actors, log_interval=config.log_interval):
    num_cpus = int(sys.argv[1])
    # address = sys.argv[2]

    # ray.init()
    ray.init(address=os.environ["ip_head"])

    buffer = GlobalBuffer.remote()
    learner = Learner.remote(buffer)
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
    learner.run.remote()
    
    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        ray.get(buffer.stats.remote(log_interval))
        print()

if __name__ == '__main__':
    main()
