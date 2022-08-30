import os
import sys
import random
import time
import torch
import numpy as np
import ray
import worker
from worker import GlobalBuffer, Learner, Actor#, ModelSaver
import config
from datetime import datetime
import logging

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(num_actors=config.num_actors, log_interval=config.log_interval, ckpt_path=None):

    ray.init()
    # ray.init(address=os.environ["ip_head"])

    if ckpt_path is not None:
        # load from checkpoint
        checkpoint = torch.load(os.path.join('./saved_models', ckpt_path), map_location='cuda')
        curriculum = checkpoint['curriculum_stat_dict']

        # print(f'curriculum: {curriculum}')
        # print(f'list(curriculum.keys()): {list(curriculum.keys())}')   # put this as env.env_settings_set

        buffer = GlobalBuffer.remote(stat_dict=curriculum)
        learner = Learner.remote(buffer)
        learner.update_from_last_model.remote(checkpoint)

    else:
        # continue without loading checkpoint
        buffer = GlobalBuffer.remote(init_env_settings=config.init_env_settings)
        learner = Learner.remote(buffer)

    time.sleep(1)

    if num_actors == 1:
        # for testing
        actor = Actor.remote(1, 0.4, learner, buffer)
        actor.run.remote()
    else:
        actors = [Actor.remote(i, 0.4**(1+(i/(num_actors-1))*7), learner, buffer) for i in range(num_actors)]

        print(f'testing actor input params')
        for i in range(num_actors):
            # formula in "distributed PER" paper
            print(f'0.4**(1+(i/(num_actors-1))*7): {0.4**(1+(i/(num_actors-1))*7)}')

        for actor in actors:
            actor.run.remote()

    while not ray.get(buffer.ready.remote()): # when buffer length == config.learning_starts
        time.sleep(5)
        ray.get(learner.stats.remote(5))
        ray.get(buffer.stats.remote(5))

    print('start training')
    buffer.run.remote()

    # from threading import Thread
    # learning_thread = Thread(target=learner._train.remote(), daemon=True)
    # learning_thread.start()

    ray.get(learner.run.remote())

    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        ray.get(buffer.stats.remote(log_interval))

        # print(f'worker global state: {worker.global_state_dict}')

        # data = ray.get(learner.get_attr.remote('temp_state_dict'))
        # print(f'worker arg state: {data}')

        # result = ray.get(learner.get_attr.remote('temp_state_dict'))
        # print(f'learner.temp_state_dict: {result}')

        print()

if __name__ == '__main__':
    # selective communication
    path = os.path.join(os.getcwd(), f'slurm/debug/selectivecomm/{config.time}')
    torch.set_printoptions(edgeitems=config.obs_radius+1)

    # logging setup
    logging.basicConfig(level=logging.INFO, filename=path)

    # num_cpus = int(sys.argv[1])
    # # address = sys.argv[2]
    # print(f'train.py args: {sys.argv}')

    # config.num_actors = num_cpus
    # config.training_steps = round(2400000/config.num_actors)
    # # config.save_interval = round(config.training_steps * 0.05)
    # config.save_interval = 300

    # print(f'updated save_interval: {config.save_interval}, training_steps: {config.training_steps}')

    print(f'training start time: {datetime.now().strftime("%y-%m-%d_at_%H.%M.%S")}')

    try:
        # train from checkpoint
        ckpt_path = sys.argv[1]
        if config.num_actors > 6:
            main(num_actors=config.num_actors-5, ckpt_path=ckpt_path)
        else:
            main(num_actors=config.num_actors, ckpt_path=ckpt_path)
    except IndexError:
        print('no checkpoint provided by SLURM script')

        # train from scratch
        if config.num_actors > 6:
            main(num_actors=config.num_actors-5)
        else:
            main(num_actors=config.num_actors)

    print(f'training end time: {datetime.now().strftime("%y-%m-%d_at_%H.%M.%S")}')