'''create test set and test model'''
from datetime import date
import os
from pickletools import optimize
import sys
import random
import pickle
from typing import Tuple, Union
import warnings
from pathlib import Path
import csv
import ray

import time

warnings.simplefilter("ignore", UserWarning)
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
# import ray.util.multiprocessing as mp
from environment import Environment
from model import Network
import config

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)


def create_test(test_env_settings: Tuple = config.test_env_settings, num_test_cases: int = config.num_test_cases):
    '''
    create test set
    '''

    for map_length, num_agents, density in test_env_settings:

        name = f'./test_set/{map_length}length_{num_agents}agents_{density}density.pth'
        print(f'-----{map_length}length {num_agents}agents {density}density-----')

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)

def core_test_model(network, model_name, datetime, test_set, pool=None):
    print(f'model name: {model_name}')
    state_dict = torch.load(os.path.join(config.test_model_path, f'{datetime}/{model_name}.pth'),
                        map_location=DEVICE)
    network.load_state_dict(state_dict)
    network.eval()
    network.share_memory()

    print(f'----------test model {model_name}----------')

    should_skip = []

    # write into file
    filepath = f'results/{datetime}_output.csv'
    output_file = Path(filepath)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(filepath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['map_size', 'num_agents', 'density', 'success_rate', 'avg_steps', 'commu_times'])

    start_time = time.time()

    for case in test_set:
        map_size = case[0]
        if map_size in should_skip:
            print(f'test with map size: {case[0]} skipped')
            continue

        print(f"test set: {case[0]} length {case[1]} agents {case[2]} density")
        with open(f'./test_set/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
            tests = pickle.load(f)

        tests = [(test, network) for test in tests]

        if pool is not None:
            # using mp
            ret = pool.map(test_one_case, tests)
            success, steps, num_comm = zip(*ret)
        else:
            # without mp
            success = np.zeros(len(tests))
            steps = np.zeros(len(tests))
            num_comm = np.zeros(len(tests))

            for id, test in enumerate(tests):
                su, st, comm = test_one_case(test)

                success[id] = su
                steps[id] = st
                num_comm[id] = comm

        success_rate = sum(success) / len(success) * 100
        avg_steps = sum(steps) / len(steps)
        commu_times = sum(num_comm) / len(num_comm)

        with open(filepath, "a") as f:
            writer = csv.writer(f)
            writer.writerow([case[0], case[1], case[2], success_rate, avg_steps, commu_times])
            # writer.writerow([f"test set: {case[0]} length {case[1]} agents {case[2]} density: ",
            #     f'{success_rate}', f'{avg_steps}', f'{commu_times}'])

        print("success rate: {:.2f}%".format(success_rate))
        print(f"average step: {avg_steps}")
        print(f"communication times: {commu_times}")

        if avg_steps == 256 or success_rate == 0:
            print(f'max steps reached, skipping other test cases with the same map size')

            should_skip.append(map_size)
            # test_set = remove_map(list(test_set), map_size)
            continue

        print()

    print(f'time used for testing model {model_name}: \
            {(time.time() - start_time)/60} mins')
    print('')

@ray.remote(num_cpus=2)
def ray_test_model(model_range: Union[int, tuple], interval, datetime: str, test_set: Tuple = config.test_env_settings):
    '''
    test model in 'saved_models' folder
    '''
    network = Network()
    network.eval()
    network.to(DEVICE)

    print('not using torch.mp')

    print(f'testing model ranging from {model_range[0]} to {model_range[1]}')

    for model_name in range(model_range[0], model_range[1] + 1, interval):
        core_test_model(network=network, model_name=model_name, datetime=datetime, test_set=test_set)
        print('\n')


def test_model(model_range: Union[int, tuple], interval, datetime: str, test_set: Tuple = config.test_env_settings):
    '''
    test model in 'saved_models' folder
    '''
    network = Network()
    network.eval()
    network.to(DEVICE)

    pool = mp.Pool(mp.cpu_count()//2) # don't run this in ICDC server
    print(f'number cpu used in pool: {mp.cpu_count()//2}')
    print('using torch.mp')

    print(f'testing model ranging from {model_range[0]} to {model_range[1]}')

    for model_name in range(model_range[0], model_range[1] + 1, interval):
        core_test_model(network=network, model_name=model_name, datetime=datetime, test_set=test_set, pool=pool)
        print('\n')

def test_one_case(args):
    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, last_act, pos = env.observe()

    done = False
    network.reset()

    step = 0
    num_comm = 0
    while not done and env.steps < config.max_episode_length:
        actions, _, _, _, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE),
                                                   torch.as_tensor(last_act.astype(np.float32)).to(DEVICE),
                                                   torch.as_tensor(pos.astype(np.int)))
        (obs, last_act, pos), _, done, _ = env.step(actions)
        step += 1
        num_comm += np.sum(comm_mask)

    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm


def code_test():
    env = Environment()
    network = Network()
    network.eval()
    obs, last_act, pos = env.observe()
    # print(f'obs: {obs}')
    # print(f'last_act: {last_act}')
    # print(f'pos: {pos}')
    network.step(torch.as_tensor(obs.astype(np.float32)).to(DEVICE),
                 torch.as_tensor(last_act.astype(np.float32)).to(DEVICE),
                 torch.as_tensor(pos.astype(np.int)))

@ray.remote
def foo(some_str):
    print(some_str)
    time.sleep(1)

@ray.remote(num_gpus=1)
class Foo(object):
    def __init__(self):
        self.value = 0

    def run(self):
        # save model via threading
        from threading import Thread
        thread = Thread(target=self.save_model, args=())
        thread.start()

    def save_model(self):
        self.value += 1
        print(f'say sth')

        # create dir
        model = Network()
        # model.to('cuda')

        path = os.path.join(os.getcwd(), f'{config.save_path}')
        print(f'cwd: {os.getcwd()}')
        print(f'path to save: {path}')

        if not os.path.exists(path):
            print('path does not exist, creating')
            os.mkdir(path)
        else:
            print('path exists')

        for i in range(10):
            print(f'saving model {i}')
            dict = torch.save(model.state_dict(), os.path.join(config.save_path, f'test_{i}.pth'))

        print(f'dict type: {type(dict)}')

        # return self.value

    def get_attr(self, attr):
        return getattr(self, attr)

if __name__ == '__main__':
    start_time = time.time()

    start_range = int(sys.argv[1])
    end_range = int(sys.argv[2])
    interval = int(sys.argv[3])
    # datetime = sys.argv[4]
    datetime = '22-07-29_at_17.40.07' # do not use "", use ''

    print(f'testing spec')
    print(f'start range: {start_range}, end range: {end_range}')
    print(f'datetime: {datetime}')

    # # test with ray
    # print(f'testing model with ray')
    # ray.init()
    # # ray.get(foo.remote('testing ray remote func'))
    # ray.get(ray_test_model.remote(model_range=(start_range, end_range), interval=interval, datetime='22-08-02_at_01.13.25'))

    # test without ray
    print(f'testing model without ray')
    test_model(model_range=(start_range, end_range), interval=interval, datetime=datetime)

    # # other tests
    # # create dir
    # path = os.path.join(os.getcwd(), f'{config.save_path}')
    # # print(f'cwd: {os.getcwd()0, path to save: {path}}')

    # if not os.path.exists(path):
    #     os.mkdir(path)

    # model = Network()
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    # loss = 0.1

    # torch.save(model.state_dict(), os.path.join(f'{path}', f'{999}.pth'))
    # torch.save({
    #     'epoch': 999,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': loss,
    # }, os.path.join(f'{path}', f'{999}.pt'))

    # print('model saved at step {}'.format(999))

    print(f'time used for testing range ({start_range}-{end_range}): {(time.time() - start_time)/60}')

    print()
