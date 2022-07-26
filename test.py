'''create test set and test model'''
import os
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
# import torch.multiprocessing as mp
import ray.util.multiprocessing as mp
from environment import Environment
from model import Network
import config

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

@ray.remote
def test_model(model_range: Union[int, tuple], datetime: str, test_set: Tuple = config.test_env_settings):
    '''
    test model in 'saved_models' folder
    '''
    network = Network()
    network.eval()
    network.to(DEVICE)

    print(f'cpu used for testing: {config.num_actors}')
    # pool = mp.Pool(mp.cpu_count()//2) # don't run this in ICDC server
    pool = mp.Pool(config.num_actors)

    print(f'testing network')

    # if isinstance(model_range, str):
    #     state_dict = torch.load(os.path.join(config.test_model_path, f'{datetime}/{model_range}.pth'),
    #                             map_location=DEVICE)
    #     network.load_state_dict(state_dict)
    #     network.eval()
    #     network.share_memory()

    #     print(f'----------test model {model_range}----------')

    #     should_skip = []

    #     for case in test_set:
    #         map_size = case[0]
    #         if map_size in should_skip:
    #             print(f'test with map size: {case[0]} skipped')
    #             continue

    #         print(f"test set: {case[0]} length {case[1]} agents {case[2]} density")
    #         with open('./test_set/{}length_{}agents_{}density.pth'.format(case[0], case[1], case[2]), 'rb') as f:
    #             tests = pickle.load(f)

    #         tests = [(test, network) for test in tests]
    #         ret = pool.map(test_one_case, tests)

    #         success, steps, num_comm = zip(*ret)

    #         if sum(steps) / len(steps) == 256:
    #             print(f'max steps reached, skipping other test cases with the same map size')

    #             should_skip.append(map_size)
    #             # test_set = remove_map(list(test_set), map_size)
    #             continue


    #         success_rate = sum(success) / len(success) * 100
    #         avg_steps = sum(steps) / len(steps)
    #         commu_times = sum(num_comm) / len(num_comm)


    #         print("success rate: {:.2f}%".format(success_rate)
    #         print("average step: {}".format(avg_steps))
    #         print("communication times: {}".format(commu_times))
    #         print()

    # elif isinstance(model_range, tuple):

    print(f'testing model ranging from {model_range[0]} to {model_range[1]}')

    for model_name in range(model_range[0], model_range[1] + 1, config.save_interval):
        state_dict = torch.load(os.path.join(config.test_model_path, f'{datetime}/{model_name}.pth'),
                            map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        print(f'----------test model {model_name}----------')

        should_skip = []

        for case in test_set:
            map_size = case[0]
            if map_size in should_skip:
                print(f'test with map size: {case[0]} skipped')
                continue

            print(f"test set: {case[0]} length {case[1]} agents {case[2]} density")
            with open(f'./test_set/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
                tests = pickle.load(f)

            tests = [(test, network) for test in tests]

            # using mp
            # ret = pool.map(test_one_case, tests)
            ret = test_one_case.remote(tests)
            success, steps, num_comm = zip(*ret)

            # # without mp
            # success = np.zeros(len(tests))
            # steps = np.zeros(len(tests))
            # num_comm = np.zeros(len(tests))
            #
            # for id, test in enumerate(tests):
            #     su, st, comm = test_one_case(test)
            #
            #     success[id] = su
            #     steps[id] = st
            #     num_comm[id] = comm

            success_rate = sum(success) / len(success) * 100
            avg_steps = sum(steps) / len(steps)
            commu_times = sum(num_comm) / len(num_comm)

            # write into file
            filepath = f'results/{datetime}_output.csv'
            output_file = Path(filepath)
            output_file.parent.mkdir(exist_ok=True, parents=True)

            with open(filepath, "w") as f:
                writer = csv.writer(f)
                writer.writerow([f"test set: {case[0]} length {case[1]} agents {case[2]} density: ",
                    f'{success_rate}', f'{avg_steps}', f'{commu_times}'])

            print("success rate: {:.2f}%".format(success_rate))
            print(f"average step: {avg_steps}")
            print(f"communication times: {commu_times}")

            if avg_steps == 256 or success_rate == 0:
                print(f'max steps reached, skipping other test cases with the same map size')

                should_skip.append(map_size)
                # test_set = remove_map(list(test_set), map_size)
                continue

            print()

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

# # remove test case of the same map size if max steps reached at fewer agents number
# def remove_map(array, size):
#     if len(array) == 0:
#         return []
#
#     element = array[0][0]
#     if element == size:
#         array.pop(0)
#         remove_map(array, size)
#     else:
#         return array

@ray.remote
def foo(some_str):
    print(some_str)
    time.sleep(1)

@ray.remote
class Foo(object):
    def __init__(self):
        self.value = 0

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

        dict = torch.save(model.state_dict(), os.path.join(config.save_path, f'{111}.pth'))
        print(f'dict type: {type(dict)}')

        # return self.value

if __name__ == '__main__':
    num_cpus = int(sys.argv[1])
    # address = sys.argv[2]
    print(f'test.py args: {sys.argv}')

    config.num_actors = num_cpus
    config.training_steps = round(2400000/config.num_actors)
    config.save_interval = round(config.training_steps * 0.05)

    print(f'updated save_interval: {config.save_interval}, training_steps: {config.training_steps}')

    # from datetime import datetime
    # time = datetime.now().strftime("%y-%m-%d at %H.%M.%S")
    # save_path = f'./saved_models/{time}'
    # print(save_path)

    # # test mkdir
    # cwd = os.getcwd()
    # path = os.path.join(cwd, config.save_path)
    # print(os.getcwd())
    # print(path)
    # os.mkdir(path) # windows

    # testing ray
    foo.remote('test str')

    def threadfunc():
        print('threadfunc')

    # save model via threading
    from threading import Thread
    thread = Thread(target=threadfunc, args=())
    thread.start()

    foo = Foo.remote()
    ray.get(foo.save_model.remote())



    # first trained model
    # test_model.remote(model_range=(150000, 150001), datetime='22-07-05_at_18.09.42')

    # obs radius = 4
    # test_model(model_range=(40000, 60000), datetime='22-07-21_at_17.42.12') # 667390
    # test_model(model_range=(80000, 90000), datetime='22-07-21_at_17.42.12') # 667392
    # test_model(model_range=(100000, 120000), datetime='22-07-21_at_17.42.12') # 667387
    # test_model(model_range=(130000, 150000), datetime='22-07-21_at_17.42.12')

    # obs radius = 5
    # test_model(model_range=(10000, 150000), datetime='22-07-23_at_13.16.32')

    # ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    # # pool = mp.Pool(mp.cpu_count()//2) # don't run this in ICDC server
    # print(f'using ncpu {ncpus} cpus')
    # print(f'using mp {mp.cpu_count()} cpus')


    print()
