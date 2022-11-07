'''create test set and test model'''
from datetime import date
import os
from pickletools import optimize
from sched import scheduler
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
# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'DEVICE: {DEVICE}')

torch.set_num_threads(1)


def create_test(obs_radius, test_env_settings: Tuple = config.test_env_settings, num_test_cases: int = config.num_test_cases):
    '''
    create test set
    '''

    for map_length, num_agents, density in test_env_settings:

        name = f'./test_set/r{obs_radius}/{map_length}length_{num_agents}agents_{density}density.pth'
        print(f'-----{map_length}length {num_agents}agents {density}density-----')

        tests = []

        env = Environment(obs_radius=obs_radius, fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()

        with open(name, 'wb') as f:
            pickle.dump(tests, f)

def core_test_model(obs_radius, network, model_name, datetime, test_set, pool=None):
    """
    run test for 1 model
    """
    print(f'model name: {model_name}')
    state_dict = torch.load(os.path.join(config.test_model_path, f'{datetime}/{model_name}.pth'),
                        map_location=DEVICE)

    # print("Model's state_dict:")
    # for param_tensor in state_dict:
    #     print(param_tensor, "\t", state_dict[param_tensor].size())

    network.load_state_dict(state_dict)
    network.to(DEVICE)
    network.eval()
    network.share_memory()

    print(f'----------test model {model_name}----------')

    should_skip = []

    # write into file
    filepath = f'results/{datetime}_output.csv'
    output_file = Path(filepath)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(filepath, "a") as f:
        writer = csv.writer(f)
        writer.writerow(['training_steps', 'map_size', 'num_agents', 'density', 'success_rate', 'avg_steps', 'commu_times',
                         # created by myself
                         'possible_commu', 'avg_steps_processed_per_second', 'avg_network_step_time',
                         # sub-process inside network.step()
                         'avg_before_select_comm_time','avg_select_comm_time', 'avg_after_select_comm_time'
                         ])

    start_time = time.time()

    for case in test_set:
        map_size = case[0]
        if map_size in should_skip:
            print(f'test with map size: {case[0]} skipped')
            continue

        print(f"test set: {case[0]} length {case[1]} agents {case[2]} density")
        with open(f'./test_set/r{obs_radius}/{case[0]}length_{case[1]}agents_{case[2]}density.pth', 'rb') as f:
            tests = pickle.load(f)

        tests = [(test, network, obs_radius) for test in tests]

        if pool is not None:
            # using mp
            ret = pool.map(test_one_case, tests)
            success, steps, num_comm, num_possible_comm = zip(*ret)
        else:
            # without mp
            success = np.zeros(len(tests))
            steps = np.zeros(len(tests))
            num_comm = np.zeros(len(tests))
            num_possible_comm = np.zeros(len(tests))
            steps_processed_per_second = np.zeros(len(tests))

            network_step_time = np.zeros(len(tests))
            before_select_comm_time = np.zeros(len(tests))
            select_comm_time = np.zeros(len(tests))
            after_select_comm_time = np.zeros(len(tests))

            for id, test in enumerate(tests):
                # refer to above names
                su, st, comm, possible_comm, spps, nst, bsct, sct, asct = test_one_case(test)

                success[id] = su
                steps[id] = st
                num_comm[id] = comm
                num_possible_comm[id] = possible_comm
                steps_processed_per_second[id] = spps
                network_step_time[id] = nst
                before_select_comm_time[id] = bsct
                select_comm_time[id] = sct
                after_select_comm_time[id] = asct

        success_rate = sum(success) / len(success) * 100
        avg_steps = sum(steps) / len(steps)
        commu_times = sum(num_comm) / len(num_comm)
        possible_commu = sum(num_possible_comm) / len(num_possible_comm)
        avg_steps_processed_per_second = sum(steps_processed_per_second) / len(steps_processed_per_second)

        avg_network_step_time = sum(network_step_time) / len(network_step_time)
        avg_before_select_comm_time = sum(before_select_comm_time) / len(before_select_comm_time)
        avg_select_comm_time = sum(select_comm_time) / len(select_comm_time)
        avg_after_select_comm_time = sum(after_select_comm_time) / len(after_select_comm_time)

        # # useless
        # ecr = commu_times/possible_commu

        with open(filepath, "a") as f:
            writer = csv.writer(f)
            writer.writerow([model_name, case[0], case[1], case[2], success_rate, avg_steps, commu_times, possible_commu,
                             avg_steps_processed_per_second, avg_network_step_time, avg_before_select_comm_time,
                             avg_select_comm_time, avg_after_select_comm_time])
            # writer.writerow([f"test set: {case[0]} length {case[1]} agents {case[2]} density: ",
            #     f'{success_rate}', f'{avg_steps}', f'{commu_times}'])

        print("success rate: {:.2f}%".format(success_rate))
        print(f"average step: {avg_steps}")
        print(f"actual communication times: {commu_times}")
        print(f"possible communication: {possible_commu}")
        print(f'avg_steps_processed_per_second: {avg_steps_processed_per_second}')

        print(f'avg_network_step_time: {avg_network_step_time}')
        print(f'avg_before_select_comm_time: {avg_before_select_comm_time}')
        print(f'avg_select_comm_time: {avg_select_comm_time}')
        print(f'avg_after_select_comm_time: {avg_after_select_comm_time}')
        # print(f'ECR: {ecr}')

        if avg_steps == 256 or success_rate == 0:
            print(f'max steps reached, skipping other test cases with the same map size')

            should_skip.append(map_size)
            # test_set = remove_map(list(test_set), map_size)
            continue

        print()

    print('')
    print(f'time used for testing model {model_name}: \
            {(time.time() - start_time)/60} mins')
    print('')

@ray.remote(num_gpus=2)
def ray_test_model(obs_radius: int, model_range: Union[int, tuple], interval, datetime: str, test_set: Tuple = config.test_env_settings):
    '''
    test model in 'saved_models' folder
    '''
    network = Network(obs_radius=obs_radius)
    network.eval()
    network.to(DEVICE)

    print('using ray and not using torch.mp for testing')

    print(f'testing model ranging from {model_range[0]} to {model_range[1]}')

    for model_name in range(model_range[1], model_range[0] - 1, -interval):
        core_test_model(obs_radius, network=network, model_name=model_name, datetime=datetime, test_set=test_set)
        print('\n')

# when not using ray
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
    env_set, network, obs_radius = args

    env = Environment(obs_radius=obs_radius)
    env.load(env_set[0], env_set[1], env_set[2]) # map, agent pos, goal pos
    obs, last_act, pos = env.observe()

    done = False
    network.reset()

    step = 0
    num_comm = 0
    num_possible_comm = 0

    # time
    network_step_time = 0 # only accumulate time used in network.step
    episode_start_time = time.time()

    while not done and env.steps < config.max_episode_length:
        obs_tensor = torch.as_tensor(obs.astype(np.float32)).to(DEVICE)
        last_act_tensor = torch.as_tensor(last_act.astype(np.float32)).to(DEVICE)
        pos_tensor = torch.as_tensor(pos.astype(np.int)).to(DEVICE)

        start_step_time = time.time()
        actions, _, _, _, comm_mask, in_obs_mask = network.step(obs_tensor, last_act_tensor, pos_tensor)
        # print(f'network.step() time: {time.time() - network_step_time}')
        # 0.02s
        network_step_time += time.time() - start_step_time

        env_step_time = time.time()
        (obs, last_act, pos), _, done, _ = env.step(actions)
        # print(f'env step time: {time.time() - env_step_time}')
        # 0.0003s

        step += 1
        num_comm += np.sum(comm_mask)
        num_possible_comm += np.sum(in_obs_mask)

    # calculate steps processed per second
    total_time_in_episode = time.time() - episode_start_time
    steps_processed_per_second = step / total_time_in_episode

    # avg_network_step_time = network_step_time / step
    episodic_avg_before_select_comm_time = network.before_select_comm_time / step
    episodic_avg_select_comm_time = network.select_comm_time / step
    episodic_avg_after_select_comm_time = network.after_select_comm_time / step
    # print(f'Total time in this episode: {total_time_in_episode}, number of commu: {num_comm}')
    # print(f'Avg network step time: {avg_network_step_time}')
    # print(f'(in network step) Avg Before select comm: {avg_before_select_comm_time}')
    # print(f'(in network step) Avg At select comm: {avg_select_comm_time}')
    # print(f'(in network step) Avg After select comm: {avg_after_select_comm_time}')
    #
    # # result = np.array_equal(env.agents_pos, env.goals_pos)
    # # success_rate = sum(result) / len(result) * 100
    # # print(f'success rate: {success_rate}')
    # print(f'steps in this episode: {step}')
    # print(f'num_comm: {num_comm}')
    # print(f'')
    #
    # exit()

    return np.array_equal(env.agents_pos, env.goals_pos), step, num_comm, num_possible_comm, steps_processed_per_second, \
           network_step_time, episodic_avg_before_select_comm_time, episodic_avg_select_comm_time, episodic_avg_after_select_comm_time

@ray.remote(num_gpus=2)
def code_test(obs_radius):
    env = Environment(obs_radius)
    network = Network(obs_radius)
    network.eval()
    network.to(DEVICE)
    obs, last_act, pos = env.observe()
    # print(f'obs: {obs}')
    # print(f'last_act: {last_act}')
    # print(f'pos: {pos}')

    obs_tensor = torch.as_tensor(obs.astype(np.float32)).to(DEVICE)
    last_act_tensor = torch.as_tensor(last_act.astype(np.float32)).to(DEVICE)
    pos_tensor = torch.as_tensor(pos.astype(np.int)).to(DEVICE)

    print(f'obs device: {obs_tensor.get_device()}')
    print(f'last act device: {last_act_tensor.get_device()}')
    print(f'pos device: {pos_tensor.get_device()}')

    # original
    network.step(obs_tensor, last_act_tensor, torch.as_tensor(pos.astype(np.int)))

    # adding pos to device
    network.step(obs_tensor, last_act_tensor, pos_tensor)

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

    obs_radius = int(sys.argv[1])

    if obs_radius == 1:
        model_train_time='22-08-11_at_22.45.38'
        start_range=105000
        end_range=105000
        interval=15000
    elif obs_radius == 2:
        model_train_time='22-08-08_at_23.37.56'
        start_range=120000 # 30-40mins per test, rmb to adjust time limit
        end_range=120000
        interval=15000
    elif obs_radius == 3:
        model_train_time = '22-08-03_at_20.55.01-r3' # forget about this -r3 tag, it was a mistake
        start_range = 90000  # 30-40mins per test, rmb to adjust time limit
        end_range = 90000
        interval = 15000
    elif obs_radius == 4:
        model_train_time='22-07-26_at_18.43.26'
        start_range=120000 # 30-40mins per test, rmb to adjust time limit
        end_range=120000
        interval=15000
    elif obs_radius == 5:
        model_train_time='22-08-02_at_01.13.25'
        start_range=90000 # 30-40mins per test, rmb to adjust time limit
        end_range=90000
        interval=15000
    else:
        raise ValueError('obs radius not recognised')

    # start_range = int(sys.argv[1])
    # end_range = int(sys.argv[2])
    # interval = int(sys.argv[3])
    # datetime = sys.argv[4]

    print(f'spec of the models for the current test')
    print(f'obs radius: {obs_radius}')
    print(f'start range: {start_range}, end range: {end_range}')
    print(f'model trained at: {model_train_time}')

    # # test with ray
    print(f'testing model with ray')
    ray.init()
    # ray.get(foo.remote('testing ray remote func'))
    # ray.get(code_test.remote())
    ray.get(ray_test_model.remote(obs_radius=obs_radius, model_range=(start_range, end_range), interval=interval, datetime=model_train_time))

    # test without ray
    # print(f'testing model without ray')
    # test_model(model_range=(start_range, end_range), interval=interval, datetime=datetime)

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

# if __name__ == "__main__":
#     # 1. create test case
#     obs_radius = int(sys.argv[1])
#     create_test(obs_radius)

#     # # 2. test if other params are saved
#     # checkpoint = torch.load(os.path.join(config.test_model_path, f'22-08-04_at_18.13.20/100.pt'),
#     #                     map_location='cuda')
#     # # model.load_state_dict(checkpoint['model_state_dict'])
#     # # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     # schedule = checkpoint['curriculum_stat_dict']
#     # print(f'schedule: {schedule}')

#     # 3.
#     path = '22-08-08_at_23.37.56/135000.pt' # r2 normal
#     checkpoint = torch.load(os.path.join(config.test_model_path, path),
#                         map_location='cuda')
#     curriculum = checkpoint['curriculum_stat_dict']

#     # print the status of curriculum learning
#     for num_agents in range(config.init_env_settings[0], config.max_num_agents+1):
#         # num agent
#         print('{:2d}'.format(num_agents), end='')

#         # stat dict
#         for map_len in range(config.init_env_settings[1], config.max_map_lenght+1, 5):
#             if (num_agents, map_len) in curriculum:
#                 print('{:4d}/{:<3d}'.format(sum(curriculum[(num_agents, map_len)]), len(curriculum[(num_agents, map_len)])), end='')
#             else:
#                 print('   N/A  ', end='')
#         print()
