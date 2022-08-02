import time
import random
import os
from copy import deepcopy
from typing import Deque, List, Tuple
import threading
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler
import numpy as np
from model import Network
from environment import Environment
from buffer import SumTree, LocalBuffer, EpisodeData
import config

@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(self, buffer_capacity=config.buffer_capacity, init_env_settings=config.init_env_settings,
                alpha=config.prioritized_replay_alpha, beta=config.prioritized_replay_beta, chunk_capacity=config.chunk_capacity):

        self.capacity = buffer_capacity # 262144
        self.chunk_capacity = chunk_capacity # 64
        self.num_chunks = buffer_capacity // chunk_capacity # 4096 = 262144 // 64
        self.ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(buffer_capacity) # 262144
        self.alpha = alpha
        self.beta = beta

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {init_env_settings:[]}
        self.lock = threading.Lock()
        self.env_settings_set = ray.put([init_env_settings])

        self.obs_buf = [None] * self.num_chunks
        self.last_act_buf = [None] * self.num_chunks
        self.act_buf = np.zeros((buffer_capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((buffer_capacity), dtype=np.float16)
        self.hid_buf = [None] * self.num_chunks
        self.size_buf = np.zeros(self.num_chunks, dtype=np.uint8)
        self.relative_pos_buf = [None] * self.num_chunks
        self.comm_mask_buf = [None] * self.num_chunks
        self.gamma_buf = np.zeros((self.capacity), dtype=np.float16)
        self.num_agents_buf = np.zeros((self.num_chunks), dtype=np.uint8)

    def __len__(self):
        return np.sum(self.size_buf)

    def run(self):
        self.background_thread = threading.Thread(target=self._prepare_data, daemon=True)
        self.background_thread.start()

    def _prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self._sample_batch(config.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)

            else:
                time.sleep(0.1)

    def get_batched_data(self):
        '''
        get one batch of data, called by learner.
        '''

        if len(self.batched_data) == 0:
            print('no prepared data')
            data = self._sample_batch(config.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data: EpisodeData):
        '''
        Add one episode data into replay buffer, called by actor if actor finished one episode.

        data: actor_id 0, num_agents 1, map_len 2, obs_buf 3, act_buf 4, rew_buf 5,
                hid_buf 6, comm_mask_buf 8, gamma 9, td_errors 10, sizes 11, done 12
        '''
        if data.actor_id >= 9: #eps-greedy < 0.01
            stat_key = (data.num_agents, data.map_len)
            if stat_key in self.stat_dict:
                self.stat_dict[stat_key].append(data.done)
                if len(self.stat_dict[stat_key]) == config.cl_history_size+1:
                    self.stat_dict[stat_key].pop(0)

        with self.lock:

            for i, size in enumerate(data.sizes):
                idxes = np.arange(self.ptr*self.chunk_capacity, (self.ptr+1)*self.chunk_capacity)
                start_idx = self.ptr*self.chunk_capacity
                # update buffer size
                self.counter += size

                self.priority_tree.batch_update(idxes, data.td_errors[i*self.chunk_capacity:(i+1)*self.chunk_capacity]**self.alpha)

                self.obs_buf[self.ptr] = np.copy(data.obs[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.last_act_buf[self.ptr] = np.copy(data.last_act[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.act_buf[start_idx:start_idx+size] = data.actions[i*self.chunk_capacity:i*self.chunk_capacity+size]
                self.rew_buf[start_idx:start_idx+size] = data.rewards[i*self.chunk_capacity:i*self.chunk_capacity+size]
                self.hid_buf[self.ptr] = np.copy(data.hiddens[i*self.chunk_capacity:i*self.chunk_capacity+size+config.forward_steps])
                self.size_buf[self.ptr] = size
                self.relative_pos_buf[self.ptr] = np.copy(data.relative_pos[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.comm_mask_buf[self.ptr] = np.copy(data.comm_mask[i*self.chunk_capacity:(i+1)*self.chunk_capacity+config.burn_in_steps+config.forward_steps])
                self.gamma_buf[start_idx:start_idx+size] = data.gammas[i*self.chunk_capacity:i*self.chunk_capacity+size]
                self.num_agents_buf[self.ptr] = data.num_agents

                self.ptr = (self.ptr+1) % self.num_chunks

            del data


    def _sample_batch(self, batch_size: int) -> Tuple:

        b_obs, b_last_act, b_steps, b_relative_pos, b_comm_mask = [], [], [], [], []
        b_hidden = []
        idxes, priorities = [], []

        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // self.chunk_capacity
            local_idxes = idxes % self.chunk_capacity
            max_num_agents = np.max(self.num_agents_buf[global_idxes])

            for global_idx, local_idx in zip(global_idxes.tolist(), local_idxes.tolist()):

                assert local_idx < self.size_buf[global_idx], \
                    'index is {} but size is {}, p {}'.format(local_idx, self.size_buf[global_idx], self.priority_tree[local_idx])

                steps = min(config.forward_steps, self.size_buf[global_idx].item()-local_idx)

                relative_pos = self.relative_pos_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                comm_mask = self.comm_mask_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                obs = self.obs_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                last_act = self.last_act_buf[global_idx][local_idx:local_idx+config.burn_in_steps+steps+1]
                hidden = self.hid_buf[global_idx][local_idx]

                if steps < config.forward_steps:
                    pad_len = config.forward_steps - steps
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0)))
                    last_act = np.pad(last_act, ((0, pad_len), (0, 0), (0, 0)))
                    relative_pos = np.pad(relative_pos, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
                    comm_mask = np.pad(comm_mask, ((0, pad_len), (0, 0), (0, 0)))

                if self.num_agents_buf[global_idx] < max_num_agents:
                    pad_len = max_num_agents - self.num_agents_buf[global_idx].item()
                    obs = np.pad(obs, ((0, 0), (0, pad_len), (0, 0), (0, 0), (0, 0)))
                    last_act = np.pad(last_act, ((0, 0), (0, pad_len), (0, 0)))
                    relative_pos = np.pad(relative_pos, ((0, 0), (0, pad_len), (0, pad_len), (0, 0)))
                    comm_mask = np.pad(comm_mask, ((0, 0), (0, pad_len), (0, pad_len)))
                    hidden = np.pad(hidden, ((0, pad_len), (0, 0)))

                b_obs.append(obs)
                b_last_act.append(last_act)
                b_steps.append(steps)
                b_relative_pos.append(relative_pos)
                b_comm_mask.append(comm_mask)
                b_hidden.append(hidden)

            # importance sampling weight
            min_p = np.min(priorities)
            # print(f'min_p: {min_p}')
            weights = np.power(priorities/min_p, -self.beta)
            # print(f'weights: {weights}')

            b_action = self.act_buf[idxes]
            b_reward = self.rew_buf[idxes]
            b_gamma = self.gamma_buf[idxes]

            data = (
                torch.from_numpy(np.stack(b_obs)).transpose(1,0).contiguous(),
                torch.from_numpy(np.stack(b_last_act)).transpose(1,0).contiguous(),
                torch.from_numpy(b_action).unsqueeze(1),
                torch.from_numpy(b_reward).unsqueeze(1),
                torch.from_numpy(b_gamma).unsqueeze(1),
                torch.ByteTensor(b_steps),

                torch.from_numpy(np.concatenate(b_hidden, axis=0)),
                torch.from_numpy(np.stack(b_relative_pos)),
                torch.from_numpy(np.stack(b_comm_mask)),

                idxes,
                torch.from_numpy(weights.astype(np.float16)).unsqueeze(1),
                self.ptr
            )

            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been discarded in replay buffer during training
            if self.ptr > old_ptr:
                # range from [old_ptr, self.ptr)
                mask = (idxes < old_ptr*self.chunk_capacity) | (idxes >= self.ptr*self.chunk_capacity)
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                # range from [0, self.ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr*self.chunk_capacity) & (idxes >= self.ptr*self.chunk_capacity)
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(np.copy(idxes), np.copy(priorities)**self.alpha)

    def stats(self, interval: int):
        '''
        Print log
        '''
        print('buffer update speed: {}/s'.format(self.counter/interval))
        print('buffer size: {}'.format(np.sum(self.size_buf)))

        print('  ', end='')
        for i in range(config.init_env_settings[1], config.max_map_lenght+1, 5):
            print('   {:2d}   '.format(i), end='')
        print()

        for num_agents in range(config.init_env_settings[0], config.max_num_agents+1):
            # if first element does not exist, skip
            if not (num_agents, config.init_env_settings[1]) in self.stat_dict:
                break

            # num agent
            print('{:2d}'.format(num_agents), end='')

            # stat dict
            for map_len in range(config.init_env_settings[1], config.max_map_lenght+1, 5):
                if (num_agents, map_len) in self.stat_dict:
                    print('{:4d}/{:<3d}'.format(sum(self.stat_dict[(num_agents, map_len)]), len(self.stat_dict[(num_agents, map_len)])), end='')
                else:
                    print('   N/A  ', end='')
            print()

        for key, val in self.stat_dict.copy().items():
            # print('{}: {}/{}'.format(key, sum(val), len(val)))
            if len(val) == config.cl_history_size and sum(val) >= config.cl_history_size*config.pass_rate:
                # add number of agents
                add_agent_key = (key[0]+1, key[1])
                if add_agent_key[0] <= config.max_num_agents and add_agent_key not in self.stat_dict:
                    self.stat_dict[add_agent_key] = []

                if key[1] < config.max_map_lenght:
                    add_map_key = (key[0], key[1]+5)
                    if add_map_key not in self.stat_dict:
                        self.stat_dict[add_map_key] = []

        self.env_settings_set = ray.put(list(self.stat_dict.keys()))

        self.counter = 0

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False

    def get_env_settings(self):
        return self.env_settings_set


# @ray.remote
# class ModelSaver:
#     def __init__(self):
#         self.state_dict = []

#     def save_model(self, model, steps):
#         print('saving model!!')

#         # self.state_dict.append(model.state_dict())

#         # # create dir
#         # path = os.path.join(os.getcwd(), f'{config.save_path}')
#         # print(f'cwd: {os.getcwd()}')
#         # print(f'path to save: {path}')

#         # if not os.path.exists(path):
#         #     os.mkdir(path)
#         #     print(f'directory {path} created')

#         # torch.save(model.state_dict(), os.path.join(config.save_path, f'{steps}.pth'))

#         # print('model saved at step {}'.format(steps))

@ray.remote(num_gpus=4, num_cpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer): #, model_saver: ModelSaver):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Network()
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=2e-4)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[40000, 80000], gamma=0.5)
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0


        # add previous steps num


        self.data_list = []

        self.store_weights()

        # self.model_saver = model_saver
        self.temp_state_dict = None

        self.state_dict_step = 0
        self.state_dict = 'fake original state dict'

        # self.learning_thread = thread

        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self._train, daemon=True)#, args=(self.state_dict, self.state_dict_step))
        self.learning_thread.start()

    def _train(self):#, state_dict, state_dict_step):
        print('training')
        scaler = GradScaler()
        b_seq_len = torch.LongTensor(config.batch_size)
        b_seq_len[:] = config.burn_in_steps+1

        for i in range(1, config.training_steps+1):

            # get a batch of data
            data_id = ray.get(self.buffer.get_batched_data.remote())
            data = ray.get(data_id)

            b_obs, b_last_act, b_action, b_reward, b_gamma, b_steps, b_hidden, b_relative_pos, b_comm_mask, idxes, weights, old_ptr = data
            b_obs, b_last_act, b_action, b_reward = b_obs.to(self.device), b_last_act.to(self.device), b_action.to(self.device), b_reward.to(self.device)
            b_gamma, weights = b_gamma.to(self.device), weights.to(self.device)
            b_hidden = b_hidden.to(self.device)
            b_relative_pos, b_comm_mask = b_relative_pos.to(self.device), b_comm_mask.to(self.device)

            b_action = b_action.long()

            b_obs, b_last_act = b_obs.half(), b_last_act.half()

            b_next_seq_len = b_seq_len + b_steps

            with torch.no_grad():
                b_q_ = self.tar_model(b_obs, b_last_act, b_next_seq_len, b_hidden, b_relative_pos, b_comm_mask).max(1, keepdim=True)[0]

            target_q = b_reward + b_gamma * b_q_

            # default forward_steps = 2
            # b_obs[:-2] => last 2 b_obs
            b_q = self.model(b_obs[:-config.forward_steps], b_last_act[:-config.forward_steps], b_seq_len, b_hidden, b_relative_pos[:, :-config.forward_steps], b_comm_mask[:, :-config.forward_steps]).gather(1, b_action)

            td_error = target_q - b_q

            priorities = td_error.detach().clone().squeeze().abs().clamp(1e-6).cpu().numpy()

            # apply MSE loss
            loss = F.mse_loss(b_q, target_q)
            self.loss += loss.item() # loss.item() => convert tensor to normal scalar and

            self.optimizer.zero_grad() # zero out the gradients before backprop because pytorch accumulates them
            scaler.scale(loss).backward()

            # TODO: what are these?
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm_dqn)

            scaler.step(self.optimizer)
            scaler.update()

            self.scheduler.step()

            # store new weights in shared memory
            if i % 2  == 0:
                self.store_weights()

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

            self.counter += 1

            # update target network, save model
            if i % config.target_network_update_freq == 0:
                self.tar_model.load_state_dict(self.model.state_dict())

            if i % config.save_interval == 0:

                # create dir
                path = os.path.join(os.getcwd(), f'{config.save_path}')
                # print(f'cwd: {os.getcwd()0, path to save: {path}}')

                if not os.path.exists(path):
                    os.mkdir(path)

                torch.save(self.model.state_dict(), os.path.join(f'{path}', f'{self.counter}.pth'))
                torch.save({
                    'training_steps': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss,
                }, os.path.join(f'{path}', f'{self.counter}.pt'))
                print('model saved at step {}'.format(i))

        self.done = True

    def stats(self, interval: int):
        '''
        print log
        '''
        print('number of updates: {}'.format(self.counter))
        print('update speed: {}/s'.format((self.counter-self.last_counter)/interval))
        if self.counter != self.last_counter:
            print('loss: {:.4f}'.format(self.loss/(self.counter-self.last_counter)))

        self.last_counter = self.counter
        self.loss = 0
        return self.done

    def get_attr(self, attr):
        return getattr(self, attr)


# @ray.remote(num_cpus=1)
# def save_model(model, steps):
#     # create dir
#     path = os.path.join(os.getcwd(), f'{config.save_path}')
#     print(f'cwd: {os.getcwd()}')
#     print(f'path to save: {path}')

#     if not os.path.exists(path):
#         os.mkdir(path)
#         print(f'directory {path} created')

#     torch.save(model.state_dict(), os.path.join(config.save_path, f'{steps}.pth'))

#     print('model saved at step {}'.format(steps))

@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer):
        self.id = worker_id
        self.model = Network()
        self.model.eval()
        self.env = Environment(curriculum=True)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = config.max_episode_length
        self.counter = 0

    def run(self):
        done = False
        obs, last_act, pos, local_buffer = self._reset() # came from env.observe()

        while True:

            # sample action
            actions, q_val, hidden, relative_pos, comm_mask = self.model.step(torch.from_numpy(obs.astype(np.float32)),
                                                                            torch.from_numpy(last_act.astype(np.float32)),
                                                                            torch.from_numpy(pos.astype(np.int)))

            if random.random() < self.epsilon:
                # Note: only one agent do random action in order to keep the environment stable
                actions[0] = np.random.randint(0, config.action_dim)

            # take action in env
            (next_obs, last_act, next_pos), rewards, done, _ = self.env.step(actions)
            # return data and update observation
            local_buffer.add(q_val[0], actions[0], last_act, rewards[0], next_obs, hidden, relative_pos, comm_mask)

            if done == False and self.env.steps < self.max_episode_length:
                obs, pos = next_obs, next_pos
            else:
                # finish and send buffer
                if done:
                    data = local_buffer.finish()
                else:
                    _, q_val, _, relative_pos, comm_mask = self.model.step(torch.from_numpy(next_obs.astype(np.float32)),
                                                                            torch.from_numpy(last_act.astype(np.float32)),
                                                                            torch.from_numpy(next_pos.astype(np.int)))
                    data = local_buffer.finish(q_val[0], relative_pos, comm_mask)

                self.global_buffer.add.remote(data)
                done = False
                obs, last_act, pos, local_buffer = self._reset()

            self.counter += 1
            if self.counter == config.actor_update_steps:
                self._update_weights()
                self.counter = 0

    def _update_weights(self):
        '''load weights from learner'''
        # update network parameters
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
        # update environment settings set (number of agents and map size)
        new_env_settings_set = ray.get(self.global_buffer.get_env_settings.remote())
        self.env.update_env_settings_set(ray.get(new_env_settings_set))

    def _reset(self):
        self.model.reset()
        obs, last_act, pos = self.env.reset()
        local_buffer = LocalBuffer(self.id, self.env.num_agents, self.env.map_size[0], obs)
        return obs, last_act, pos, local_buffer
