from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import config


class CommLayer(nn.Module):
    def __init__(self, input_dim=config.hidden_dim, message_dim=32, pos_embed_dim=16, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.message_dim = message_dim
        self.pos_embed_dim = pos_embed_dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(input_dim)

        self.position_embeddings = nn.Linear((2*config.obs_radius+1)**2, pos_embed_dim)

        self.message_key = nn.Linear(input_dim+pos_embed_dim, message_dim * num_heads)
        self.message_value = nn.Linear(input_dim+pos_embed_dim, message_dim * num_heads)
        self.hidden_query = nn.Linear(input_dim, message_dim * num_heads)

        self.head_agg = nn.Linear(message_dim * num_heads, message_dim * num_heads)

        self.update = nn.GRUCell(num_heads*message_dim, input_dim)

    def position_embed(self, relative_pos, dtype, device):

        batch_size, num_agents, _, _ = relative_pos.size()
        # mask out out of FOV agent
        relative_pos[(relative_pos.abs() > config.obs_radius).any(3)] = 0

        fov_length = 2*config.obs_radius+1
        one_hot_position = torch.zeros((batch_size*num_agents*num_agents, fov_length*fov_length), dtype=dtype, device=device)

        relative_pos += config.obs_radius
        relative_pos = relative_pos.reshape(batch_size*num_agents*num_agents, 2)

        # basically turning matrix into 1D array
        relative_pos_idx = relative_pos[:, 0] + relative_pos[:, 1]*fov_length
        # print(f'relative pos idx: {relative_pos_idx}')

        # buggy
        # when obs_radius is 1, relative_pos_idx is out of range
        # IndexError: index 94 is out of bounds for dimension 1 with size 81
        one_hot_position[torch.arange(batch_size*num_agents*num_agents), relative_pos_idx.long()] = 1
        position_embedding = self.position_embeddings(one_hot_position)

        return position_embedding

    def forward(self, hidden, relative_pos, comm_mask):
        batch_size, num_agents, hidden_dim = hidden.size()
        attn_mask = (comm_mask==False).unsqueeze(3).unsqueeze(4)
        relative_pos = relative_pos.clone()

        position_embedding = self.position_embed(relative_pos, hidden.dtype, hidden.device)

        input = hidden

        hidden = self.norm(hidden)

        hidden_q = self.hidden_query(hidden).view(batch_size, 1, num_agents, self.num_heads, self.message_dim) # batch_size x num_agents x message_dim*num_heads

        message_input = hidden.repeat_interleave(num_agents, dim=1).view(batch_size*num_agents*num_agents, hidden_dim)
        message_input = torch.cat((message_input, position_embedding), dim=1)
        message_input = message_input.view(batch_size, num_agents, num_agents, self.input_dim+self.pos_embed_dim)
        message_k = self.message_key(message_input).view(batch_size, num_agents, num_agents, self.num_heads, self.message_dim)
        message_v = self.message_value(message_input).view(batch_size, num_agents, num_agents, self.num_heads, self.message_dim)

        # attention
        attn_score = (hidden_q * message_k).sum(4, keepdim=True) / self.message_dim**0.5 # batch_size x num_agents x num_agents x self.num_heads x 1
        attn_score.masked_fill_(attn_mask, torch.finfo(attn_score.dtype).min)
        attn_weights = F.softmax(attn_score, dim=1)

        # agg
        agg_message = (message_v * attn_weights).sum(1).view(batch_size, num_agents, self.num_heads*self.message_dim)
        agg_message = self.head_agg(agg_message)

        # update hidden with request message
        input = input.view(-1, hidden_dim)
        agg_message = agg_message.view(batch_size*num_agents, self.num_heads*self.message_dim)
        updated_hidden = self.update(agg_message, input)

        # some agents may not receive message, keep it as original
        update_mask = comm_mask.any(1).view(-1, 1)
        hidden = torch.where(update_mask, updated_hidden, input)
        hidden = hidden.view(batch_size, num_agents, hidden_dim)

        return hidden



class CommBlock(nn.Module):
    def __init__(self, hidden_dim=config.hidden_dim, message_dim=128, pos_embed_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.pos_embed_dim = pos_embed_dim

        self.request_comm = CommLayer()
        self.reply_comm = CommLayer()


    def forward(self, latent, relative_pos, comm_mask):
        '''
        latent shape: batch_size x num_agents x latent_dim
        relative_pos shape: batch_size x num_agents x num_agents x 2
        comm_mask shape: batch_size x num_agents x num_agents
        '''

        batch_size, num_agents, latent_dim = latent.size()

        assert relative_pos.size() == (batch_size, num_agents, num_agents, 2), relative_pos.size()
        assert comm_mask.size() == (batch_size, num_agents, num_agents), comm_mask.size()

        if torch.sum(comm_mask).item() == 0:
            return latent

        hidden = self.request_comm(latent, relative_pos, comm_mask)

        comm_mask = torch.transpose(comm_mask, 1, 2)

        hidden = self.reply_comm(hidden, relative_pos, comm_mask)

        return hidden

class Network(nn.Module):
    def __init__(self, input_shape=config.obs_shape, selective_comm=config.selective_comm):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.latent_dim = self.hidden_dim + 5
        self.obs_shape = input_shape
        self.selective_comm = selective_comm

        if config.obs_radius == 1:
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(input_shape[0], 128, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 128, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Flatten(),
            )
        elif config.obs_radius == 2:
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(input_shape[0], 128, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 256, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Flatten(),
            )
        elif config.obs_radius == 3:
            print(f'input_shape[0] when obs_radius=3: {input_shape[0]}')

            self.obs_encoder = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Flatten(),
            )
        else:
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 192, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(192, 256, 3, 1),
                nn.LeakyReLU(0.2, True),
                nn.Flatten(),
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim).to(device)
        self.comm = CommBlock(self.hidden_dim)

        self.hidden = None

        # dueling q structure
        self.adv = nn.Linear(self.hidden_dim, 5)
        self.state = nn.Linear(self.hidden_dim, 1)

    @torch.no_grad()
    def step(self, obs, last_act, pos):
        """
        return actions, q_val.numpy(), self.hidden.squeeze(0).numpy(), relative_pos.numpy(), comm_mask.numpy()
        """
        num_agents = obs.size(0)
        agent_indexing = torch.arange(num_agents)
        relative_pos = pos.unsqueeze(0)-pos.unsqueeze(1)

        in_obs_mask = (relative_pos.abs() <= config.obs_radius).all(2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_obs_mask = in_obs_mask.to(device)

        # print(f'relativepos: {relative_pos.abs()}')
        # print(f'(relative_pos.abs() <= config.obs_radius) type: {type((relative_pos.abs() <= config.obs_radius))}')
        # print(f'in_obs_mask: {in_obs_mask},  device: {in_obs_mask.get_device()}')

        in_obs_mask[agent_indexing, agent_indexing] = 0

        if self.selective_comm:
            test_mask = in_obs_mask.clone()
            test_mask[agent_indexing, agent_indexing] = 1
            num_in_obs_agents = test_mask.sum(1)
            origin_agent_idx = torch.zeros(num_agents, dtype=torch.long)
            for i in range(num_agents-1):
                origin_agent_idx[i+1] = test_mask[i, i:].sum() + test_mask[i+1, :i+1].sum() + origin_agent_idx[i]
            test_obs = torch.repeat_interleave(obs, num_agents, dim=0).view(num_agents, num_agents, *config.obs_shape)[test_mask]

            test_relative_pos = relative_pos[test_mask]
            test_relative_pos += config.obs_radius

            test_obs[torch.arange(num_in_obs_agents.sum()), 0, test_relative_pos[:, 0], test_relative_pos[:, 1]] = 0

            # check the dim of test_* and self.recurrent input dim
            test_last_act = torch.repeat_interleave(last_act, num_in_obs_agents, dim=0)
            if self.hidden is None:
                test_hidden = torch.zeros((num_in_obs_agents.sum(), self.hidden_dim))
            else:
                test_hidden = torch.repeat_interleave(self.hidden, num_in_obs_agents, dim=0)

            test_latent = self.obs_encoder(test_obs)
            # print(f'test_latent: {test_latent.size()}')
            # exit()

            test_latent = torch.cat((test_latent, test_last_act), dim=1)

            test_latent = test_latent.to(device)
            test_hidden = test_hidden.to(device)

            test_hidden = self.recurrent(test_latent, test_hidden)
            self.hidden = test_hidden[origin_agent_idx]

            adv_val = self.adv(test_hidden)
            state_val = self.state(test_hidden)
            test_q_val = (state_val + adv_val - adv_val.mean(1, keepdim=True))
            test_actions = torch.argmax(test_q_val, 1)
            test_actions = test_actions.to(device)

            actions_mat = torch.ones((num_agents, num_agents), dtype=test_actions.dtype) * -1
            actions_mat = actions_mat.to(device)

            actions_mat[test_mask] = test_actions
            diff_action_mask = actions_mat != actions_mat[agent_indexing, agent_indexing].unsqueeze(1)

            assert (in_obs_mask[agent_indexing, agent_indexing] == 0).all()
            diff_action_mask = diff_action_mask.to(device)
            comm_mask = torch.bitwise_and(in_obs_mask, diff_action_mask)

        else:

            latent = self.obs_encoder(obs)
            latent = torch.cat((latent, last_act), dim=1)

            # mask out agents that are far away
            dist_mat = (relative_pos[:, :, 0]**2 + relative_pos[:, :, 1]**2)
            _, ranking = dist_mat.topk(min(config.max_comm_agents, num_agents), dim=1, largest=False)
            dist_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool)
            dist_mask.scatter_(1, ranking, True)
            comm_mask = torch.bitwise_and(in_obs_mask, dist_mask)
            # comm_mask[torch.arange(num_agents), torch.arange(num_agents)] = 0

            if self.hidden is None:
                self.hidden = self.recurrent(latent)
            else:
                self.hidden = self.recurrent(latent, self.hidden)

        assert (comm_mask[agent_indexing, agent_indexing] == 0).all()

        self.hidden = self.comm(self.hidden.unsqueeze(0), relative_pos.unsqueeze(0), comm_mask.unsqueeze(0))
        self.hidden = self.hidden.squeeze(0)

        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)

        q_val = (state_val + adv_val - adv_val.mean(1, keepdim=True))

        actions = torch.argmax(q_val, 1).tolist()

        return actions, q_val.cpu().numpy(), self.hidden.cpu().squeeze(0).numpy(), relative_pos.cpu().numpy(), comm_mask.cpu().numpy()

    def reset(self):
        self.hidden = None

    @autocast() # allow mixed precision
    def forward(self, obs, last_act, steps, hidden, relative_pos, comm_mask):
        '''
        used for training

        in worker.py ~ line 321 or 325:
        321: b_q_ = self.tar_model(b_obs, b_last_act, b_next_seq_len, b_hidden, b_relative_pos, b_comm_mask).max(1, keepdim=True)[0]
        325: b_q = self.model(b_obs[:-config.forward_steps], b_last_act[:-config.forward_steps], b_seq_len, b_hidden, b_relative_pos[:, :-config.forward_steps], b_comm_mask[:, :-config.forward_steps]).gather(1, b_action)
        '''
        # obs shape: seq_len, batch_size, num_agents, obs_shape
        # relative_pos shape: batch_size, seq_len, num_agents, num_agents, 2

        # print(f'obs, last_act, steps, hidden, relative_pos, comm_mask shape: {obs.shape, last_act.shape, steps.shape, hidden.shape, relative_pos.shape, comm_mask.shape}')
        # exit()

        seq_len, batch_size, num_agents, *_ = obs.size()

        obs = obs.view(seq_len*batch_size*num_agents, *self.obs_shape)
        last_act = last_act.view(seq_len*batch_size*num_agents, config.action_dim)

        latent = self.obs_encoder(obs)
        latent = torch.cat((latent, last_act), dim=1)
        latent = latent.view(seq_len, batch_size*num_agents, self.latent_dim)

        hidden_buffer = []
        for i in range(seq_len):
            # hidden size: batch_size*num_agents x self.hidden_dim
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(batch_size, num_agents, self.hidden_dim)
            hidden = self.comm(hidden, relative_pos[:, i], comm_mask[:, i])
            # only hidden from agent 0
            hidden_buffer.append(hidden[:, 0])
            hidden = hidden.view(batch_size*num_agents, self.hidden_dim)

        # hidden buffer size: batch_size x seq_len x self.hidden_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)

        # hidden size: batch_size x self.hidden_dim
        dummyx = torch.arange(config.batch_size)
        dummyy = (steps-1).to(torch.long)
        print(f'x type: {torch.arange(config.batch_size).type()}')
        print(f'y type: {(dummyy).type()}')
        hidden = hidden_buffer[dummyx, dummyy]
        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val
