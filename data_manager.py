import torch
import numpy as np
from collections import OrderedDict
from itertools import product
import copy


class StateManager:
    """Main minecraft state wrapper, creates image and vector out of the state information. Some inventory items
    (of feasible average amount) are encoded as multi-hot vectors, the rest as normalized float values."""

    def __init__(self, device):
        self.device = device
    def get_img(self, state):
        img = state['pov']
        return img

    def get_torch_img(self, img_list):
        img_torch = torch.tensor(img_list, dtype=torch.float32, device=self.device).div_(255).permute(0, 3, 1, 2)
        return img_torch


class ActionManager:
    """Main minecraft action wrapper. Simplifies action space to 130 discrete actions"""

    def __init__(self, device, c_action_magnitude=22.5):
        self.device = device
        self.c_action_magnitude = c_action_magnitude

        self.zero_action = OrderedDict([('attack', 0),
                                        ('back', 0),
                                        ('camera', np.array([0., 0.])),
                                        ('craft', 0),
                                        ('equip', 0),
                                        ('forward', 0),
                                        ('jump', 0),
                                        ('left', 0),
                                        ('nearbyCraft', 0),
                                        ('nearbySmelt', 0),
                                        ('place', 0),
                                        ('right', 0),
                                        ('sneak', 0),
                                        ('sprint', 0)])

        # ['sneak'] is ignored
        # camera discretization:
        self.camera_dict = OrderedDict([
            ('turn_up', np.array([-c_action_magnitude, 0.])),
            ('turn_down', np.array([c_action_magnitude, 0.])),
            ('turn_left', np.array([0., -c_action_magnitude])),
            ('turn_right', np.array([0., c_action_magnitude]))
        ])

        self.fully_connected_no_camera = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sprint']
        self.camera_actions = ['turn_up', 'turn_down', 'turn_left', 'turn_right']
        self.fully_connected = self.fully_connected_no_camera + self.camera_actions

        # following action combinations are excluded:
        self.exclude = [('forward', 'back'), ('left', 'right'), ('attack', 'jump'),
                        ('turn_up', 'turn_down', 'turn_left', 'turn_right')]

        # sprint only allowed when forward is used:
        self.only_if = [('sprint', 'forward')]

        # Maximal allowed mount of actions within one action:
        self.remove_size = 3

        # if more than 3 actions are present, actions are removed using this list until only 3 actions remain:
        self.remove_first_list = ['sprint', 'left', 'right', 'back',
                                  'turn_up', 'turn_down', 'turn_left', 'turn_right',
                                  'attack', 'jump', 'forward']

        self.fully_connected_list = list(product(range(2), repeat=len(self.fully_connected)))

        remove = []
        for el in self.fully_connected_list:
            for tuple_ in self.exclude:
                if sum([el[self.fully_connected.index(a)] for a in tuple_]) > 1:
                    if el not in remove:
                        remove.append(el)
            for a, b in self.only_if:
                if el[self.fully_connected.index(a)] == 1 and el[self.fully_connected.index(b)] == 0:
                    if el not in remove:
                        remove.append(el)
            if sum(el) > self.remove_size:
                if el not in remove:
                    remove.append(el)

        for r in remove:
            self.fully_connected_list.remove(r)

        self.action_list = []
        for el in self.fully_connected_list:
            new_action = copy.deepcopy(self.zero_action)
            for key, value in zip(self.fully_connected, el):
                if key in self.camera_actions:
                    if value:
                        new_action['camera'] = self.camera_dict[key]
                else:
                    new_action[key] = value
            self.action_list.append(new_action)
        """"I don't think this is needed, might be about inventory, not 100%sure
        self.separate_id_dict = OrderedDict()
        for i, key in enumerate(self.separate):
            self.separate_id_dict[key] = OrderedDict()
            for id_ in self.separate_values[i]:
                new_action = copy.deepcopy(self.zero_action)
                new_action[key] = id_
                self.separate_id_dict[key][id_] = len(self.action_list)
                self.action_list.append(new_action)
"""
        self.num_action_ids_list = [len(self.action_list)]
        self.act_continuous_size = 0

    def get_action(self, id_):
        a = copy.deepcopy(self.action_list[int(id_)])
        a['camera'] += np.random.normal(0., 0.5, 2)
        return a

    def print_action(self, id_):
        a = copy.deepcopy(self.action_list[int(id_)])
        out = ""
        for k, v in a.items():
            if k != 'camera':
                if v != 0:
                    if k in self.separate_str_lists:
                        out += f'{k} {self.separate_str_lists[k][v]} '
                    else:
                        out += f'{k} '
            else:
                if (v != np.zeros(2)).any():
                    out += k

        print(out)

    def get_id(self, action):
        """Also inv related but not super sure
        for key in self.separate:
            if action[key] != 0:
                if action[key] in self.separate_id_dict[key]:
                    action_id = self.separate_id_dict[key][action[key]]
                    return action_id
        """
        action = copy.deepcopy(action)

        # discretize 'camera':
        camera = action['camera']
        camera_action_amount = 0
        if - self.c_action_magnitude / 2. < camera[0] < self.c_action_magnitude / 2.:
            action['camera'][0] = 0.
            if - self.c_action_magnitude / 2. < camera[1] < self.c_action_magnitude / 2.:
                action['camera'][1] = 0.
            else:
                camera_action_amount = 1
                action['camera'][1] = self.c_action_magnitude * np.sign(camera[1])
        else:
            camera_action_amount = 1
            action['camera'][0] = self.c_action_magnitude * np.sign(camera[0])

            action['camera'][1] = 0.

        # simplify action:
        for tuple_ in self.exclude:
            if len(tuple_) == 2:
                a, b = tuple_
                if action[a] and action[b]:
                    action[b] = 0
        for a, b in self.only_if:
            if not action[b]:
                if action[a]:
                    action[a] = 0
        for a in self.remove_first_list:
            if sum([action[key] for key in self.fully_connected_no_camera]) > \
                    (self.remove_size - camera_action_amount):
                if a in self.camera_actions:
                    action['camera'] = np.array([0., 0.])
                    camera_action_amount = 0
                else:
                    action[a] = 0
            else:
                break

        # set one_hot camera keys:
        for key in self.camera_actions:
            action[key] = 0
        for key, val in self.camera_dict.items():
            if (action['camera'] == val).all():
                action[key] = 1
                break

        non_separate_values = tuple(action[key] for key in self.fully_connected)

        return self.fully_connected_list.index(non_separate_values)

    def get_torch_action(self, a_id_batch_list):
        a_id_torch_list = [torch.tensor(a_id_batch, dtype=torch.int64, device=self.device) for a_id_batch in
                           a_id_batch_list]

        return a_id_torch_list