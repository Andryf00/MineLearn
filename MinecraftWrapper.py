from gym.core import Wrapper
import torch

class MinecraftWrapper(Wrapper):
    def __init__(self, env, action_manager):

        self.action_manager = action_manager

        self.done = False

        self.last_obs = None  # used for logging

        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        self.done = False
        return torch.from_numpy(obs['pov'].copy())

    def step(self, action):
        assert not self.done

        action = self.action_manager.get_action(action)

        if 'craft' in action:
            if \
                    action['attack'] == 0 and \
                    action['craft'] == 0 and \
                    action['nearbyCraft'] == 0 and \
                    action['nearbySmelt'] == 0 and \
                    action['place'] == 0:
                action['jump'] = 1
        else:
            if action['attack'] == 0:
                action['jump'] = 1

        obs, r, self.done, info = super().step(action)

        self.last_obs = obs

        return torch.from_numpy(obs['pov'].copy()), r, self.done