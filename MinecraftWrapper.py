from gym.core import Wrapper
import torch
from data_manager import ActionManager2

class MinecraftWrapper(Wrapper):
    def __init__(self, env, action_manager):

        self.action_manager = action_manager
        self.action_manager2=ActionManager2(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.done = False

        self.last_obs = None  # used for logging

        self.craft= ["none", "torch", "stick", "planks", "crafting_table"]
        self.equip= ["none", "air", "wooden_axe", "wooden_pickaxe", "stone_axe", "stone_pickaxe", "iron_axe", "iron_pickaxe"]
        self.nearbyCraft= ["none", "wooden_axe", "wooden_pickaxe", "stone_axe", "stone_pickaxe", "iron_axe", "iron_pickaxe", "furnace"]
        self.nearbySmelt= ["none", "iron_ingot", "coal"]
        self.place= ["none", "dirt", "stone", "cobblestone", "crafting_table", "furnace", "torch"]

        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        self.done = False
        return torch.from_numpy(obs['pov'].copy())

    def step(self, action, crafting):
        assert not self.done
        #if(crafting):print("PRE",action)
        
        if crafting:
            a=action
            action = self.action_manager2.get_action(action)
            if a==0:
                action['equip']=3
        else:action = self.action_manager.get_action(action)
        action['craft']=self.craft[action['craft']]
        action['equip']=self.equip[action['equip']]
        action['nearbyCraft']=self.nearbyCraft[action['nearbyCraft']]
        action['nearbySmelt']=self.nearbySmelt[action['nearbySmelt']]
        action['place']=self.place[action['place']]

        

        obs, r, self.done, info = super().step(action)
        """        if action['craft']!='none': print("after craft",obs['inventory'])
        if action['nearbyCraft']!='none': print("after nearbycraft",obs['inventory'])"""

        
        self.last_obs = obs

        return obs, r, self.done