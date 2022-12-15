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
            action = self.action_manager2.get_action(action)
        else:action = self.action_manager.get_action(action)
        #if(crafting):print("PRE",action)
        """if action['craft']!=0: print(" CRAFTING:", self.craft[action['craft']])
        if action['nearbyCraft']!=0: print(" NEARBYCRAFTING:", self.nearbyCraft[action['nearbyCraft']])
        if action['place']!=0: print(" PLACING:", self.place[action['place']])"""
        action['craft']=self.craft[action['craft']]
        action['equip']=self.equip[action['equip']]
        action['nearbyCraft']=self.nearbyCraft[action['nearbyCraft']]
        action['nearbySmelt']=self.nearbySmelt[action['nearbySmelt']]
        action['place']=self.place[action['place']]

        #if(crafting):print(action)
        

        
        """if action['attack'] == 0:
            action['jump'] = 1"""

        obs, r, self.done, info = super().step(action)
        """        if action['craft']!='none': print("after craft",obs['inventory'])
        if action['nearbyCraft']!='none': print("after nearbycraft",obs['inventory'])"""

        
        self.last_obs = obs

        return obs, r, self.done