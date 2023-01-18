import minerl
from collections import deque
from copy import deepcopy
from operator import itemgetter
from dataset import Transition

from collections import OrderedDict


def put_data_into_dataset(action_manager, dataset, minecraft_human_data_dir,
                          continuous_action_stacking_amount=3,
                          only_successful=True, max_duration_steps=None, max_reward=256.,
                          test=False,env_name="MineRLObtainDiamond-v0"):
    """
    :param env_name: Minecraft env name
    :param action_manager: expects object of data_manager.ActionManager
    :param dataset: expects object of dataset.Dataset
    :param minecraft_human_data_dir: location of Minecraft human data
    :param continuous_action_stacking_amount: number of consecutive states that are used to get the continuous action
    (since humans move the camera slowly we add up the continuous actions of multiple consecutive states)
    :param only_successful: skip trajectories that don't reach final reward when true
    :param max_duration_steps: skip trajectories that take longer than max_duration_steps to reach the final reward
    :param max_reward: remove trajectory part beyond the max_reward. Used to remove the "obtain diamond" part, since
    the imitation policy never obtains diamonds anyway
    :param test: if true a mini dataset is created for debugging
    further all samples without rewards, and without terminal states, and with no_op action are removed
    """

    print(f"\n Adding data from {env_name} \n")


    def is_success(sample):
        return sample[-1]['success']

    def is_no_op(sample):
        action = sample[1]
        a_id = action_manager.get_id(action)
        assert type(a_id) == int
        return a_id == 0  # no_op action has id of 0

    def process_sample_inv(sample, last_reward):
        """adding single sample to dataset if all conditions are met, expects sample with already stacked
        camera action"""
        
        reward = sample[2]
        
        dataset.append_sample_inv(sample)
        counter_change = 1

        return counter_change, reward

    def process_sample(sample, last_reward):
        """adding single sample to dataset if all conditions are met, expects sample with already stacked
        camera action"""
        
        reward = sample[2]
        
        
        if reward != 0.:
            dataset.append_sample(sample)
            #dataset.update_last_reward_index()
            counter_change = 1
        else:
            if not is_no_op(sample) or sample[4]:  # remove no_op transitions, unless it is a terminal state
                dataset.append_sample(sample)
                counter_change = 1
            else:
                counter_change = 0

        return counter_change, reward

    data = minerl.data.make(env_name, data_dir=minecraft_human_data_dir)
    #data=data.filter(lambda meta: meta['succes']==True) might be useful, from 
    trajs = data.get_trajectory_names()

    # the ring buffer is used to stack the camera action of multiple consecutive states:
    sample_que = deque(maxlen=continuous_action_stacking_amount)

    total_trajs_counter = 0
    added_sample_counter = 0

    #initial_sample_amount = dataset.transitions.current_size()
    print(dataset.index,dataset.capacity)
    finished=False
    last_sample=0
    for n, traj in enumerate(trajs):
        if finished: break
        start=False
        finish=False
        print("AGUGA",dataset.index,dataset.capacity)
        for j, sample in enumerate(data.load_data(traj, include_metadata=True)):
            """print(sample[0]['inventory'])
            print(sample[0]['inventory']['wooden_pickaxe'])"""
            
            if(dataset.index==dataset.capacity-1-continuous_action_stacking_amount):
                finished=True
                print("YAAAAAAAAA")
                break
            # at first we check if the trajectory will be used :
            if j == 0:
                last_sample=sample
                print(sample[-1])

                if not is_success(sample):
                    print("skipping trajectory")
                    break

                total_trajs_counter += 1
                last_reward = 0.

            sample_que.append(sample)

            # Only continue when we have enough states to stack the camera actions:
            if len(sample_que) == continuous_action_stacking_amount:

                # Stacking camera action for the oldest sample in the queue:
                for i in range(1, continuous_action_stacking_amount):
                    sample_que[0][1]['camera'] += sample_que[i][1]['camera']

                    if sample_que[i][2] != 0.:  # (if reward != 0)
                        break  # no camera action stacking after a reward
                #if(sample_que[0][0]['inventory']['crafting_table','furnace','iron_pickaxe','planks','stick', 'stone_pickaxe','torch','wooden_pickaxe']!=last_sample[0]['inventory'][]):
                """TO SELECT ONLY CRAFTING WOOD DATA FROM OBTAIN D
                if(
                    sample_que[0][1]['craft']=="stick" or sample_que[0][1]['craft']== "planks" or sample_que[0][1]['craft']== "crafting_table"
                    or sample_que[0][1]['nearbyCraft']=='wooden_pickaxe'
                    or sample_que[0][1]['place']=='crafting_table'):
                    added_samples, last_reward = process_sample_inv(sample_que[0], last_reward)
                    added_sample_counter += added_samples"""
                """TOO SELECT ONLY DATA WHEN WOODEN PICKAXE IS EQUIPPED
                if sample_que[0][0]["inventory"]['wooden_pickaxe']!=0 and not start: start=True
                if sample_que[0][0]["inventory"]['furnace']!=0 and start: finish=True
                if(start and not finish):
                    added_samples, last_reward = process_sample(sample_que[0], last_reward)
                    added_sample_counter += added_samples 
                """
                if(
                    sample_que[0][1]['craft']=="stick" or sample_que[0][1]['craft']== "crafting_table"
                    or sample_que[0][1]['nearbyCraft']=='stone_pickaxe'or sample_que[0][1]['nearbyCraft']=='furnace'
                    or sample_que[0][1]['place']=='crafting_table'):
                    added_samples, last_reward = process_sample_inv(sample_que[0], last_reward)
                    added_sample_counter += added_samples
            """TO SELECT ONLY TREE CHOP DATA FROM OBTAIN DIAMOND
            if(last_reward>=2):
                print("stopping traj")
                break"""
        

            #not need for vecs
            """if len(sample_que) > 0:  # otherwise not successful traj
            # for the last samples in the queue we don't stack the the camera actions
            for i in range(1, continuous_action_stacking_amount):
                added_samples, last_reward = process_sample(sample_que[i], last_reward)
                added_sample_counter += added_samples
            """
            # a terminal state could be reached without exceeding max_reward:
            #added_sample_counter -= dataset.remove_new_data()

            # making sure the last state from trajectory is terminal:
            """            last_transition = deepcopy(dataset.transitions[dataset.index - 1])
            dataset.transitions[dataset.index - 1] = \
                Transition(last_transition.state,
                           last_transition.action, last_transition.reward, False)"""

        sample_que.clear()

        print(f"{n+1} / {len(trajs)}, added: {total_trajs_counter}")
        #assert dataset.transitions.current_size() - initial_sample_amount == added_sample_counter

        if test:
            if total_trajs_counter >= 2:
                assert total_trajs_counter == 2
                break