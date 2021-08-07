from gym.envs.registration import register
import rlbench.backend.task as task
import os
from rlbench.utils import name_to_task_class
from wrapper.robot_envs import RLBenchEnv

TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py')]

for task_file in TASKS:
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    register(
        id='%s-state-v1' % task_name,
        entry_point='wrapper.robot_envs:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state'
        }
    )
    register(
        id='%s-vision-v1' % task_name,
        entry_point='wrapper.robot_envs:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision'
        }
    )
