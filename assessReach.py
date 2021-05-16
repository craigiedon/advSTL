import probRobScene
import pyrep.objects
from probRobScene.wrappers.coppelia.setupFuncs import top_of
from pyrep import PyRep
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Camera
import numpy as np
from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath
from scipy.optimize import minimize

from stl import *
import sys
from probRobScene.wrappers.coppelia import robotControl as rc
from probRobScene.wrappers.coppelia.prbCoppeliaWrapper import cop_from_prs
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper


def get_above_object_path(agent: Panda, target_obj: pyrep.objects.Object, z_offset: float = 0.0,
                          ig_cols: bool = False) -> ArmConfigurationPath:
    pos = top_of(target_obj)
    pos[2] += z_offset

    path = agent.get_path(position=pos, euler=[-np.pi, 0.0, np.pi / 2.0], ignore_collisions=ig_cols)  # , euler=orient)
    # path.visualize()
    return path


scenario = probRobScene.scenario_from_file("scenarios/cubeOnTable.prs")
pr = PyRep()
pr.launch("scenes/emptyVortex.ttt", headless=False, responsive_ui=True)

scene_view = Camera('DefaultCamera')
scene_view.set_position([-0.43, 3.4, 2.25])
scene_view.set_orientation(np.array([114, 0.0, 0.0]) * np.pi / 180.0)

ex_world, used_its = scenario.generate()
c_objs = cop_from_prs(pr, ex_world)

cube = c_objs["CUBOID"][0]
initial_cube_pos = np.array(cube.get_position())
# print(cube)
panda_1, gripper_1 = Panda(0), PandaGripper(0)

initial_arm_config = panda_1.get_configuration_tree()
initial_arm_joint_pos = panda_1.get_joint_positions()
initial_gripper_config = gripper_1.get_configuration_tree()
initial_gripper_joint_pos = gripper_1.get_joint_positions()


def reset_arm():
    pr.set_configuration_tree(initial_arm_config)
    pr.set_configuration_tree(initial_gripper_config)
    panda_1.set_joint_positions(initial_arm_joint_pos, disable_dynamics=True)
    panda_1.set_joint_target_velocities([0] * 7)
    gripper_1.set_joint_positions(initial_gripper_joint_pos, disable_dynamics=True)
    gripper_1.set_joint_target_velocities([0] * 2)


pr.start()
pr.step()

ts = pr.get_simulation_timestep()
print("timestep:", ts)


def sim_fun(cube_x_guess: np.ndarray, obj_spec: List[STLExp]) -> float:
    reset_arm()
    new_cube_pos = np.array(initial_cube_pos) + np.array([cube_x_guess[0], 0.0, 0.0])
    cube.set_position(new_cube_pos)

    max_timesteps = 100
    state_information = []

    try:
        arm_path = get_above_object_path(panda_1, cube, 0.03)
        move_done = False
    except ConfigurationPathError as e:
        print(e)
        move_done = True

    for t in range(max_timesteps):
        # print(t)
        if not move_done:
            move_done = arm_path.step()

        pr.step()

        target_pos = np.array(top_of(cube)) + np.array([0.0, 0.0, 0.03])
        arm_pos = panda_1.get_tip().get_position()
        state_information.append((target_pos, arm_pos))

    for i in range(2000):
        pr.step()

    score = agm_rob(obj_spec[0], state_information, 0)
    print("Cube offset: ", cube_x_guess, "score:", score)
    return -score


dist_func = lambda state: np.linalg.norm(state[0] - state[1]) / 10.0
spec = F(LEQ0(dist_func), 0, 99)
# # minimization loop
pr.start()
result = minimize(sim_fun, np.array([-0.2]), [spec], method='Powell', bounds=[(-1.0, 1.0)], options={'disp': True})
# print(result)

# Objective: Get near that cube!

# metrics = [dist_func(x) for x in state_information]
# print("Distances:", metrics)
# print("Robustness spec: ", stl_rob(spec, state_information, 0))
#
input("Simulation Finished. To Quit, Press Enter")
pr.stop()
pr.shutdown()
