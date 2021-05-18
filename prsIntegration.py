# Load in a PRS scenario file with "cube on table"
from typing import Any, List

import numpy as np
from multimethod import multimethod

from probRobScene import scenario_from_string, scenario_from_file
from probRobScene.core.distributions import RejectionException, sample_all
from probRobScene.core.regions import Region, ConvexPolyhedronRegion, feasible_point, PointInRegionDistribution, \
    Rectangle3DRegion
from probRobScene.core.scenarios import Scenario, Scene

# def try_sample(external_sampler, dependencies, objects, workspace, active_reqs):
#     try:
#         if external_sampler is not None:
#             external_sampler.sample(external_sampler.rejectionFeedback)
#         sample = sample_all(dependencies)
#     except RejectionException as e:
#         return None, e
#
#     obj_samples = [sample[o] for o in objects]
#     collidable = [o for o in obj_samples if not o.allowCollisions]
#
#     for o in obj_samples:
#         if not workspace.contains_object(o):
#             return None, 'object containment'
#
#     if any(cuboids_intersect(vi, vj) for (i, vi) in enumerate(collidable) for vj in collidable[:i]):
#         return None, 'object intersection'
#
#     for (i, req) in enumerate(active_reqs):
#         if not req(sample):
#             return None, f'user-specified requirement {i}'
#
#     return sample, None
from probRobScene.core.vectors import rotate_euler
from scipy.spatial import HalfspaceIntersection

from invRosenblattExp import good_lattice_point
from volumeApprox import inv_rosen


@multimethod
def unif_design(r: Region, n: int) -> np.ndarray:
    raise NotImplementedError(f"Not yet implemented uniform designs for region type: {type(r)}")


@multimethod
def unif_design(r: ConvexPolyhedronRegion, n: int) -> np.ndarray:
    cube_design = good_lattice_point(3, n)
    region_design = inv_rosen(cube_design, r.hsi.halfspaces)
    return region_design

@multimethod
def unif_design(r: Rectangle3DRegion, n: int) -> np.ndarray:
    square_design = good_lattice_point(2, n)
    flat_design = inv_rosen(square_design, r.to_hsi().halfspaces)
    flat_3d = np.hstack((flat_design, np.zeros((n, 1))))

    exp_design = rotate_euler(flat_3d, r.rot) + r.origin
    return exp_design



@multimethod
def unif_design(d: PointInRegionDistribution, n: int) -> np.ndarray:
    return unif_design(d.region, n)


def unif_design_from_scenario(scenario: Scenario, n: int) -> List[Scene]:
    design_map = {}
    # So i think, up here what you should do is loop the dependencies, and turn each into a list of instantiations, mapped to the original object
    # Then bubble this recursively back up to the top?
    for o in scenario.objects:
        design_map[o] = {}
        # print(f"Object: {type(o)}")
        # print("Dependencies:")
        for dep in o._dependencies:
            # print("\t", dep)
            exp_design = unif_design(dep, n)
            # print("\t", exp_design.shape)
            design_map[o][dep] = exp_design

    for i in range(n):

    print(design_map)
    # Scene(workspace, objects, params)

    # Ok, how about this: Just get *one* instantiation of the scene from your design thingy...
    return design_map


def run():
    scenario = scenario_from_file("scenarios/cubeOnTableSimple.prs")
    unif_design_from_scenario(scenario, 10)
    # hsi = np.array([
    #     [-1.0, 0.0, 0.0, 0.0],
    #     [1.0, 0.0, 0.0, -1.0],
    #     [0.0, -1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0, -1.0],
    #     [0.0, 0.0, 1.0, -1.0],
    #     [0.0, 0.0, -1.0, 0.0],
    # ])
    #
    # cpr = ConvexPolyhedronRegion(HalfspaceIntersection(hsi, feasible_point(hsi)))
    # design = unif_design(cpr, 10)
    # print(design)


if __name__ == "__main__":
    run()

# Definte a function thats like "Make uniform design for each of hte object properties"
# Initially just return a map of each of the object properties with a distribution? Follow the "sample" function for inspiration
# Now actually implement the meat of it using the invRosenblatt functions + any new ones you might need
# Find a creative way to vizualize this stuff + evaluate?
