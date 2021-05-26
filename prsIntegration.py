# Load in a PRS scenario file with "cube on table"
from typing import Any, List

import numpy as np
from multimethod import multimethod
import matplotlib.pyplot as plt

from probRobScene import scenario_from_string, scenario_from_file
from probRobScene.core.distributions import RejectionException, sample_all, DefaultIdentityDict, sample
from probRobScene.core.plotUtil3d import draw_cube
from probRobScene.core.regions import Region, ConvexPolyhedronRegion, feasible_point, PointInRegionDistribution, \
    Rectangle3DRegion
from probRobScene.core.scenarios import Scenario, Scene

def try_sample_copy(external_sampler, dependencies, objects, workspace, active_reqs):
    try:
        if external_sampler is not None:
            external_sampler.sample(external_sampler.rejectionFeedback)
        sample = sample_all(dependencies)
    except RejectionException as e:
        return None, e

    obj_samples = [sample[o] for o in objects]
    collidable = [o for o in obj_samples if not o.allowCollisions]

    for o in obj_samples:
        if not workspace.contains_object(o):
            return None, 'object containment'

    if any(cuboids_intersect(vi, vj) for (i, vi) in enumerate(collidable) for vj in collidable[:i]):
        return None, 'object intersection'

    for (i, req) in enumerate(active_reqs):
        if not req(sample):
            return None, f'user-specified requirement {i}'

    return sample, None

from probRobScene.core.vectors import rotate_euler, Vector3D
from scipy.spatial import HalfspaceIntersection, ConvexHull

from invRosenblattExp import good_lattice_point
from volumeApprox import inv_rosen, inv_rosen_given_dependencies, inv_cdf_convex_2d_single, inv_cdf_convex_3d_single, central_composite_discrepancy


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

@multimethod
def inv_trans(r: ConvexPolyhedronRegion, unit_point: np.ndarray) -> np.ndarray:
    return inv_cdf_convex_3d_single(unit_point, r.hsi.halfspaces, [0, 1, 2])

@multimethod
def inv_trans(r: Rectangle3DRegion, unit_point: np.ndarray) -> np.ndarray:
    flat_point = inv_cdf_convex_2d_single(unit_point, r.to_hsi().halfspaces, [0, 1, 2])
    flat_3d = np.concatenate((flat_point, [0.0]))

    transformed_point = rotate_euler(flat_3d, r.rot) + r.origin
    return transformed_point

@multimethod
def inv_trans(r: Region, unit_point: np.ndarray) -> np.ndarray:
    raise NotImplementedError(f"Not yet implemented inverse transform for region of type {type(r)}")


def unif_design_from_scenario(scenario: Scenario, n: int) -> List[Scene]:
    # Step 0: Dependency graph of properties that depend on one another?
    # TODO: Actually do proper dependency ordering here
    dependency_ordering = []
    for o in scenario.objects:
        for dep in o._dependencies:
            dependency_ordering.append(dep)

    # Step 0.1: Calculate Hypercube dimensionality
    #TODO: Make this not hard-wired
    hypercube_dims = len(dependency_ordering) * 2

    # Step 1: Create hypercube of this dimensionality
    hypercube_design = good_lattice_point(hypercube_dims, n)

    # Step 2: Do top-level transform
    top_level_design = unif_design(dependency_ordering[0], n)

    """
    # Step 3: Do dependent level transform by looping and feeding it in.

    dependent_regions = []
    for i in range(n):
        design_instances = DefaultIdentityDict({dependency_ordering[0]: Vector3D(*top_level_design[i])})
        dependent_regions.append(sample(dependency_ordering[1].region, design_instances))

    dependent_level_transform = [inv_trans(dependent_regions[i], hypercube_design[i, 0+1:]) for i in range(n)]

    # Step 4: Convert into fully instantiated objects with filled in parts...
    design_map = {
        dependency_ordering[0]: top_level_design,
        dependency_ordering[1]: dependent_level_transform
    }
    """
    design_map = {
        dependency_ordering[0]: top_level_design
    }

    scenes = []
    for i in range(n):
        design_instances = DefaultIdentityDict({k: Vector3D(*v[i]) for k, v in design_map.items()})
        instantiated_objects = [o._conditioned.sample_given_dependencies(design_instances) for o in scenario.objects]
        scenes.append(Scene(scenario.workspace, instantiated_objects, scenario.params))

    # print(scenes)

    return scenes


def run():
    n = 17
    scenario = scenario_from_file("scenarios/cubeTableSimple.prs")
    scenes = unif_design_from_scenario(scenario, n)

    sample_scenes = [scenario.generate()[0] for i in range(n)]

    # Extract random "design"
    random_design = np.array([s.objects[1].position for s in sample_scenes])
    print(random_design.shape)
    cube_reg = scenario.objects[1]._conditioned._dependencies[0].region
    flattened_random = rotate_euler(random_design - cube_reg.origin, cube_reg.rev_rot)
    print("Flat: ", flattened_random)
    cube_hull = ConvexHull(cube_reg.to_hsi().intersections)
    print(cube_hull)
    random_ccd = central_composite_discrepancy(cube_hull, flattened_random[:, :2], 5)
    print("Random CCD: ", random_ccd)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # print("Num objects:", len(self.objects))

    w_min_corner, w_max_corner = scenes[0].workspace.getAABB()
    w_dims = w_max_corner - w_min_corner

    draw_cube(ax, (w_max_corner + w_min_corner) * 0.5, w_dims, np.zeros(3), color='purple', alpha=0.03)

    total_min, total_max = np.min(w_min_corner), np.max(w_max_corner)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()

    scenes[0].objects[0].show_3d(ax)
    for scene in scenes:
        for obj in scene.objects[1:]:
            obj.show_3d(ax)
    # sample_scenes[0].objects[0].show_3d(ax)
    # for scene in sample_scenes:
    #     for obj in scene.objects[1:]:
    #         obj.show_3d(ax)

    plt.show()

    # OK! Now plot the scenes!!!!

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
