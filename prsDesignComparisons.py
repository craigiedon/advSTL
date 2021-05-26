import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from probRobScene import scenario_from_file
from probRobScene.core.vectors import rotate_euler
from scipy.spatial import ConvexHull

from prsIntegration import unif_design_from_scenario
from volumeApprox import central_composite_discrepancy


def compare_cube_simple():
    grid_num = 5
    random_ccds = []
    random_times = []

    uniform_ccds = []
    uniform_times = []

    ns = [10, 25, 50, 100]
    for n in ns:
        scenario = scenario_from_file("scenarios/cubeTableSimple.prs")

        cube_reg = scenario.objects[1]._conditioned._dependencies[0].region

        inv_start = time.perf_counter()
        inv_rosen_scenes = unif_design_from_scenario(scenario, n)
        uniform_times.append(time.perf_counter() - inv_start)

        inv_rosen_design = np.array([s.objects[1].position for s in inv_rosen_scenes])
        flattened_rosen = rotate_euler(inv_rosen_design - cube_reg.origin, cube_reg.rev_rot)

        rand_start = time.perf_counter()
        random_sample_scenes = [scenario.generate()[0] for i in range(n)]
        random_times.append(time.perf_counter() - rand_start)

        random_design = np.array([s.objects[1].position for s in random_sample_scenes])
        flattened_random = rotate_euler(random_design - cube_reg.origin, cube_reg.rev_rot)

        cube_hull = ConvexHull(cube_reg.to_hsi().intersections)

        random_ccd = central_composite_discrepancy(cube_hull, flattened_random[:, :2], grid_num)
        uniform_ccd = central_composite_discrepancy(cube_hull, flattened_rosen[:, :2], grid_num)

        print(f"N = {n}, random-ccd: {random_ccd}, uniform-ccd: {uniform_ccd}")

        random_ccds.append(random_ccd)
        uniform_ccds.append(uniform_ccd)

    comparison_plots(ns, [random_ccds, uniform_ccds], ["Random", "Inv-Ros"], ["blue", "orange"], ("N", "CCD"))
    comparison_plots(ns, [random_times, uniform_times], ["Random", "Inv-Ros"], ["blue", "orange"], ("N", "Time (Seconds)"))


def comparison_plots(xs: List[int], comparison_ys: List[List[float]], labels: List[str], colors: List[str], ax_names: Tuple[str, str]) -> None:
    fig, ax = plt.subplots()

    fig.set_size_inches(3, 3)

    for label, c, ys in zip(labels, colors, comparison_ys):
        ax.plot(xs, ys, linestyle='-', color=c, linewidth=1, zorder=1, label=label)
        ax.scatter(xs, ys, color='white', s=100, zorder=2)
        ax.scatter(xs, ys, color=c, s=20, zorder=3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_bounds(min(xs), max(xs))

    ax.spines['left'].set_bounds(np.min(comparison_ys), np.max(comparison_ys))

    ax.set_xlabel(ax_names[0])
    ax.set_ylabel(ax_names[1])

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_cube_simple()

# Extract random "design"
# print(random_design.shape)
# print("Flat: ", flattened_random)
# print(cube_hull)
# print("Random CCD: ", random_ccd)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # print("Num objects:", len(self.objects))
#
# w_min_corner, w_max_corner = scenes[0].workspace.getAABB()
# w_dims = w_max_corner - w_min_corner
#
# draw_cube(ax, (w_max_corner + w_min_corner) * 0.5, w_dims, np.zeros(3), color='purple', alpha=0.03)
#
# total_min, total_max = np.min(w_min_corner), np.max(w_max_corner)
#
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
#
# plt.tight_layout()
#
# scenes[0].objects[0].show_3d(ax)
# for scene in scenes:
#     for obj in scene.objects[1:]:
#         obj.show_3d(ax)
# sample_scenes[0].objects[0].show_3d(ax)
# for scene in sample_scenes:
#     for obj in scene.objects[1:]:
#         obj.show_3d(ax)

# plt.show()

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
