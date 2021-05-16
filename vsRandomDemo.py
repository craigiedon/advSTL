from invRosenblattExp import good_lattice_point
import matplotlib.pyplot as plt
import numpy as np

for N in [10, 25, 50, 101]:
    random_points = np.random.random((N, N))
    square_points = good_lattice_point(2, N)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"N = {N}", fontsize=16)

    ax1.scatter(random_points[:, 0], random_points[:, 1])
    ax1.set_aspect("equal")
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title("Random")

    ax2.set_title("Uniform Design")
    ax2.set_aspect("equal")
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.scatter(square_points[:, 0], square_points[:, 1])
    plt.show()
