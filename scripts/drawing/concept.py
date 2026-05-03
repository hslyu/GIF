import matplotlib as mpl
import numpy as np

mpl.use("Agg")
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SURFACE_COEFFS = (0.5, 0.25)
POINT = (1, -0.75)
X_RANGE = (-0.75, 4)
Y_RANGE = (-4, 0.75)
Z_RANGE = (-13, 5)
COLOR_MIN = -6.5
COLORMAP = "viridis"
# COLORMAP = "magma"
# COLORMAP = "inferno"
COLORMAP_MAX = 0.82


def concave_surface(x, y, coeffs):
    a, b = coeffs
    return -(a * x**2 + b * y**2)


def concave_gradient(x, y, coeffs):
    a, b = coeffs
    return np.array([-2 * a * x, -2 * b * y])


def concave_hessian(coeffs):
    a, b = coeffs
    return np.array([[-2 * a, 0.0], [0.0, -2 * b]])


def tangent_z(x, y, point, coeffs):
    x0, y0 = point
    z0 = concave_surface(x0, y0, coeffs)
    fx0, fy0 = concave_gradient(x0, y0, coeffs)
    return z0 + fx0 * (x - x0) + fy0 * (y - y0)


def unit_vector(vector, length=2.4):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return length * vector / norm


def truncated_colormap(name, min_value=0.0, max_value=COLORMAP_MAX, n=256):
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(min_value, max_value, n))
    return LinearSegmentedColormap.from_list(f"{name}_truncated", colors)


def plot_concave_surface(ax, coeffs, x_range, y_range, cmap):
    x = np.linspace(*x_range, 30)
    y = np.linspace(*y_range, 30)
    X, Y = np.meshgrid(x, y)
    Z = concave_surface(X, Y, coeffs)
    norm = Normalize(vmin=COLOR_MIN, vmax=Z.max(), clip=True)

    return ax.plot_surface(
        X,
        Y,
        Z,
        cmap=truncated_colormap(cmap),
        norm=norm,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        alpha=0.8,
        zorder=1,
    )


def plot_tangent_plane(ax, point, coeffs, half_width=1.8):
    x0, y0 = point
    z0 = concave_surface(x0, y0, coeffs)
    plane_corners_xy = [
        (x0 - half_width, y0 - half_width),
        (x0 + half_width, y0 - half_width),
        (x0 + half_width, y0 + half_width),
        (x0 - half_width, y0 + half_width),
    ]
    plane_corners = [
        (px, py, tangent_z(px, py, point, coeffs)) for px, py in plane_corners_xy
    ]

    ax.add_collection3d(
        Poly3DCollection(
            [plane_corners],
            facecolors="white",
            edgecolors="none",
            linewidths=0,
            alpha=0.75,
            zorder=5,
        )
    )
    ax.scatter([x0], [y0], [z0], color="black", s=15, depthshade=False, zorder=10)


def plot_direction_arrow(ax, point, coeffs, direction_xy, color, zorder):
    x0, y0 = point
    z0 = concave_surface(x0, y0, coeffs)
    fx0, fy0 = concave_gradient(x0, y0, coeffs)
    direction_xy = unit_vector(np.asarray(direction_xy, dtype=float))
    direction_z = fx0 * direction_xy[0] + fy0 * direction_xy[1]

    x1 = x0 + direction_xy[0]
    y1 = y0 + direction_xy[1]
    z1 = z0 + direction_z

    ax.plot(
        [x0, x1],
        [y0, y1],
        [z0, z1],
        color=color,
        linewidth=2.0,
        zorder=zorder,
    )
    return x1, y1, z1


def plot_gradient_descent_arrow(ax, point, coeffs):
    direction_xy = -concave_gradient(*point, coeffs)
    plot_direction_arrow(ax, point, coeffs, direction_xy, color="red", zorder=15)


def plot_negative_newton_arrow(ax, point, coeffs):
    gradient = concave_gradient(*point, coeffs)
    hessian = concave_hessian(coeffs)
    direction_xy = np.linalg.solve(hessian, gradient)
    x1, y1, z1 = plot_direction_arrow(
        ax, point, coeffs, direction_xy, color="blue", zorder=16
    )
    ax.text(
        x1 + 0.1,
        y1,
        z1 + 0.2,
        r"$\mathbf{H}^{-1}\mathbf{g}$",
        color="blue",
        fontsize=16,
        zorder=20,
    )


def plot_x_only_newton_arrow(ax, point, coeffs):
    gradient = concave_gradient(*point, coeffs)
    hessian = concave_hessian(coeffs)
    direction_xy = np.array([gradient[0] / hessian[0, 0], 0.0])
    plot_direction_arrow(ax, point, coeffs, direction_xy, color="limegreen", zorder=18)


def style_axes(ax):
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_zlim(*Z_RANGE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-45)


def render(cmap, output_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    surf = plot_concave_surface(ax, SURFACE_COEFFS, X_RANGE, Y_RANGE, cmap)
    # plot_tangent_plane(ax, POINT, SURFACE_COEFFS)
    plot_gradient_descent_arrow(ax, POINT, SURFACE_COEFFS)
    plot_negative_newton_arrow(ax, POINT, SURFACE_COEFFS)
    plot_x_only_newton_arrow(ax, POINT, SURFACE_COEFFS)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    render(COLORMAP, "concept.pdf")


if __name__ == "__main__":
    main()
