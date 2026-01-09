import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
import pyemma
import numpy as np
import matplotlib.cm as cm

def plot_fes(proj, title, max_points=10000):
    # Subsample if needed
    if proj.shape[0] > max_points:
        idx = np.random.choice(proj.shape[0], max_points, replace=False)
        proj = proj[idx]

    x, y = proj[:,0], proj[:,1]
    z = gaussian_kde(np.vstack([x,y]))(np.vstack([x,y]))
    plt.scatter(x, y, c=z, s=1, cmap='viridis')
    plt.title(f'Free Energy: {title}')
    plt.xlabel('TICA 1'); plt.ylabel('TICA 2')
    plt.colorbar(label='Density')
    plt.tight_layout()
    plt.savefig(f'/Users/marl/Code/SpaceTime/images/fes_{title.replace(" ", "_").lower()}.png', dpi=300)
    plt.close()

def compute_its(proj, name):
    cluster = pyemma.coordinates.cluster_kmeans(proj, k=10, max_iter=100)
    its = pyemma.msm.its(cluster.dtrajs, lags=[1,2,5,10,20,50], nits=3)
    return its

def kl_2d(p, q, bins=50):
    Hp, _, _ = np.histogram2d(p[:,0], p[:,1], bins=bins, density=True)
    Hq, _, _ = np.histogram2d(q[:,0], q[:,1], bins=bins, density=True)
    Hp += 1e-12; Hq += 1e-12
    return entropy(Hp.flatten(), Hq.flatten())

def plot_its_custom(its_obj, label, cmap='tab10', max_modes=5):
    lags = its_obj.lagtimes
    timescales = its_obj.timescales  # shape: (n_lags, n_modes)

    colormap = cm.get_cmap(cmap, max_modes)

    for i in range(min(max_modes, timescales.shape[1])):
        color = colormap(i)
        plt.plot(lags, timescales[:, i], label=f'{label} τ{i+1}', color=color)

    plt.xscale("linear")
    plt.yscale("log")
    plt.xlabel("Lag time / steps")
    plt.ylabel("Timescale / steps")
    plt.legend()