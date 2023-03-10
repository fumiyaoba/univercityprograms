import numpy as np
import pymc3 as pm
import theano.tensor as tt
from scipy import stats

#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse # 楕円を描く
import seaborn as sns
with pm.Model() as model:
    lam = pm.Exponential('lam', lam=1, shape=(2,))  # `shape=(2,)` indicates two mixture components.

    # As we just need the logp, rather than add a RV to the model, we need to call .dist()
    components = pm.Poisson.dist(mu=lam, shape=(2,))
    print(type(components))
    w = pm.Dirichlet('w', a=np.array([1, 1]))  # two mixture component weights.

    #like = pm.Mixture('like', w=w, comp_dists=components, observed=data)