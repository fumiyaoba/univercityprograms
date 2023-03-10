import numpy as np
import pymc3 as pm
import theano.tensor as tt
from scipy import stats

#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse # 楕円を描く
import seaborn as sns

colors = sns.color_palette().as_hex()
n_colors = len(colors)

np.random.seed(100)

N = 1000
mus_sv = np.array([1, 5, 8, ])
taus_sv = np.array([1., 1., 1.5,])
pi = np.array([0.3, 0.5, 0.2])
K_true = len(mus_sv)

ss = np.random.choice(K_true, size=1000, replace=True, p=pi)
x = np.random.randn(N) * 1.0/np.sqrt(taus_sv[ss]) + mus_sv[ss]

fig = plt.figure(figsize=(8, 3))
ax = fig.subplots(1,1)
sns.distplot(x, bins=50, ax=ax)
plt.savefig('/home/oba/programs/bayes/dirichlet_test_result/dirichlet_test_4,2.png')

def SBP(vi):
    """Stick Breaking Process
    混合比の事前分布
    """
    stick_len = tt.concatenate([[1], tt.extra_ops.cumprod(1 - vi)[:-1]])
    pis = vi * stick_len
    return pis

K = 32
alpha = 2

with pm.Model() as model:
    vi = pm.Beta('vi', 1., alpha, shape=K)   
    pi = pm.Deterministic('pi', SBP(vi))

    
    comp_dist = []
    for k in range(K):
        mu = pm.Normal('mu%i'%k, mu=0, sd=10)
        tau = pm.Gamma('tau%i'%k, alpha=1., beta=0.1)
        comp = pm.Normal.dist(mu=mu, tau=tau)
        #comp_dist.append(pm.Normal.dist(mu=mu, tau=tau))
        #print(type(comp))
        comp_dist.append(comp)
    #print(len(comp_dist))
    obs = pm.Mixture('obs', pi, comp_dist , observed=x)
"""print(len(comp_dist))
print(len(x))"""



with model:
    trace = pm.sample(1000, chains=1)

pi_post = trace['pi']
pi_mean = pi_post.mean(axis=0)
fig = plt.figure(figsize=(7, 4))
ax = fig.subplots(1, 1)


ax.bar(np.arange(K)[:20], trace['pi'].mean(axis=0)[:20])
plt.savefig('/home/oba/programs/bayes/dirichlet_test_result/dirichlet_test_4,3.png')

fig = plt.figure(figsize=(7, 7))
ax = fig.subplots(2, 1)


pi_thresh = 0.05
for k in range(K):
    if pi_mean[k] < pi_thresh:
        continue
    ax[0].plot(trace[f'mu{k}'], label=f'c{k}')
    ax[1].plot(trace[f'tau{k}'], label=f'c{k}')
ax[0].legend()
ax[1].legend()
plt.savefig('/home/oba/programs/bayes/dirichlet_test_result/dirichlet_test_4,4.png')

x_plot = np.linspace(-10, 10, 500)

components = np.array([stats.norm.pdf(x_plot, trace[f'mu{k}'].mean(), 1./np.sqrt(trace[f'tau{k}'].mean())) for k in range(K)])
pi_mean = trace['pi'].mean(axis=0)
post_components = (pi_mean[:,np.newaxis] * components)
post_pdf = post_components.sum(axis=0)
fig = plt.figure(figsize=(7, 4))
ax = fig.subplots(1, 1)


ax.hist(x, bins=50, alpha=0.5, density=True, label='data')
ax.plot(x_plot, post_pdf, label='post dist')
for i in range(5):
    ax.plot(x_plot, post_components[i,:], '--', label=f'comp{i}')
ax.legend()
plt.savefig('/home/oba/programs/bayes/dirichlet_test_result/dirichlet_test_4,5.png')

def component_assignment(x, trace, pi_thresh=0.05):
    def likelihood(x, pi, mu, tau):
        return pi * stats.norm.pdf(x, mu, 1./np.sqrt(tau))
    liks = []
    pi_post = trace['pi']
    pi_mean = pi_post.mean(axis=0)
    nk = pi_post.shape[1]
    for k in range(nk):
        if pi_mean[k] < pi_thresh:
            # 混合比が小さいコンポーネントは無視する
            continue
        mu_dist = trace[f'mu{k}']
        tau_dist = trace[f'tau{k}']
        lik_k = likelihood(x, pi_post[:,k], mu_dist, tau_dist)
        liks.append(lik_k)
    liks = np.array(liks)
    marginal_lik = liks.sum(axis=0)
    p_k = (liks / marginal_lik).T
    return p_k.mean(axis=0)
fig = plt.figure(figsize=(7, 4))
ax = fig.subplots(1, 1)


# ベースのデータのヒストグラム
vals, bins, patches = ax.hist(x, bins=50, alpha=0.5, density=True)

# ヒストグラムのビン毎の所属確率を算出
lst_p_k = list(map(
    lambda x:component_assignment(x, trace, pi_thresh=0.01), 
    bins[1:]))
for i, p in enumerate(lst_p_k):
    comp_num = np.argmax(p)
    patches[i].set_facecolor(colors[comp_num%n_colors])

ax.plot(x_plot, post_pdf, c='#000000', label='post dist')
plt.savefig('/home/oba/programs/bayes/dirichlet_test_result/dirichlet_test_4,6.png')
for i in range(5):
    ax.plot(x_plot, post_components[i,:], '--', c=colors[i], label=f'comp{i}')
    ax.legend()
plt.savefig('/home/oba/programs/bayes/dirichlet_test_result/dirichlet_test_4,7.png')