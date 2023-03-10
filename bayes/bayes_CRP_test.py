import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline

sns.set()

np.random.seed(100)

def prob_table_choice(lst_n, alpha=1.0):
    """テーブルの選択確率を算出
    """
    n = sum(lst_n) + 1
    p = [ni / (n - 1 + alpha) for ni in lst_n]
    p.append( alpha / (n - 1 + alpha) )
    return p

def CRP(n, alpha=1.0):
    n_c = []
    history = {
        'table_nums':[], 
        'chosen_tables':[], 
    }
    for i in range(n):
        table_ids = np.arange(0, len(n_c)+1)
        p = prob_table_choice(n_c, alpha=alpha)
        chosen_table = np.random.choice(table_ids, p=p)
        if len(n_c) == chosen_table:
            n_c.append(1)
        else:
            n_c[chosen_table] += 1
        history['table_nums'].append(len(n_c))
        history['chosen_tables'].append(chosen_table)
    return n_c, history
N = 1000
customer_counts1, history1 = CRP(N, alpha=10)
customer_counts2, history2 = CRP(N, alpha=2)

fig = plt.figure(figsize=(7, 4))
ax = fig.subplots(1, 1)
ax.plot(history1['table_nums'], label=f'$\\alpha=10$')
ax.plot(history2['table_nums'], label=f'$\\alpha=2$')
ax.legend()
ax.set_xlabel('number of customer')
ax.set_ylabel('number of table')
plt.savefig('bayes_CRP_test_1.png')
#Text(0, 0.5, 'number of table')

fig = plt.figure(figsize=(12, 4))
ax = fig.subplots(1, 2, sharex=True, sharey=True)
ax[0].bar(np.arange(1, len(customer_counts1)+1), sorted(customer_counts1, reverse=True), label=f'$\\alpha=10$')
ax[1].bar(np.arange(1, len(customer_counts2)+1), sorted(customer_counts2, reverse=True), label=f'$\\alpha=2$')
ax[0].legend()
ax[1].legend()
plt.savefig('bayes_CRP_test_2.png')