#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:38:16 2022


"""

"""
Plotting script for the dataset of Bernoulli observations used in Mitra et al.'s paper.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# df = pd.read_csv('flsea.csv')
# delta_vals = [np.exp(-1*i) for i in np.arange(1.0, 10.0, 1)]  # error probability
# delta_vals = [int(np.log(1/delta)) for delta in delta_vals]

df = pd.read_csv('total-cost-varying-C-bernoulli.csv')
delta_vals = [np.exp(-1*i) for i in np.arange(1.0, 6.0, 2.0)]  # error probability
delta_vals = [int(np.log(1/delta)) for delta in delta_vals]

a = df['total-cost-C=0-FedElim0'].to_numpy(dtype = int) 
b = df['total-cost-C=0-FedElim'].to_numpy(dtype = int) 
c = df['total-cost-C=10-FedElim'].to_numpy(dtype = int)
d = df['total-cost-C=100-FedElim'].to_numpy(dtype = int)
err1 = df['error-bar-total-cost-C=0-FedElim0'].to_numpy(dtype = float)
err2 = df['error-bar-total-cost-C=0-FedElim'].to_numpy(dtype = float)
err3 = df['error-bar-total-cost-C=10-FedElim'].to_numpy(dtype = float)
err4 = df['error-bar-total-cost-C=100-FedElim'].to_numpy(dtype = float)


width = 0.2

fig, ax = plt.subplots()
x = np.arange(0,len(a))
rects1 = ax.bar(x - 1.5*width, a, width, yerr = err1,
                  align = 'center', 
                  alpha = 0.5,
                  ecolor = 'purple',
                  error_kw = dict(lw=10, capsize=15, capthick=10),
                  label=r'$C=0$, FedElim0')
rects2 = ax.bar(x - 0.5*width, b, width, yerr = err2,
                  align = 'center', 
                  alpha = 0.5,
                  ecolor = 'purple',
                  error_kw = dict(lw=10, capsize=15, capthick=10),
                  label=r'$C=0$, FedElim')
rects3 = ax.bar(x + 0.5*width, c, width, yerr = err3,
                  align = 'center', 
                  alpha = 0.5,
                  ecolor = 'purple',
                  error_kw = dict(lw=10, capsize=15, capthick=10),
                  label=r'$C=10$, FedElim')
rects4 = ax.bar(x + 1.5*width, d, width, yerr = err4,
                  align = 'center', 
                  alpha = 0.5,
                  ecolor = 'purple',
                  error_kw = dict(lw=10, capsize=12, capthick=10),
                  label=r'$C=100$, FedElim')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis='both', which='major', labelsize=100)
ax.set_xlabel(r'$ \ln \frac{1}{\delta}$', size=100)
ax.set_ylim([325000, 520000])
ax.set_ylabel(r'Total Cost  ($\times \ 10^5$)', size=100)

# ax.set_title('FLSEA Parameters', size=15)
ax.set_xticks(x, delta_vals, size=100)
# ax.legend(loc='best', prop={'size': 70})
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
ax.legend(fontsize='100', ncol=2,handleheight=2.4, labelspacing=0.05, loc='lower right')

# ax.bar_label(rects1, padding=3, size=15)
# ax.bar_label(rects2, padding=3, size = 15)
fig.set_figwidth(60)
fig.set_figheight(40)

plt.show()
fig.savefig('total-cost-varying-C-bernoulli.pdf', dpi=300)

# # ----------------------------------------------------------------------------------------------------------'

df1 = pd.read_csv('FedElim-and-periodic-comm-cost-comparison-bernoulli.csv')

a1 = df1['comm-cost-H-is-1'].to_numpy(dtype=int)
a2 = df1['comm-cost-H-is-5'].to_numpy(dtype=int)
a3 = df1['comm-cost-H-is-10'].to_numpy(dtype=int)
a4 = df1['comm-cost-FedElim'].to_numpy(dtype=int)
err1 = df1['error-comm-cost-H-is-1'].to_numpy(dtype=float)
err2 = df1['error-comm-cost-H-is-5'].to_numpy(dtype=float)
err3 = df1['error-comm-cost-H-is-10'].to_numpy(dtype=float)
err4 = df1['error-comm-cost-FedElim'].to_numpy(dtype=float)


fig, ax = plt.subplots()
x = np.arange(len(a1))
rects1 = ax.bar(x - 1.5*width, a4, width, 
                yerr = 2*err4, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=10, capsize=12, capthick=10),
                label=r'FedElim')
rects2 = ax.bar(x - 0.5*width, a3, width, yerr = 2*err3, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=10, capsize=12, capthick=10),
                label=r'$H=10$')
rects3 = ax.bar(x + 0.5*width, a2, width, yerr = 2*err2, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=10, capsize=12, capthick=10),
                label=r'$H=5$')
rects4 = ax.bar(x + 1.5*width, a1, width, yerr = 2*err1, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=10, capsize=12, capthick=10),
                label=r'$H=1$')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis='y', which='major', labelsize=100)
ax.set_xlabel(r'$\ln \frac{1}{\delta}$', size=100)
ax.set_ylabel(r'$\ln($Communication cost$)$', size=100)
# ax.set_title(r'Comm Cost: Periodic Communication with period $H$ vs FedElim', size=30)
ax.set_xticks(x, delta_vals, size=100)
# ax.legend(loc='best', prop={'size': 50})
ax.legend(fontsize='100', ncol=2,handleheight=2.4, labelspacing=0.05, loc='lower right')


# ax.bar_label(rects1, padding=3, size=15)
# ax.bar_label(rects2, padding=3, size = 15)
# ax.bar_label(rects3, padding=3, size = 15)
# ax.bar_label(rects4, padding=3, size = 15)

fig.set_figwidth(60)
fig.set_figheight(40)
ax.autoscale()
plt.show()
fig.savefig('comm-cost-comparison-bernoulli.pdf', dpi=300)

# ----------------------------------------------------------------------------------------------------------'

width = 0.13
df1 = pd.read_csv('FedElim-and-periodic-comm-total-cost-comparison-bernoulli.csv')

a1 = df1['total-cost-H=1'].to_numpy(dtype=int)
a2 = df1['total-cost-H=10'].to_numpy(dtype=int)
a3 = df1['total-cost-H=100'].to_numpy(dtype=int)
a4 = df1['total-cost-H=1000'].to_numpy(dtype=int)
a5 = df1['total-cost-H=10000'].to_numpy(dtype=int)
a6 = df1['total-cost-H=100000'].to_numpy(dtype=int)
a7 = df1['total-cost-FedElim'].to_numpy(dtype=int)

err1 = df1['error-bar-H=1'].to_numpy(dtype=float)
err2 = df1['error-bar-H=10'].to_numpy(dtype=float)
err3 = df1['error-bar-H=100'].to_numpy(dtype=float)
err4 = df1['error-bar-H=1000'].to_numpy(dtype=float)
err5 = df1['error-bar-H=10000'].to_numpy(dtype=float)
err6 = df1['error-bar-H=100000'].to_numpy(dtype=float)
err7 = df1['error-bar-FedElim'].to_numpy(dtype=float)

fig, ax = plt.subplots()
x = np.arange(len(a1))
rects1 = ax.bar(x - 4*width, a1, width, yerr = err1, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=5, capsize=8, capthick=5),
                label=r'$H=1$')
rects2 = ax.bar(x - 3*width, a2, width, yerr = err2, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=5, capsize=8, capthick=5),
                label=r'$H=10$')
rects3 = ax.bar(x - 2*width, a3, width, yerr = err3, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=5, capsize=8, capthick=5),
                label=r'$H=10^2$')
rects4 = ax.bar(x-width, a4, width, yerr = err4, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=5, capsize=8, capthick=5),
                label=r'$H=10^3$')
rects5 = ax.bar(x, a7, width, yerr = err7, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=5, capsize=8, capthick=5),
                label=r'FedElim')
rects6 = ax.bar(x+width, a5, width, yerr = err5, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=5, capsize=8, capthick=5),
                label=r'$H=10^4$')
rects7 = ax.bar(x + 2*width, a6, width, yerr = err6, 
                align = 'center', 
                alpha = 0.5,
                ecolor = 'purple',
                error_kw = dict(lw=5, capsize=8, capthick=5),
                label=r'$H=10^5$')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis='y', which='major', labelsize=100)
ax.set_xlabel(r'$\ln \frac{1}{\delta}$', size=100)
ax.set_ylim([8,15])
# ax.set_ylabel(r'Total cost ' + '(x $10^6$)', size=100)
ax.set_ylabel(r'$\ln($Total cost$)$', size=100)

# ax.set_title(r'Total Cost: Periodic Communication with period $H$ versus FedElim', size=30)
ax.set_xticks(x, delta_vals, size=100)
# ax.legend(loc='best', prop={'size': 50})
ax.legend(fontsize='100', ncol=3,handleheight=2.4, labelspacing=0.05, loc='lower right')

# ax.bar_label(rects1, padding=0, size=15)
# ax.bar_label(rects2, padding=0, size = 15)
# ax.bar_label(rects3, padding=0, size = 15)
# ax.bar_label(rects4, padding=0, size = 15)

fig.set_figwidth(60)
fig.set_figheight(40)
plt.show()
fig.savefig('FedElim-vs-periodic-total-cost-comparison-bernoulli.pdf', dpi=300)