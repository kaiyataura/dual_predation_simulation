import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

RUN = 'N'
SCALE = 8
scale = 2900 * SCALE * SCALE / 2.9E8
scale = 1

exp = pd.read_csv('Nottingham Phage Data.csv')
print(exp)
dif = pd.read_csv(f'simdata/{RUN}.csv')
# sim = pd.read_csv(f'data_{RUN}_{SCALE}x_P0.0011_B0.004_I0.005_NS+S0.1_P+NR0.6_P+NS0.6_V+NS0.8_V+NP0.8_NR+M0.99999,1e-05_NS+M0.99999,1e-05_NP+M1.csv')
sim = pd.read_csv(f'data_{RUN}_gray_scott.csv').clip(lower=1)
exp.iloc[:,1:] = (exp.iloc[:,1:] * scale).clip(lower=1)
dif.iloc[:,2:] = (dif.iloc[:,2:] * scale).clip(lower=1)

COLORS = {"N":  (0.0, 0.0, 0.0),
          "NR": (0.0, 0.0, 1.0),
          "NS": (0.0, 1.0, 1.0),
          "NP": (0.0, 0.5, 1.0),
          "M":  (0.75, 0.0, 1.0),
          "P":  (1.0, 0.0, 0.0),
          "B":  (1.0, 0.5, 0.0),
          "S":  (1.0, 0.0, 1.0),
          "V":  (0.0, 1.0, 0.0),
          "I":  (1.0, 1.0, 0.0)}

N = len(exp['times'])
# N = 6

simN = (interp1d(sim['t'], sim['N'], kind='linear', fill_value='extrapolate'))(exp[:N]['times'])
simP = (interp1d(sim['t'], sim['P'], kind='linear', fill_value='extrapolate'))(exp[:N]['times'])
simV = (interp1d(sim['t'], sim['V'], kind='linear', fill_value='extrapolate'))(exp[:N]['times'])
difN = (interp1d(dif['t'], dif['N'], kind='linear', fill_value='extrapolate'))(exp[:N]['times'])
difP = (interp1d(dif['t'], dif['P'], kind='linear', fill_value='extrapolate'))(exp[:N]['times'])
difV = (interp1d(dif['t'], dif['V'], kind='linear', fill_value='extrapolate'))(exp[:N]['times'])


def score(sim, exp):
    return np.sqrt(np.sum(np.square(np.log10(sim) - np.log10(exp)))) / N

if RUN == "Dual":
    scoreSimN = score(simN, exp[:N]['EColiAll'])
    scoreSimP = score(simP, exp[:N]['BdAll'])
    scoreSimV = score(simV, exp[:N]['PhageAll'])
    scoreDifN = score(difN, exp[:N]['EColiAll'])
    scoreDifP = score(difP, exp[:N]['BdAll'])
    scoreDifV = score(difV, exp[:N]['PhageAll'])
    print(f'simN: {scoreSimN}, simP: {scoreSimP}, simV: {scoreSimV}')
    print(f'difN: {scoreDifN}, difP: {scoreDifP}, difV: {scoreDifV}')
    plt.plot(sim["t"], sim["N"], label="N", color=COLORS["N"])
    plt.plot(sim["t"], sim["P"], label="P", color=COLORS["P"])
    plt.plot(sim["t"], sim["V"], label="V", color=COLORS["V"])
    plt.plot(dif["t"], dif["N"], label="N", color=(*COLORS["N"], 0.3))
    plt.plot(dif["t"], dif["P"], label="P", color=(*COLORS["P"], 0.3))
    plt.plot(dif["t"], dif["V"], label="V", color=(*COLORS["V"], 0.3))
    plt.scatter(exp["times"], exp["EColiAll"], label="N", color=COLORS["N"])
    plt.scatter(exp["times"], exp["BdAll"], label="P", color=COLORS["P"])
    plt.scatter(exp["times"], exp["PhageAll"], label="V", color=COLORS["V"])
elif RUN == "V":
    scoreSimN = score(simN, exp[:N]['EColiWithPhage'])
    scoreSimV = score(simV, exp[:N]['PhageWithEColiOnly'])
    scoreDifN = score(difN, exp[:N]['EColiWithPhage'])
    scoreDifV = score(difV, exp[:N]['PhageWithEColiOnly'])
    print(f'simN: {scoreSimN}, simV: {scoreSimV}')
    print(f'difN: {scoreDifN}, difV: {scoreDifV}')
    plt.plot(sim["t"], sim["N"], label="N", color=COLORS["N"])
    plt.plot(sim["t"], sim["V"], label="V", color=COLORS["V"])
    plt.plot(dif["t"], dif["N"], label="N", color=(*COLORS["N"], 0.3))
    plt.plot(dif["t"], dif["V"], label="V", color=(*COLORS["V"], 0.3))
    plt.scatter(exp["times"], exp["EColiWithPhage"], label="N", color=COLORS["N"])
    plt.scatter(exp["times"], exp["PhageWithEColiOnly"], label="V", color=COLORS["V"])
elif RUN == "P":
    scoreSimN = score(simN, exp[:N]['EColiWithBd'])
    scoreSimP = score(simP, exp[:N]['BdWithEColiOnly'])
    scoreDifN = score(difN, exp[:N]['EColiWithBd'])
    scoreDifP = score(difP, exp[:N]['BdWithEColiOnly'])
    print(f'simN: {scoreSimN}, simP: {scoreSimP}')
    print(f'difN: {scoreDifN}, difP: {scoreDifP}')
    plt.plot(sim["t"], sim["N"], label="N", color=COLORS["N"])
    plt.plot(sim["t"], sim["P"], label="P", color=COLORS["P"])
    plt.plot(dif["t"], dif["N"], label="N", color=(*COLORS["N"], 0.3))
    plt.plot(dif["t"], dif["P"], label="P", color=(*COLORS["P"], 0.3))
    plt.scatter(exp["times"], exp["EColiWithBd"], label="N", color=COLORS["N"])
    plt.scatter(exp["times"], exp["BdWithEColiOnly"], label="P", color=COLORS["P"])
elif RUN == "N":
    scoreSimN = score(simN, exp[:N]['EColiOnly'])
    scoreDifN = score(difN, exp[:N]['EColiOnly'])
    print(f'simN: {scoreSimN}')
    print(f'difN: {scoreDifN}')
    plt.plot(sim["t"], sim["N"], label="N", color=COLORS["N"])
    plt.plot(dif["t"], dif["N"], label="N", color=(*COLORS["N"], 0.3))
    plt.scatter(exp["times"], exp["EColiOnly"], label="N", color=COLORS["N"])

plt.yscale('log')
plt.ylim(bottom=0.5)
plt.show()
