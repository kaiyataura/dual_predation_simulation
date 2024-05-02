import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np


FR = 8.6E-6
m = 6.4E-2

uN = 0.46
uP = 0.33
uV = 3.9E-9

KMN = 4.2E7
KNP = 3.2E6

kP = 1.1
kV = 4.2
kD = 4.0E-12
kM = 7.5E-10

YNM = 1.2
YVI = 24
YIV = 1
YMP = 3.3E-3
YMV = 2.1E-2
YVM = 1
YPB = 4.17 # fixed
YBP = 1
YBN = 1
YIN = 1

T, dt = 50, 0.001
t = 0
P = 0
B = 0
S = 0
I = 0
V = 0
NR = 2.9E8 * FR
NS = 2.9E8 * (1 - FR)
NP = 0
M = 0

history = DataFrame(columns=["t", "P", "B", "S", "I", "V", "NR", "NS", "NP", "M"], index=range(int(T / dt + 1)))

history.loc[0] = {"t": t, "P": P, "B": B, "S": S, "I": I, "V": V, "NR": NR, "NS": NS, "NP": NP, "M": M}
for i in range(1, int(T / dt) + 1):
    dM = - uN * M * (NS + NP + NR) / ((KMN + M) * YNM) + YMP * kP * B + YMV * kV * I
    dNS = (NS + NP) * uN * M / (KMN + M) - P * uP * NS / ((KNP + NS + NP + NR) * YBN) - V * uV * NS / YIN - kD * S * NS - kM * NS
    dNP = - V * uV * NP / YIN + kD * S * NS
    dNR = NR * uN * M / (KMN + M) - P * uP * NR / ((KNP + NS + NP + NR) * YBN) + kM * NS
    dP = kP * B - m * P - P * uP * (NR + NS) / ((KNP + NS + NP + NR) * YBP)
    dB = - kP * B / YPB + P * uP * (NR + NS) / (KNP + NS + NP + NR)
    dV = kV * I - V * uV * (NP + NS) / YIV
    dI = - kV * I / YVI + V * uV * (NP + NS)
    dS = kP * B / YPB
    
    # print(f"dM: {dM:.3E}, dNS: {dNS:.3E}, dNP: {dNP:.3E}, dNR: {dNR:.3E}, dP: {dP:.3E}, dB: {dB:.3E}, dV: {dV:.3E}, dI: {dI:.3E}, dS: {dS:.3E}, dN: {(dNS + dNP + dNR):.3E}")
    M = max(M + dM * dt, 0)
    NS = max(NS + dNS * dt, 0)
    NP = max(NP + dNP * dt, 0)
    NR = max(NR + dNR * dt, 0)
    P = max(P + dP * dt, 0)
    B = max(B + dB * dt, 0)
    V = max(V + dV * dt, 0)
    I = max(I + dI * dt, 0)
    S = max(S + dS * dt, 0)
    t = max(t + dt, 0)

    history.loc[i] = [t,P,B,S,I,V,NR,NS,NP,M]


history["N"] = history["NS"] + history["NP"] + history["NR"]
history[["M","NS","NP","NR","P","B","V","I","S","N"]] = history[["M","NS","NP","NR","P","B","V","I","S","N"]].clip(lower=1)

print(history)

COLORS = {"N": (0.0, 0.0, 0.0),
          "NR": (0.0, 0.0, 1.0),
          "NS": (0.0, 1.0, 1.0),
          "NP": (0.0, 0.5, 1.0),
          "M":  (0.75, 0.0, 1.0),
          "P":  (1.0, 0.0, 0.0),
          "B":  (1.0, 0.5, 0.0),
          "S":  (1.0, 0.0, 1.0),
          "V":  (0.0, 1.0, 0.0),
          "I":  (1.0, 1.0, 0.0)}

# plt.plot(history["t"], history["NR"], label="NR", color=COLORS["NR"])
# plt.plot(history["t"], history["NS"], label="NS", color=COLORS["NS"])
# plt.plot(history["t"], history["NP"], label="NP", color=COLORS["NP"])

# plt.plot(history["t"], history["B"], label="B", color=COLORS["B"])
# plt.plot(history["t"], history["I"], label="I", color=COLORS["I"])
# plt.plot(history["t"], history["S"], label="S", color=COLORS["S"])

# plt.plot(history["t"], history["M"], label="M", color=COLORS["M"])
# plt.plot(history["t"], history["V"], label="V", color=COLORS["V"])
# plt.plot(history["t"], history["P"], label="P", color=COLORS["P"])
plt.plot(history["t"], history["N"], label="N", color=COLORS["N"])

RUN = "N"
data = pd.read_csv("Nottingham Phage Data.csv")
speed = 1
if RUN == "Dual":
    plt.scatter(data["times"] * speed, (data["EColiAll"]).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(data["times"] * speed, (data["BdAll"]).clip(lower=1), label="P", color=COLORS["P"])
    plt.scatter(data["times"] * speed, (data["PhageAll"]).clip(lower=1), label="V", color=COLORS["V"])
elif RUN == "V":
    plt.scatter(data["times"] * speed, (data["EColiWithPhage"]).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(data["times"] * speed, (data["PhageWithEColiOnly"]).clip(lower=1), label="V", color=COLORS["V"])
elif RUN == "P":
    plt.scatter(data["times"] * speed, (data["EColiWithBd"]).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(data["times"] * speed, (data["BdWithEColiOnly"]).clip(lower=1), label="P", color=COLORS["P"])
elif RUN == "N":
    plt.scatter(data["times"] * speed, (data["EColiOnly"]).clip(lower=1), label="N", color=COLORS["N"])

plt.xlabel("Time")
plt.ylim(bottom=0.5, top=1e11)
plt.yscale("log")
plt.ylabel("Concentration")
plt.legend()

plt.show()
