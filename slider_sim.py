import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from scipy.integrate import odeint
import numpy as np
from matplotlib.widgets import Slider

types = ['M', 'NS', 'NP', 'NR', 'P', 'B', 'V', 'I', 'S']
type_dict = {k: i for i, k in enumerate(types)}
ID = type_dict

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


T, dt = 50, 0.1
ts = np.arange(0, T, dt)

P = 5E6
B = 0
S = 0
I = 0
V = 5E6
N = 2.9E8
M = 0


def model(y, t, params):
    FR,m,uN,uP,uV,KMN,KNP,kP,kV,kD,kM,YNM,YVI,YIV,YMP,YMV,YVM,YBP,YPB,YBN,YIN = params
    M, NS, NP, NR, P, B, V, I, S = y

    dM = - uN * M * (NS + NP + NR) / ((KMN + M) * YNM) + YMP * kP * B + YMV * kV * I
    dNS = (NS + NP) * uN * M / (KMN + M) - P * uP * NS / ((KNP + NS + NP + NR) * YBN) - V * uV * NS / YIN - kD * S * NS - kM * NS
    dNP = - V * uV * NP / YIN + kD * S * NS
    dNR = NR * uN * M / (KMN + M) - P * uP * NR / ((KNP + NS + NP + NR) * YBN) + kM * NS
    dP = kP * B - m * P - P * uP * (NR + NS) / ((KNP + NS + NP + NR) * YBP)
    dB = - kP * B / YPB + P * uP * (NR + NS) / (KNP + NS + NP + NR)
    dV = kV * I - V * uV * (NP + NS) / YIV
    dI = - kV * I / YVI + V * uV * (NP + NS)
    dS = kP * B / YPB

    return [dM, dNS, dNP, dNR, dP, dB, dV, dI, dS]


def update(val):
    params = (FRSlider.val,
              mSlider.val,
              uNSlider.val,
              uPSlider.val,
              uVSlider.val,
              KMNSlider.val,
              KNPSlider.val,
              kPSlider.val,
              kVSlider.val,
              kDSlider.val,
              kMSlider.val,
              YNMSlider.val,
              YVISlider.val,
              YIVSlider.val,
              YMPSlider.val,
              YMVSlider.val,
              YVMSlider.val,
              YBPSlider.val,
              YPBSlider.val,
              YBNSlider.val,
              YINSlider.val)

    print(params)

    solDual = odeint(model, [M, N * (1 - FRSlider.val), 0, N * FRSlider.val, P, B, V, I, S], ts, args=(params,)).clip(min=1)
    solP =    odeint(model, [M, N * (1 - FRSlider.val), 0, N * FRSlider.val, P, B, 0, I, S], ts, args=(params,)).clip(min=1)
    solV =    odeint(model, [M, N * (1 - FRSlider.val), 0, N * FRSlider.val, 0, B, V, I, S], ts, args=(params,)).clip(min=1)
    solN =    odeint(model, [M, N * (1 - FRSlider.val), 0, N * FRSlider.val, 0, B, 0, I, S], ts, args=(params,)).clip(min=1)

    DualNGraph.set_ydata(solDual[:,ID["NR"]] + solDual[:,ID["NS"]] + solDual[:,ID["NP"]])
    DualNRGraph.set_ydata(solDual[:,ID["NR"]])
    DualNSGraph.set_ydata(solDual[:,ID["NS"]])
    DualNPGraph.set_ydata(solDual[:,ID["NP"]])
    DualMGraph.set_ydata(solDual[:,ID["M"]])
    DualPGraph.set_ydata(solDual[:,ID["P"]])
    DualBGraph.set_ydata(solDual[:,ID["B"]])
    DualSGraph.set_ydata(solDual[:,ID["S"]])
    DualVGraph.set_ydata(solDual[:,ID["V"]])
    DualIGraph.set_ydata(solDual[:,ID["I"]])

    VNGraph.set_ydata(solV[:,ID["NR"]] + solV[:,ID["NS"]] + solV[:,ID["NP"]])
    VNRGraph.set_ydata(solV[:,ID["NR"]])
    VNSGraph.set_ydata(solV[:,ID["NS"]])
    VNPGraph.set_ydata(solV[:,ID["NP"]])
    VMGraph.set_ydata(solV[:,ID["M"]])
    VPGraph.set_ydata(solV[:,ID["P"]])
    VBGraph.set_ydata(solV[:,ID["B"]])
    VSGraph.set_ydata(solV[:,ID["S"]])
    VVGraph.set_ydata(solV[:,ID["V"]])
    VIGraph.set_ydata(solV[:,ID["I"]])

    PNGraph.set_ydata(solP[:,ID["NR"]] + solP[:,ID["NS"]] + solP[:,ID["NP"]])
    PNRGraph.set_ydata(solP[:,ID["NR"]])
    PNSGraph.set_ydata(solP[:,ID["NS"]])
    PNPGraph.set_ydata(solP[:,ID["NP"]])
    PMGraph.set_ydata(solP[:,ID["M"]])
    PPGraph.set_ydata(solP[:,ID["P"]])
    PBGraph.set_ydata(solP[:,ID["B"]])
    PSGraph.set_ydata(solP[:,ID["S"]])
    PVGraph.set_ydata(solP[:,ID["V"]])
    PIGraph.set_ydata(solP[:,ID["I"]])

    NNGraph.set_ydata(solN[:,ID["NR"]] + solN[:,ID["NS"]] + solN[:,ID["NP"]])
    NNRGraph.set_ydata(solN[:,ID["NR"]])
    NNSGraph.set_ydata(solN[:,ID["NS"]])
    NNPGraph.set_ydata(solN[:,ID["NP"]])
    NMGraph.set_ydata(solN[:,ID["M"]])
    NPGraph.set_ydata(solN[:,ID["P"]])
    NBGraph.set_ydata(solN[:,ID["B"]])
    NSGraph.set_ydata(solN[:,ID["S"]])
    NVGraph.set_ydata(solN[:,ID["V"]])
    NIGraph.set_ydata(solN[:,ID["I"]])

    plt.draw()

data = pd.read_csv("Nottingham Phage Data.csv")

fig, ((axDual, axV), (axP, axN)) = plt.subplots(2, 2)
fig.set_size_inches(15,10)
plt.subplots_adjust(left=0.43)

DualNGraph, = axDual.plot(ts, np.zeros(ts.shape), label="N", color=COLORS["N"])
DualNRGraph, = axDual.plot(ts, np.zeros(ts.shape), label="NR", color=COLORS["NR"])
DualNSGraph, = axDual.plot(ts, np.zeros(ts.shape), label="NS", color=COLORS["NS"])
DualNPGraph, = axDual.plot(ts, np.zeros(ts.shape), label="NP", color=COLORS["NP"])
DualMGraph, = axDual.plot(ts, np.zeros(ts.shape), label="M", color=COLORS["M"])
DualPGraph, = axDual.plot(ts, np.zeros(ts.shape), label="P", color=COLORS["P"])
DualBGraph, = axDual.plot(ts, np.zeros(ts.shape), label="B", color=COLORS["B"])
DualSGraph, = axDual.plot(ts, np.zeros(ts.shape), label="S", color=COLORS["S"])
DualVGraph, = axDual.plot(ts, np.zeros(ts.shape), label="V", color=COLORS["V"])
DualIGraph, = axDual.plot(ts, np.zeros(ts.shape), label="I", color=COLORS["I"])
axDual.scatter(data["times"], data["EColiAll"], label="N", color=COLORS["N"])
axDual.scatter(data["times"], data["BdAll"], label="P", color=COLORS["P"])
axDual.scatter(data["times"], data["PhageAll"], label="V", color=COLORS["V"])
axDual.set_yscale("log")
axDual.set_ylim(bottom=1, top=1e11)

VNGraph, = axV.plot(ts, np.zeros(ts.shape), label="N", color=COLORS["N"])
VNRGraph, = axV.plot(ts, np.zeros(ts.shape), label="NR", color=COLORS["NR"])
VNSGraph, = axV.plot(ts, np.zeros(ts.shape), label="NS", color=COLORS["NS"])
VNPGraph, = axV.plot(ts, np.zeros(ts.shape), label="NP", color=COLORS["NP"])
VMGraph, = axV.plot(ts, np.zeros(ts.shape), label="M", color=COLORS["M"])
VPGraph, = axV.plot(ts, np.zeros(ts.shape), label="P", color=COLORS["P"])
VBGraph, = axV.plot(ts, np.zeros(ts.shape), label="B", color=COLORS["B"])
VSGraph, = axV.plot(ts, np.zeros(ts.shape), label="S", color=COLORS["S"])
VVGraph, = axV.plot(ts, np.zeros(ts.shape), label="V", color=COLORS["V"])
VIGraph, = axV.plot(ts, np.zeros(ts.shape), label="I", color=COLORS["I"])
axV.scatter(data["times"], data["EColiWithPhage"], label="N", color=COLORS["N"])
axV.scatter(data["times"], data["PhageWithEColiOnly"], label="V", color=COLORS["V"])
axV.set_yscale("log")
axV.set_ylim(bottom=1, top=1e11)

PNGraph, = axP.plot(ts, np.zeros(ts.shape), label="N", color=COLORS["N"])
PNRGraph, = axP.plot(ts, np.zeros(ts.shape), label="NR", color=COLORS["NR"])
PNSGraph, = axP.plot(ts, np.zeros(ts.shape), label="NS", color=COLORS["NS"])
PNPGraph, = axP.plot(ts, np.zeros(ts.shape), label="NP", color=COLORS["NP"])
PMGraph, = axP.plot(ts, np.zeros(ts.shape), label="M", color=COLORS["M"])
PPGraph, = axP.plot(ts, np.zeros(ts.shape), label="P", color=COLORS["P"])
PBGraph, = axP.plot(ts, np.zeros(ts.shape), label="B", color=COLORS["B"])
PSGraph, = axP.plot(ts, np.zeros(ts.shape), label="S", color=COLORS["S"])
PVGraph, = axP.plot(ts, np.zeros(ts.shape), label="V", color=COLORS["V"])
PIGraph, = axP.plot(ts, np.zeros(ts.shape), label="I", color=COLORS["I"])
axP.scatter(data["times"], data["EColiWithBd"], label="N", color=COLORS["N"])
axP.scatter(data["times"], data["BdWithEColiOnly"], label="P", color=COLORS["P"])
axP.set_yscale("log")
axP.set_ylim(bottom=1, top=1e11)


NNGraph, = axN.plot(ts, np.zeros(ts.shape), label="N", color=COLORS["N"])
NNRGraph, = axN.plot(ts, np.zeros(ts.shape), label="NR", color=COLORS["NR"])
NNSGraph, = axN.plot(ts, np.zeros(ts.shape), label="NS", color=COLORS["NS"])
NNPGraph, = axN.plot(ts, np.zeros(ts.shape), label="NP", color=COLORS["NP"])
NMGraph, = axN.plot(ts, np.zeros(ts.shape), label="M", color=COLORS["M"])
NPGraph, = axN.plot(ts, np.zeros(ts.shape), label="P", color=COLORS["P"])
NBGraph, = axN.plot(ts, np.zeros(ts.shape), label="B", color=COLORS["B"])
NSGraph, = axN.plot(ts, np.zeros(ts.shape), label="S", color=COLORS["S"])
NVGraph, = axN.plot(ts, np.zeros(ts.shape), label="V", color=COLORS["V"])
NIGraph, = axN.plot(ts, np.zeros(ts.shape), label="I", color=COLORS["I"])
NNGraph, = axN.plot(ts, np.zeros(ts.shape), label="N", color=COLORS["N"])
axN.scatter(data["times"], data["EColiOnly"], label="N", color=COLORS["N"])
axN.set_yscale("log")
axN.set_ylim(bottom=1, top=1e11)


axFR  = plt.axes([0.055, 0.84, 0.25, 0.02])
axm   = plt.axes([0.055, 0.8, 0.25, 0.02])
axuN  = plt.axes([0.055, 0.76, 0.25, 0.02])
axuP  = plt.axes([0.055, 0.72, 0.25, 0.02])
axuV  = plt.axes([0.055, 0.68, 0.25, 0.02])
axKMN = plt.axes([0.055, 0.64, 0.25, 0.02])
axKNP = plt.axes([0.055, 0.6, 0.25, 0.02])
axkP  = plt.axes([0.055, 0.56, 0.25, 0.02])
axkV  = plt.axes([0.055, 0.52, 0.25, 0.02])
axkD  = plt.axes([0.055, 0.48, 0.25, 0.02])
axkM  = plt.axes([0.055, 0.44, 0.25, 0.02])
axYNM = plt.axes([0.055, 0.4, 0.25, 0.02])
axYVI = plt.axes([0.055, 0.36, 0.25, 0.02])
axYIV = plt.axes([0.055, 0.32, 0.25, 0.02])
axYMP = plt.axes([0.055, 0.28, 0.25, 0.02])
axYMV = plt.axes([0.055, 0.24, 0.25, 0.02])
axYVM = plt.axes([0.055, 0.2, 0.25, 0.02])
axYBP = plt.axes([0.055, 0.16, 0.25, 0.02])
axYPB = plt.axes([0.055, 0.12, 0.25, 0.02])
axYBN = plt.axes([0.055, 0.08, 0.25, 0.02])
axYIN = plt.axes([0.055, 0.04, 0.25, 0.02])

FRSlider  = Slider(axFR,  'FR',  0,     1E-5,  valinit=8.6E-6) # valinit=8.6E-6)
mSlider   = Slider(axm,   'm',   0,     1E0,   valinit=6.4E-2) # valinit=6.4E-2)
uNSlider  = Slider(axuN,  'uN',  0,     1E0,   valinit=0.575) # valinit=0.46)
uPSlider  = Slider(axuP,  'uP',  0,     1E0,   valinit=0.33) # valinit=0.33)
uVSlider  = Slider(axuV,  'uV',  0,     1E-7,  valinit=3.2E-9) # valinit=3.9E-9)
KMNSlider = Slider(axKMN, 'KMN', 1E6,   1E8,   valinit=3.1624E7) # valinit=4.2E7)
KNPSlider = Slider(axKNP, 'KNP', 1E5,   1E7,   valinit=5.314E6) # valinit=3.2E6)
kPSlider  = Slider(axkP,  'kP',  0,     1E1,   valinit=1.04) # valinit=1.1)
kVSlider  = Slider(axkV,  'kV',  0,     1E1,   valinit=3.36) # valinit=4.2)
kDSlider  = Slider(axkD,  'kD',  0,     1E-10, valinit=4.0e-12) # valinit=4.0E-12)
kMSlider  = Slider(axkM,  'kM',  0,     1E-8,  valinit=7.5E-10) # valinit=7.5E-10)
YNMSlider = Slider(axYNM, 'YNM', 1E-10, 1E1,   valinit=1.2) # valinit=1.2)
YVISlider = Slider(axYVI, 'YVI', 1E-10, 1E2,   valinit=24) # valinit=24)
YIVSlider = Slider(axYIV, 'YIV', 1E-10, 1E2,   valinit=1) # valinit=1)
YMPSlider = Slider(axYMP, 'YMP', 1E-10, 1E0,   valinit=0.0213) # valinit=3.3E-3)
YMVSlider = Slider(axYMV, 'YMV', 1E-10, 1E0,   valinit=0.012) # valinit=2.1E-2)
YVMSlider = Slider(axYVM, 'YVM', 1E-10, 1E1,   valinit=1) # valinit=1)
YBPSlider = Slider(axYBP, 'YBP', 1E-10, 1E1,   valinit=1) # valinit=1)
YPBSlider = Slider(axYPB, 'YPB', 1E-10, 1E1,   valinit=4.17) # valinit=4.17)
YBNSlider = Slider(axYBN, 'YBN', 1E-10, 1E1,   valinit=1) # valinit=1)
YINSlider = Slider(axYIN, 'YIN', 1E-10, 1E1,   valinit=1) # valinit=1)

# FRSlider  = Slider(axFR,  'FR',  0,     1E-5,  valinit=8.6E-6)
# mSlider   = Slider(axm,   'm',   0,     1E0,   valinit=6.4E-2)
# uNSlider  = Slider(axuN,  'uN',  0,     1E0,   valinit=0.46)
# uPSlider  = Slider(axuP,  'uP',  0,     1E0,   valinit=0.33)
# uVSlider  = Slider(axuV,  'uV',  0,     1E-7,  valinit=3.9E-9)
# KMNSlider = Slider(axKMN, 'KMN', 1E6,   1E8,   valinit=4.2E7)
# KNPSlider = Slider(axKNP, 'KNP', 1E5,   1E7,   valinit=3.2E6)
# kPSlider  = Slider(axkP,  'kP',  0,     1E1,   valinit=1.1)
# kVSlider  = Slider(axkV,  'kV',  0,     1E1,   valinit=4.2)
# kDSlider  = Slider(axkD,  'kD',  0,     1E-10, valinit=4.0E-12)
# kMSlider  = Slider(axkM,  'kM',  0,     1E-8,  valinit=7.5E-10)
# YNMSlider = Slider(axYNM, 'YNM', 1E-10, 1E1,   valinit=1.2)
# YVISlider = Slider(axYVI, 'YVI', 1E-10, 1E2,   valinit=24)
# YIVSlider = Slider(axYIV, 'YIV', 1E-10, 1E2,   valinit=1)
# YMPSlider = Slider(axYMP, 'YMP', 1E-10, 1E0,   valinit=3.3E-3)
# YMVSlider = Slider(axYMV, 'YMV', 1E-10, 1E0,   valinit=2.1E-2)
# YVMSlider = Slider(axYVM, 'YVM', 1E-10, 1E1,   valinit=1)
# YBPSlider = Slider(axYBP, 'YBP', 1E-10, 1E1,   valinit=1)
# YPBSlider = Slider(axYPB, 'YPB', 1E-10, 1E1,   valinit=4.17)
# YBNSlider = Slider(axYBN, 'YBN', 1E-10, 1E1,   valinit=1)
# YINSlider = Slider(axYIN, 'YIN', 1E-10, 1E1,   valinit=1)

FRSlider.on_changed(update)
mSlider.on_changed(update)
uNSlider.on_changed(update)
uPSlider.on_changed(update)
uVSlider.on_changed(update)
KMNSlider.on_changed(update)
KNPSlider.on_changed(update)
kPSlider.on_changed(update)
kVSlider.on_changed(update)
kDSlider.on_changed(update)
kMSlider.on_changed(update)
YNMSlider.on_changed(update)
YVISlider.on_changed(update)
YIVSlider.on_changed(update)
YMPSlider.on_changed(update)
YMVSlider.on_changed(update)
YVMSlider.on_changed(update)
YBPSlider.on_changed(update)
YPBSlider.on_changed(update)
YBNSlider.on_changed(update)
YINSlider.on_changed(update)

update(0)

plt.show()


