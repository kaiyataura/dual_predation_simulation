import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.collections import EllipseCollection
import numpy as np
import time
import math
import random
from collections import deque
from QuadTree import QuadTree

SCALE = 8
WIDTH, HEIGHT = SCALE * 1000, SCALE * 1000 
COOLDOWN = 20
FRAMES = 4800
QT_N = 30
dt = 0.01

SETUPS = {"Dual": {"NS": 2900 * SCALE * SCALE, "NR": 1, "V": 50 * SCALE * SCALE,  "P": 50 * SCALE * SCALE},
          "V":    {"NS": 2900 * SCALE * SCALE, "NR": 1, "V": 50 * SCALE * SCALE},
          "P":    {"NS": 2900 * SCALE * SCALE, "NR": 1,                           "P": 50 * SCALE * SCALE},
          "N":    {"NS": 2900 * SCALE * SCALE, "NR": 1}}

RUN = "V"

REACTIONS = dict()

# P -m-> ø                        predator mortality
# P + NR -uP-> P + B              bacteriovorus attack NR
# P + NS -uP-> P + B              bacteriovorus attack NS
# B -kP-> (YPB) P + S + (YMP) M   bacteriovorus maturation
# S + NS -kD-> NP                 plastic resistance
# V + NS -uV-> I                  phage attack NS
# V + NP -uV-> I                  phage attack NP
# I -kV-> (YVI) V + (YMV) M       infected cell lysis
# (1/YNM) M + NR -uN-> 2 NR       resistant prey growth
# (1/YNM) M + NS -uN-> 2 NS       sensitive prey growth
# (1/YNM) M + NP -uN-> 2 NS       growth coupled reversion
# NS -kM-> NR                     phage resistance


# REACTIONS["P"] = [{"out": {}, "prob": 0.0011}]
# REACTIONS["S"] = [{"out": {}, "prob": 0.001}]
# REACTIONS["B"] = [{"out": {"P": 4.17, "S": 2, "M": 0.3}, "prob": 0.004}]
# REACTIONS["I"] = [{"out": {"V": 24, "M": 0.35}, "prob": 0.004}]
# # REACTIONS["B"] = [{"out": {"P": 4.17, "S": 2, "M": 0.3}, "prob": 0.01}]
# # REACTIONS["I"] = [{"out": {"V": 24, "M": 0.35}, "prob": 0.005}]
# REACTIONS["NR"] = [{"out": {"NS": 1}, "prob": 0.00001}]
# REACTIONS["NS"] = [{"out": {"NR": 1}, "prob": 0.00001}]

# # REACTIONS[("P", "NR")] = [{"out": {"B": 1}, "prob": 0.01}]
# # REACTIONS[("P", "NS")] = [{"out": {"B": 1}, "prob": 0.01}]
# REACTIONS[("P", "NR")] = [{"out": {"B": 1}, "prob": 1}]
# REACTIONS[("P", "NS")] = [{"out": {"B": 1}, "prob": 1}]

# REACTIONS[("NS", "S")] = [{"out": {"NP": 1}, "prob": 1}]

# # REACTIONS[("V", "NS")] = [{"out": {"I": 1}, "prob": 0.065}]
# # REACTIONS[("V", "NP")] = [{"out": {"I": 1}, "prob": 0.065}]
# REACTIONS[("V", "NS")] = [{"out": {"I": 1}, "prob": 1}]
# REACTIONS[("V", "NP")] = [{"out": {"I": 1}, "prob": 1}]

# REACTIONS[("NR", "M")] = [{"out": {"NR": 2}, "prob": 1}]
# REACTIONS[("NS", "M")] = [{"out": {"NS": 2}, "prob": 1}]
# REACTIONS[("NP", "M")] = [{"out": {"NS": 2}, "prob": 1}]

# REACTIONS["P"] = [{"out": {}, "prob": 0.0011}]
# REACTIONS["B"] = [{"out": {"P": 4.17, "S": 2, "M": 0.3}, "prob": 0.004}]
# REACTIONS["I"] = [{"out": {"V": 24, "M": 0.35}, "prob": 0.004}]

# REACTIONS[("NS", "S")] = [{"out": {"NP": 1}, "prob": 1}]

# REACTIONS[("P", "NR")] = [{"out": {"B": 1}, "prob": 1}]
# REACTIONS[("P", "NS")] = [{"out": {"B": 1}, "prob": 1}]
# REACTIONS[("V", "NS")] = [{"out": {"I": 1}, "prob": 1}]
# REACTIONS[("V", "NP")] = [{"out": {"I": 1}, "prob": 1}]

# REACTIONS[("NR", "M")] = [{"out": {"NR": 2}, "prob": 0.99999},
#                           {"out": {"NR": 1, "NS": 1} , "prob": 0.00001}]
# REACTIONS[("NS", "M")] = [{"out": {"NS": 2}, "prob": 0.99999},
#                           {"out": {"NS": 1, "NR": 1} , "prob": 0.00001}]
# REACTIONS[("NP", "M")] = [{"out": {"NS": 2}, "prob": 1}]

REACTIONS["P"] = [{"out": {}, "prob": 0.0011}]
REACTIONS["B"] = [{"out": {"P": 4.17, "S": 1, "M": 0.3}, "prob": 0.004}]
REACTIONS["I"] = [{"out": {"V": 24, "M": 0.35}, "prob": 0.005}]

REACTIONS[("NS", "S")] = [{"out": {"NP": 1}, "prob": 0.1}]

REACTIONS[("P", "NR")] = [{"out": {"B": 1}, "prob": 0.6}]
REACTIONS[("P", "NS")] = [{"out": {"B": 1}, "prob": 0.6}]
REACTIONS[("V", "NS")] = [{"out": {"I": 1}, "prob": 0.8}]
REACTIONS[("V", "NP")] = [{"out": {"I": 1}, "prob": 0.8}]

REACTIONS[("NR", "M")] = [{"out": {"NR": 2}, "prob": 0.99999},
                          {"out": {"NR": 1, "NS": 1} , "prob": 0.00001}]
REACTIONS[("NS", "M")] = [{"out": {"NS": 2}, "prob": 0.99999},
                          {"out": {"NS": 1, "NR": 1} , "prob": 0.00001}]
REACTIONS[("NP", "M")] = [{"out": {"NS": 2}, "prob": 1}]

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

RADII = {"NR": 5,
         "NS": 5,
         "NP": 5,
         "M":  1,
         "P":  3,
         "B":  5,
         "S":  1,
         "V":  2,
         "I":  5}

class Particle:
    def __init__(self, name, x, y, cooldown=0):
        self.name = name
        self.x = x
        self.y = y
        self.r = RADII[name]
        self.cooldown = cooldown
        self.delete = False
        self.randomVel()

    def randomVel(self):
        d = 2 * math.pi * random.random()
        r = 100 * random.random()
        self.vx = r * math.cos(d)
        self.vy = r * math.sin(d)

    def move(self):
        self.x += dt * self.vx
        self.y += dt * self.vy

        if self.x < self.r: self.vx = abs(self.vx); self.x = self.r
        if self.x >= WIDTH - self.r: self.vx = -abs(self.vx); self.x = WIDTH - self.r
        if self.y < self.r: self.vy = abs(self.vy); self.y = self.r
        if self.y >= HEIGHT - self.r: self.vy = -abs(self.vy); self.y = HEIGHT - self.r

        if self.cooldown: self.cooldown -= 1

    def react(self, other):
        if self.delete or other.delete or self.cooldown or other.cooldown: return []

        reactions = REACTIONS.get((self.name, other.name), None)
        if not reactions: reactions = REACTIONS.get((other.name, self.name), None)
        if not reactions: return []

        rand = random.random()
        out = deque()
        prob = 0
        for reaction in reactions:
            prob += reaction['prob']
            if rand >= prob: self.cooldown = other.cooldown = COOLDOWN; continue

            self.delete = True
            other.delete = True
            
            for prod, num in reaction['out'].items():
                n = int(num)
                for _ in range(n + (random.random() < num - n)): 
                    out.append(Particle(prod, (self.x + other.x) / 2, (self.y + other.y) / 2, COOLDOWN))
            break
        return out

    def autoReact(self):
        if self.delete or self.cooldown: return []

        reactions = REACTIONS.get(self.name, None)
        if not reactions: return []

        rand = random.random()
        out = deque()
        prob = 0
        for reaction in reactions:
            prob += reaction['prob']
            if rand >= prob: continue

            self.delete = True
            
            for prod, num in reaction['out'].items():
                n = int(num)
                for _ in range(n + (random.random() < num - n)): 
                    out.append(Particle(prod, self.x, self.y, COOLDOWN))
            break
        return out
    
def update(frame):
    global qtree, history

    t1 = time.perf_counter()

    particles = qtree.query_all()
    add = deque()
    t3 = time.perf_counter()
    pairs = qtree.query_pairs()
    t4 = time.perf_counter()
    print(t4 - t3)
    for p1 in particles:
        for p in p1.autoReact():
            add.append(p)

    for p1, p2 in pairs:
        for p in p1.react(p2):
            add.append(p)

    ax1.clear()
    ax1.set(xlim=[0, WIDTH], ylim=[0, HEIGHT])

    diameters = [2 * p.r for p in particles]
    circles = EllipseCollection(diameters, diameters, 0, units='xy', 
                                facecolors=[COLORS[p.name] for p in particles], 
                                offsets=[(p.x, p.y) for p in particles],
                                transOffset=ax1.transData)

    ax1.add_collection(circles)

    NR, NS, NP, M, P, B, S, V, I = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for p in particles:
        if p.name == "NR": NR += 1
        if p.name == "NS": NS += 1
        if p.name == "NP": NP += 1
        if p.name == "M": M += 1
        if p.name == "P": P += 1
        if p.name == "B": B += 1
        if p.name == "S": S += 1
        if p.name == "V": V += 1
        if p.name == "I": I += 1

    print('N:', NR + NS + NP, 'NR:', NR, 'NS:', NS, 'NP:', NP, 'M:', M, 'P:', P, 'B:', B, 'S:', S, 'V:', V, 'I:', I)

    datapoint = np.array([dt * frame, NR + NS + NP, NR, NS, NP, M, P, B, S, V, I])
    datapoint[1:] = datapoint[1:].clip(min=1)
    history.append(datapoint)
    # if len(history) > 1000:  history.popleft()
    data = np.array(history)
    ts = data[:,0]
    NGraph.set_xdata(ts)
    NRGraph.set_xdata(ts)
    NSGraph.set_xdata(ts)
    NPGraph.set_xdata(ts)
    MGraph.set_xdata(ts)
    PGraph.set_xdata(ts)
    BGraph.set_xdata(ts)
    SGraph.set_xdata(ts)
    VGraph.set_xdata(ts)
    IGraph.set_xdata(ts)

    NGraph.set_ydata(data[:,1])
    NRGraph.set_ydata(data[:,2])
    NSGraph.set_ydata(data[:,3])
    NPGraph.set_ydata(data[:,4])
    MGraph.set_ydata(data[:,5])
    PGraph.set_ydata(data[:,6])
    BGraph.set_ydata(data[:,7])
    SGraph.set_ydata(data[:,8])
    VGraph.set_ydata(data[:,9])
    IGraph.set_ydata(data[:,10])

    data = simData[:frame+1]

    SimNGraph.set_xdata(data["t"])
    SimNRGraph.set_xdata(data["t"])
    SimNSGraph.set_xdata(data["t"])
    SimNPGraph.set_xdata(data["t"])
    SimMGraph.set_xdata(data["t"])
    SimPGraph.set_xdata(data["t"])
    SimBGraph.set_xdata(data["t"])
    SimSGraph.set_xdata(data["t"])
    SimVGraph.set_xdata(data["t"])
    SimIGraph.set_xdata(data["t"])

    SimNGraph.set_ydata(data["N"])
    SimNRGraph.set_ydata(data["NR"])
    SimNSGraph.set_ydata(data["NS"])
    SimNPGraph.set_ydata(data["NP"])
    SimMGraph.set_ydata(data["M"])
    SimPGraph.set_ydata(data["P"])
    SimBGraph.set_ydata(data["B"])
    SimSGraph.set_ydata(data["S"])
    SimVGraph.set_ydata(data["V"])
    SimIGraph.set_ydata(data["I"])

    ax2.relim()
    ax2.autoscale()

    new = QuadTree(0, 0, WIDTH, HEIGHT, QT_N)
    particles.extend(add)
    for particle in particles:
        particle.move()
        if not particle.delete:
            new.insert(particle.x, particle.y, particle.r, particle)
    qtree = new


    t2 = time.perf_counter()
    print(f'Frame: {frame}, FPS: {1 / (t2 - t1): .2f}')

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(15,10)

plot = ax1.plot()
ax1.set_aspect("equal")


qtree = QuadTree(0, 0, WIDTH, HEIGHT, QT_N)
for p, c in SETUPS[RUN].items():
    for _ in range(c):
        x, y = random.random() * WIDTH, random.random() * HEIGHT
        qtree.insert(x, y, RADII[p], Particle(p, x, y, int(random.random() * COOLDOWN))) 

history = deque()

NRGraph, = ax2.plot([], [], label="NR", color=COLORS["NR"])
NSGraph, = ax2.plot([], [], label="NS", color=COLORS["NS"])
NPGraph, = ax2.plot([], [], label="NP", color=COLORS["NP"])
MGraph,  = ax2.plot([], [], label="M",  color=COLORS["M"])
PGraph,  = ax2.plot([], [], label="P",  color=COLORS["P"])
BGraph,  = ax2.plot([], [], label="B",  color=COLORS["B"])
SGraph,  = ax2.plot([], [], label="S",  color=COLORS["S"])
VGraph,  = ax2.plot([], [], label="V",  color=COLORS["V"])
IGraph,  = ax2.plot([], [], label="I",  color=COLORS["I"])
NGraph, = ax2.plot([], [], label="N", color=COLORS["N"])

ax2.legend()

data = pd.read_csv("Nottingham Phage Data.csv")
speed = 1
scale = 2900 * SCALE * SCALE / 2.9E8
if RUN == "Dual":
    plt.scatter(data["times"] * speed, (data["EColiAll"] * scale).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(data["times"] * speed, (data["BdAll"] * scale).clip(lower=1), label="P", color=COLORS["P"])
    plt.scatter(data["times"] * speed, (data["PhageAll"] * scale).clip(lower=1), label="V", color=COLORS["V"])
elif RUN == "V":
    plt.scatter(data["times"] * speed, (data["EColiWithPhage"] * scale).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(data["times"] * speed, (data["PhageWithEColiOnly"] * scale).clip(lower=1), label="V", color=COLORS["V"])
elif RUN == "P":
    plt.scatter(data["times"] * speed, (data["EColiWithBd"] * scale).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(data["times"] * speed, (data["BdWithEColiOnly"] * scale).clip(lower=1), label="P", color=COLORS["P"])
elif RUN == "N":
    plt.scatter(data["times"] * speed, (data["EColiOnly"] * scale).clip(lower=1), label="N", color=COLORS["N"])
        

simData = pd.read_csv(f"simdata/{RUN}.csv")[::int(dt / 0.001)]
simData[["N","NR","NS","NP","M","P","B","S","V","I"]] = (simData[["N","NR","NS","NP","M","P","B","S","V","I"]] * scale).clip(lower=1)

SimNGraph, = ax2.plot([], [], label="N", color=(*COLORS["N"], 0.3))
SimNRGraph, = ax2.plot([], [], label="NR", color=(*COLORS["NR"], 0.3))
SimNSGraph, = ax2.plot([], [], label="NS", color=(*COLORS["NS"], 0.3))
SimNPGraph, = ax2.plot([], [], label="NP", color=(*COLORS["NP"], 0.3))
SimMGraph, = ax2.plot([], [], label="M", color=(*COLORS["M"], 0.3))
SimPGraph, = ax2.plot([], [], label="P", color=(*COLORS["P"], 0.3))
SimBGraph, = ax2.plot([], [], label="B", color=(*COLORS["B"], 0.3))
SimSGraph, = ax2.plot([], [], label="S", color=(*COLORS["S"], 0.3))
SimVGraph, = ax2.plot([], [], label="V", color=(*COLORS["V"], 0.3))
SimIGraph, = ax2.plot([], [], label="I", color=(*COLORS["I"], 0.3))

plt.yscale("log")

info = f'{SCALE}x_' + '_'.join([f'{"+".join(list(k))}{",".join(str(r["prob"]) for r in v)}' for k, v in REACTIONS.items()])

def saveData():
    df = pd.DataFrame(history, columns=["t","N","NR","NS","NP","M","P","B","S","V","I"])[1:]
    df.to_csv(f'data_test_{RUN}_{info}.csv')

def video():
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    x = time.perf_counter()
    anim = FuncAnimation(fig, update, frames=FRAMES, interval=1/60, repeat=False)
    writervideo = FFMpegWriter(fps=60)
    anim.save(f'anim_test_{RUN}_{info}.mp4', writer=writervideo)
    y = time.perf_counter()
    print(y - x)
    saveData()

def window():    
    A = anim.FuncAnimation(fig, update, interval=1)
    fig.canvas.mpl_connect('key_press_event', lambda event: saveData() if event.key == ' ' else None)
    plt.show()

video()