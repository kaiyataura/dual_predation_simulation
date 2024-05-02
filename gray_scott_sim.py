# import numpy as np
# import matplotlib.pyplot as plt

# ROWS = 100
# COLS = 100

# FR = 8.6E-6
# m = 6.4E-2

# uN = 0.46
# uP = 0.33
# uV = 3.9E-9

# KMN = 4.2E7
# KNP = 3.2E6

# kP = 1.1
# kV = 4.2
# kD = 4.0E-12
# kM = 7.5E-10

# YNM = 1.2
# YVI = 24
# YIV = 1
# YMP = 3.3E-3
# YMV = 2.1E-2
# YVM = 1
# YPB = 4.17 # fixed
# YBP = 1
# YBN = 1
# YIN = 1

# dt = 0.1


# types = ['NR', 'NS', 'NP', 'M', 'P', 'B', 'S', 'V', 'I']
# type_dict = {k: i for i, k in enumerate(types)}
# num_types = len(types)
# cell_type = np.dtype([(n, float) for n in ['NR', 'NS', 'NP', 'M', 'P', 'B', 'S', 'V', 'I']])

# # color_grid = np.random.rand(ROWS, COLS, num_types)
# grid = np.zeros((ROWS, COLS, num_types))
# grid[:,:,1] = 2.9E8 * np.random.rand(ROWS, COLS)
# grid[:,:,4] = 5E6 * np.random.rand(ROWS, COLS)
# grid[:,:,7] = 5E6 * np.random.rand(ROWS, COLS)
# # grid[np.random.randint(0, ROWS, 10),np.random.randint(0, ROWS, 10), 4] = 5E6 * np.random.rand(10,)
# # grid[np.random.randint(0, ROWS, 10),np.random.randint(0, ROWS, 10), 7] = 5E6 * np.random.rand(10,)
# color_grid = np.zeros((ROWS, COLS, 3), dtype=float)
# # history = {k: [] for k in "tNPVM"}
# history = {k: [] for k in types + ['t']}

# COLORS = {"NR": (0.0, 0.0, 1.0),
#           "NS": (0.0, 1.0, 1.0),
#           "NP": (0.0, 0.5, 1.0),
#           "M":  (0.75, 0.0, 1.0),
#           "P":  (1.0, 0.0, 0.0),
#           "B":  (1.0, 0.5, 0.0),
#           "S":  (1.0, 0.0, 1.0),
#           "V":  (0.0, 1.0, 0.0),
#           "I":  (1.0, 1.0, 0.0)}

# # P -m-> ø                        predator mortality
# # P + NR -uP-> P + B              bacteriovorus attack NR
# # P + NS -uP-> P + B              bacteriovorus attack NS
# # B -kP-> (YPB) P + S + (YMP) M   bacteriovorus maturation
# # S + NS -kD-> NP                 plastic resistance
# # V + NS -uV-> I                  phage attack NS
# # V + NP -uV-> I                  phage attack NP
# # I -kV-> (YVI) V + (YMV) M       infected cell lysis
# # M + NR -uN-> 2 NR               resistant prey growth
# # (1/YNM) M + NS -uN-> 2 NS       sensitive prey growth
# # (1/YNM) M + NP -uN-> 2 NS       growth coupled reversion
# # NS -kM-> NR                     phage resistance

# def T(name): return type_dict[name]

# # reaction phase
# def reaction_phase(grid: np.ndarray, dt: float):
#     # return
#     M = grid[:,:,T("M")]
#     NS = grid[:,:,T("NS")]
#     NP = grid[:,:,T("NP")]
#     NR = grid[:,:,T("NR")]
#     P = grid[:,:,T("P")]
#     B = grid[:,:,T("B")]
#     V = grid[:,:,T("V")]
#     I = grid[:,:,T("I")]
#     S = grid[:,:,T("S")]


#     dM = - uN * M * (NS + NP + NR) / ((KMN + M) * YNM) + YMP * kP * B + YMV * kV * I
#     dNS = (NS + NP) * uN * M / (KMN + M) - P * uP * NS / ((KNP + NS + NP + NR) * YBN) - V * uV * NS / YIN - kD * S * NS - kM * NS
#     dNP = - V * uV * NP / YIN + kD * S * NS
#     dNR = NR * uN * M / (KMN + M) - P * uP * NR / ((KNP + NS + NP + NR) * YBN) + kM * NS
#     dP = kP * B - m * P - P * uP * (NR + NS) / ((KNP + NS + NP + NR) * YBP)
#     dB = - kP * B / YPB + P * uP * (NR + NS) / (KNP + NS + NP + NR)
#     dV = kV * I - V * uV * (NP + NS) / YIV
#     dI = - kV * I / YVI + V * uV * (NP + NS)
#     dS = kP * B / YPB
    
#     d = np.array([dNR, dNS, dNP, dM, dP, dB, dS, dV, dI]).transpose((1,2,0))
#     grid[:,:,:] = (grid + d * dt).clip(min=0)


# # diffusion phase
# def diffusion_phase(grid: np.ndarray, dt):
#     # D = .2
#     # du_dx = np.diff(grid, axis=1)
#     # du_dy = np.diff(grid, axis=0)

#     # grid[:,1:,:] -= D * du_dx
#     # grid[:,:-1,:] += D * du_dx
#     # grid[1:,:,:] -= D * du_dy
#     # grid[:-1,:,:] += D * du_dy
#     # grid[:,:,:] = grid.clip(min=0)

#     d2u_dx2 = np.diff(grid, n=2, axis=1, prepend=grid[:,[0]], append=grid[:,[-1]])
#     d2u_dy2 = np.diff(grid, n=2, axis=0, prepend=grid[[0],:], append=grid[[-1],:])
#     diffusion_rates = np.array([.1, .1, .1, .2, .1, .1, .2, .2, .1])
#     grid[:,:,:] = (grid + diffusion_rates * (d2u_dx2 + d2u_dy2) * dt).clip(min=0)

# def plot_grid(grid: np.ndarray, color_grid: np.ndarray):
#     colors = {"NR": (0.0, 0.0, 1.0),
#               "NS": (0.0, 1.0, 1.0),
#               "NP": (0.0, 0.5, 1.0),
#               "M":  (1.0, 0.0, 1.0),
#               "P":  (1.0, 0.0, 0.0),
#               "B":  (1.0, 0.5, 0.0),
#               "S":  (1.0, 0.0, 0.5),
#               "V":  (0.0, 1.0, 0.0),
#               "I":  (1.0, 1.0, 0.0)}

#     temp = grid.copy()
#     temp[:,:,T("V")] /= 20
#     color_grid[:,:,0] = np.dot(np.nan_to_num(temp / np.sum(temp, axis=2, keepdims=True), nan=1), [c[0] for c in colors.values()]).clip(min=0, max=1)
#     color_grid[:,:,1] = np.dot(np.nan_to_num(temp / np.sum(temp, axis=2, keepdims=True), nan=1), [c[1] for c in colors.values()]).clip(min=0, max=1)
#     color_grid[:,:,2] = np.dot(np.nan_to_num(temp / np.sum(temp, axis=2, keepdims=True), nan=1), [c[2] for c in colors.values()]).clip(min=0, max=1)

# def save_history(history, grid, dt):
#     # history['t'].append(0 if dt is None else history['t'][-1] + dt)
#     # history['N'].append(np.sum(grid[:,:,[I('NR'), I('NS'), I('NP')]]))
#     # history['P'].append(np.sum(grid[:,:,I('P')]))
#     # history['V'].append(np.sum(grid[:,:,I('V')]))
#     # history['M'].append(np.sum(grid[:,:,I('M')]))
#     history['t'].append(0 if dt is None else history['t'][-1] + dt)
#     history['NR'].append(max(np.sum(grid[:,:,T('NR')]), 1))
#     history['NS'].append(max(np.sum(grid[:,:,T('NS')]), 1))
#     history['NP'].append(max(np.sum(grid[:,:,T('NP')]), 1))
#     history['M'].append(max(np.sum(grid[:,:,T('M')]), 1))
#     history['P'].append(max(np.sum(grid[:,:,T('P')]), 1))
#     history['B'].append(max(np.sum(grid[:,:,T('B')]), 1))
#     history['S'].append(max(np.sum(grid[:,:,T('S')]), 1))
#     history['V'].append(max(np.sum(grid[:,:,T('V')]), 1))
#     history['I'].append(max(np.sum(grid[:,:,T('I')]), 1))



# def tick(frame):
#     reaction_phase(grid, dt)
#     diffusion_phase(grid, dt)
#     grid[grid < 1E-5] = 0
#     plot_grid(grid, color_grid)
#     save_history(history, grid, dt)

#     print(grid.max(axis=(0,1)))
#     NRPlot.set_data(1 - ((1 - np.array(COLORS["NR"])) * (grid[:,:,[type_dict["NR"]]] / grid[:,:,[type_dict["NR"]]].max(initial=1E-10))).clip(min=0, max=1))
#     NSPlot.set_data(1 - ((1 - np.array(COLORS["NS"])) * (grid[:,:,[type_dict["NS"]]] / grid[:,:,[type_dict["NS"]]].max(initial=1E-10))).clip(min=0, max=1))
#     NPPlot.set_data(1 - ((1 - np.array(COLORS["NP"])) * (grid[:,:,[type_dict["NP"]]] / grid[:,:,[type_dict["NP"]]].max(initial=1E-10))).clip(min=0, max=1))
#     MPlot.set_data(1 - ((1 - np.array(COLORS["M"])) * (grid[:,:,[type_dict["M"]]] / grid[:,:,[type_dict["M"]]].max(initial=1E-10))).clip(min=0, max=1))
#     PPlot.set_data(1 - ((1 - np.array(COLORS["P"])) * (grid[:,:,[type_dict["P"]]] / grid[:,:,[type_dict["P"]]].max(initial=1E-10))).clip(min=0, max=1))
#     BPlot.set_data(1 - ((1 - np.array(COLORS["B"])) * (grid[:,:,[type_dict["B"]]] / grid[:,:,[type_dict["B"]]].max(initial=1E-10))).clip(min=0, max=1))
#     SPlot.set_data(1 - ((1 - np.array(COLORS["S"])) * (grid[:,:,[type_dict["S"]]] / grid[:,:,[type_dict["S"]]].max(initial=1E-10))).clip(min=0, max=1))
#     VPlot.set_data(1 - ((1 - np.array(COLORS["V"])) * (grid[:,:,[type_dict["V"]]] / grid[:,:,[type_dict["V"]]].max(initial=1E-10))).clip(min=0, max=1))
#     IPlot.set_data(1 - ((1 - np.array(COLORS["I"])) * (grid[:,:,[type_dict["I"]]] / grid[:,:,[type_dict["I"]]].max(initial=1E-10))).clip(min=0, max=1))
    
#     im.set_data(color_grid)
    
#     # NGraph.set_xdata(history["t"])
#     # PGraph.set_xdata(history["t"])
#     # VGraph.set_xdata(history["t"])
#     # MGraph.set_xdata(history["t"])

#     # NGraph.set_ydata(history["N"])
#     # PGraph.set_ydata(history["P"])
#     # VGraph.set_ydata(history["V"])
#     # MGraph.set_ydata(history["M"])

#     NRGraph.set_xdata(history["t"])
#     NSGraph.set_xdata(history["t"])
#     NPGraph.set_xdata(history["t"])
#     MGraph.set_xdata(history["t"])
#     PGraph.set_xdata(history["t"])
#     BGraph.set_xdata(history["t"])
#     SGraph.set_xdata(history["t"])
#     VGraph.set_xdata(history["t"])
#     IGraph.set_xdata(history["t"])

#     NRGraph.set_ydata(history["NR"])
#     NSGraph.set_ydata(history["NS"])
#     NPGraph.set_ydata(history["NP"])
#     MGraph.set_ydata(history["M"])
#     PGraph.set_ydata(history["P"])
#     BGraph.set_ydata(history["B"])
#     SGraph.set_ydata(history["S"])
#     VGraph.set_ydata(history["V"])
#     IGraph.set_ydata(history["I"])

#     ax13.relim()
#     ax13.autoscale()
#     plt.draw()


# save_history(history, grid, None)

# fig, ((ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23)) = plt.subplots(3, 4)

# # NGraph, = ax13.plot(history["t"], history["N"], label="N", color="blue")
# # PGraph, = ax13.plot(history["t"], history["P"], label="P", color="red")
# # VGraph, = ax13.plot(history["t"], history["V"], label="V", color="green")
# # MGraph, = ax13.plot(history["t"], history["M"], label="M", color="magenta")

# im = ax03.imshow(color_grid)

# NRPlot = ax00.imshow(np.ones((ROWS, COLS, 3), dtype=float))
# NSPlot = ax01.imshow(np.ones((ROWS, COLS, 3), dtype=float))
# NPPlot = ax02.imshow(np.ones((ROWS, COLS, 3), dtype=float))
# MPlot  = ax10.imshow(np.ones((ROWS, COLS, 3), dtype=float))
# PPlot  = ax11.imshow(np.ones((ROWS, COLS, 3), dtype=float))
# BPlot  = ax12.imshow(np.ones((ROWS, COLS, 3), dtype=float))
# SPlot  = ax20.imshow(np.ones((ROWS, COLS, 3), dtype=float))
# VPlot  = ax21.imshow(np.ones((ROWS, COLS, 3), dtype=float))
# IPlot  = ax22.imshow(np.ones((ROWS, COLS, 3), dtype=float))

# NRGraph, = ax13.plot(history["t"], history["NR"], label="NR", color=COLORS["NR"])
# NSGraph, = ax13.plot(history["t"], history["NS"], label="NS", color=COLORS["NS"])
# NPGraph, = ax13.plot(history["t"], history["NP"], label="NP", color=COLORS["NP"])
# MGraph,  = ax13.plot(history["t"], history["M"],  label="M",  color=COLORS["M"])
# PGraph,  = ax13.plot(history["t"], history["P"],  label="P",  color=COLORS["P"])
# BGraph,  = ax13.plot(history["t"], history["B"],  label="B",  color=COLORS["B"])
# SGraph,  = ax13.plot(history["t"], history["S"],  label="S",  color=COLORS["S"])
# VGraph,  = ax13.plot(history["t"], history["V"],  label="V",  color=COLORS["V"])
# IGraph,  = ax13.plot(history["t"], history["I"],  label="I",  color=COLORS["I"])

# # ax13.legend()
# ax13.set_yscale("log")

# fig.canvas.mpl_connect('key_press_event', lambda event: tick(0) if event.key == ' ' else None)

# # plt.show()


# from matplotlib.animation import FuncAnimation, FFMpegWriter
# anim = FuncAnimation(fig, tick, frames=1000, interval=1/60, repeat=False)
# writervideo = FFMpegWriter(fps=60)
# anim.save(f'gray_scott_anim_2.mp4', writer=writervideo)
 



# while True:
#     tick(grid, color_grid, history, dt)
#     plt.pause(0.0001)





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd
import noise


ROWS = 100
COLS = 100

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

RUN = 'P'

dt = 0.01

types = ['NR', 'NS', 'NP', 'M', 'P', 'B', 'S', 'V', 'I']
type_dict = {k: i for i, k in enumerate(types)}
num_types = len(types)
cell_type = np.dtype([(n, float) for n in ['NR', 'NS', 'NP', 'M', 'P', 'B', 'S', 'V', 'I']])

# grid = np.random.rand(ROWS, COLS, num_types)
grid = np.zeros((ROWS, COLS, num_types))

for i in range(100):
    for j in range(100):
        grid[i,j,1] = 2.9E8 * (0.5 + 0.5 * noise.pnoise3(x=i/10, y=j/10, z=0.5, octaves=3, repeatx=10, repeaty=10, base=0))
        if RUN == 'Dual' or RUN == 'P': grid[i,j,4] = 5E6 * (0.5 + 0.5 * noise.pnoise3(x=i/10, y=j/10, z=1.5, octaves=3, repeatx=10, repeaty=10, base=0))
        if RUN == 'Dual' or RUN == 'V': grid[i,j,7] = 5E6 * (0.5 + 0.5 * noise.pnoise3(x=i/10, y=j/10, z=2.5, octaves=3, repeatx=10, repeaty=10, base=0))

num = 1
# grid[np.random.randint(0, ROWS, num),np.random.randint(0, ROWS, num), 4] = 5E6 * ROWS * COLS / num
# grid[np.random.randint(0, ROWS, num),np.random.randint(0, ROWS, num), 7] = 5E6 * ROWS * COLS / num
color_grid = np.zeros((ROWS, COLS, 3), dtype=float)
# history = {k: [] for k in "tNPVM"}
history = {k: [] for k in ['t', 'N'] + types}


# P -m-> ø                        predator mortality
# P + NR -uP-> P + B              bacteriovorus attack NR
# P + NS -uP-> P + B              bacteriovorus attack NS
# B -kP-> (YPB) P + S + (YMP) M   bacteriovorus maturation
# S + NS -kD-> NP                 plastic resistance
# V + NS -uV-> I                  phage attack NS
# V + NP -uV-> I                  phage attack NP
# I -kV-> (YVI) V + (YMV) M       infected cell lysis
# M + NR -uN-> 2 NR               resistant prey growth
# (1/YNM) M + NS -uN-> 2 NS       sensitive prey growth
# (1/YNM) M + NP -uN-> 2 NS       growth coupled reversion
# NS -kM-> NR                     phage resistance

def T(name): return type_dict[name]

# reaction phase
def reaction_phase(grid: np.ndarray, dt: float):
    # return
    M = grid[:,:,T("M")]
    NS = grid[:,:,T("NS")]
    NP = grid[:,:,T("NP")]
    NR = grid[:,:,T("NR")]
    P = grid[:,:,T("P")]
    B = grid[:,:,T("B")]
    V = grid[:,:,T("V")]
    I = grid[:,:,T("I")]
    S = grid[:,:,T("S")]


    dM = - uN * M * (NS + NP + NR) / ((KMN + M) * YNM) + YMP * kP * B + YMV * kV * I
    dNS = (NS + NP) * uN * M / (KMN + M) - P * uP * NS / ((KNP + NS + NP + NR) * YBN) - V * uV * NS / YIN - kD * S * NS - kM * NS
    dNP = - V * uV * NP / YIN + kD * S * NS
    dNR = NR * uN * M / (KMN + M) - P * uP * NR / ((KNP + NS + NP + NR) * YBN) + kM * NS
    dP = kP * B - m * P - P * uP * (NR + NS) / ((KNP + NS + NP + NR) * YBP)
    dB = - kP * B / YPB + P * uP * (NR + NS) / (KNP + NS + NP + NR)
    dV = kV * I - V * uV * (NP + NS) / YIV
    dI = - kV * I / YVI + V * uV * (NP + NS)
    dS = kP * B / YPB
    
    d = np.array([dNR, dNS, dNP, dM, dP, dB, dS, dV, dI]).transpose((1,2,0))
    grid[:,:,:] = (grid + d * dt).clip(min=0)


# diffusion phase
def diffusion_phase(grid: np.ndarray, dt):
    # D = .2
    # du_dx = np.diff(grid, axis=1)
    # du_dy = np.diff(grid, axis=0)

    # grid[:,1:,:] -= D * du_dx
    # grid[:,:-1,:] += D * du_dx
    # grid[1:,:,:] -= D * du_dy
    # grid[:-1,:,:] += D * du_dy
    # grid[:,:,:] = grid.clip(min=0)

    d2u_dx2 = np.diff(grid, n=2, axis=1, prepend=0, append=0)
    d2u_dy2 = np.diff(grid, n=2, axis=0, prepend=0, append=0)
    diffusion_rates = 1
    grid[:,:,:] = (grid + diffusion_rates * (d2u_dx2 + d2u_dy2) * dt).clip(min=0)

def plot_grid(grid: np.ndarray, color_grid: np.ndarray):
    colors = {"NR": (0.0, 0.0, 1.0),
              "NS": (0.0, 1.0, 1.0),
              "NP": (0.0, 0.5, 1.0),
              "M":  (1.0, 0.0, 1.0),
              "P":  (1.0, 0.0, 0.0),
              "B":  (1.0, 0.5, 0.0),
              "S":  (1.0, 0.0, 0.5),
              "V":  (0.0, 1.0, 0.0),
              "I":  (1.0, 1.0, 0.0)}

    grid_copy = grid.copy()
    grid_copy = grid
    # grid_copy[:,:,T("V")] /= 10
    color_grid[:,:,0] = np.dot(np.nan_to_num(grid_copy / np.sum(grid_copy, axis=2, keepdims=True), nan=1), [c[0] for c in colors.values()]).clip(min=0, max=1)
    color_grid[:,:,1] = np.dot(np.nan_to_num(grid_copy / np.sum(grid_copy, axis=2, keepdims=True), nan=1), [c[1] for c in colors.values()]).clip(min=0, max=1)
    color_grid[:,:,2] = np.dot(np.nan_to_num(grid_copy / np.sum(grid_copy, axis=2, keepdims=True), nan=1), [c[2] for c in colors.values()]).clip(min=0, max=1)

def save_history(history, grid, frame, dt):
    # history['t'].append(0 if dt is None else history['t'][-1] + dt)
    # history['N'].append(np.sum(grid[:,:,[I('NR'), I('NS'), I('NP')]]))
    # history['P'].append(np.sum(grid[:,:,I('P')]))
    # history['V'].append(np.sum(grid[:,:,I('V')]))
    # history['M'].append(np.sum(grid[:,:,I('M')]))
    history['t'].append(frame * dt)
    history['NR'].append(np.sum(grid[:,:,T('NR')]) / (ROWS * COLS))
    history['NS'].append(np.sum(grid[:,:,T('NS')]) / (ROWS * COLS))
    history['NP'].append(np.sum(grid[:,:,T('NP')]) / (ROWS * COLS))
    history['N'].append(history['NR'][-1] + history['NS'][-1] + history['NP'][-1])
    history['M'].append(np.sum(grid[:,:,T('M')]) / (ROWS * COLS))
    history['P'].append(np.sum(grid[:,:,T('P')]) / (ROWS * COLS))
    history['B'].append(np.sum(grid[:,:,T('B')]) / (ROWS * COLS))
    history['S'].append(np.sum(grid[:,:,T('S')]) / (ROWS * COLS))
    history['V'].append(np.sum(grid[:,:,T('V')]) / (ROWS * COLS))
    history['I'].append(np.sum(grid[:,:,T('I')]) / (ROWS * COLS))


save_history(history, grid, 0, dt)

def tick(frame):
    frame += 1
    print(f'Frame: {frame}')
    reaction_phase(grid, dt)
    diffusion_phase(grid, dt)
    plot_grid(grid, color_grid)
    save_history(history, grid, frame, dt)
    im.set_data(color_grid)
    
    # NGraph.set_xdata(history["t"])
    # PGraph.set_xdata(history["t"])
    # VGraph.set_xdata(history["t"])
    # MGraph.set_xdata(history["t"])

    # NGraph.set_ydata(history["N"])
    # PGraph.set_ydata(history["P"])
    # VGraph.set_ydata(history["V"])
    # MGraph.set_ydata(history["M"])

    NRGraph.set_xdata(history["t"])
    NSGraph.set_xdata(history["t"])
    NPGraph.set_xdata(history["t"])
    MGraph.set_xdata(history["t"])
    PGraph.set_xdata(history["t"])
    BGraph.set_xdata(history["t"])
    SGraph.set_xdata(history["t"])
    VGraph.set_xdata(history["t"])
    IGraph.set_xdata(history["t"])
    NGraph.set_xdata(history["t"])

    NRGraph.set_ydata([max(x, 1) for x in history["NR"]])
    NSGraph.set_ydata([max(x, 1) for x in history["NS"]])
    NPGraph.set_ydata([max(x, 1) for x in history["NP"]])
    MGraph.set_ydata([max(x, 1) for x in history["M"]])
    PGraph.set_ydata([max(x, 1) for x in history["P"]])
    BGraph.set_ydata([max(x, 1) for x in history["B"]])
    SGraph.set_ydata([max(x, 1) for x in history["S"]])
    VGraph.set_ydata([max(x, 1) for x in history["V"]])
    IGraph.set_ydata([max(x, 1) for x in history["I"]])
    NGraph.set_ydata([max(x, 1) for x in history["N"]])

    ax2.relim()
    ax2.autoscale()
    plt.draw()


# save_history(history, grid, None)

fig, (ax1, ax2) = plt.subplots(1, 2)
im = ax1.imshow(color_grid)

# NGraph, = ax2.plot(history["t"], history["N"], label="N", color="blue")
# PGraph, = ax2.plot(history["t"], history["P"], label="P", color="red")
# VGraph, = ax2.plot(history["t"], history["V"], label="V", color="green")
# MGraph, = ax2.plot(history["t"], history["M"], label="M", color="magenta")

NRGraph, = ax2.plot(history["t"], history["NR"], label="NR", color=COLORS["NR"])
NSGraph, = ax2.plot(history["t"], history["NS"], label="NS", color=COLORS["NS"])
NPGraph, = ax2.plot(history["t"], history["NP"], label="NP", color=COLORS["NP"])
MGraph,  = ax2.plot(history["t"], history["M"],  label="M",  color=COLORS["M"])
PGraph,  = ax2.plot(history["t"], history["P"],  label="P",  color=COLORS["P"])
BGraph,  = ax2.plot(history["t"], history["B"],  label="B",  color=COLORS["B"])
SGraph,  = ax2.plot(history["t"], history["S"],  label="S",  color=COLORS["S"])
VGraph,  = ax2.plot(history["t"], history["V"],  label="V",  color=COLORS["V"])
IGraph,  = ax2.plot(history["t"], history["I"],  label="I",  color=COLORS["I"])
NGraph,  = ax2.plot(history["t"], history["N"],  label="N",  color=COLORS["N"])

ax2.legend()
plt.yscale("log")


fig.canvas.mpl_connect('key_press_event', lambda event: tick(0) if event.key == ' ' else None)
fig.set_size_inches(15,10)

exp = pd.read_csv('Nottingham Phage Data.csv')

if RUN == "Dual":
    plt.scatter(exp["times"], (exp["EColiAll"]).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(exp["times"], (exp["BdAll"]).clip(lower=1), label="P", color=COLORS["P"])
    plt.scatter(exp["times"], (exp["PhageAll"]).clip(lower=1), label="V", color=COLORS["V"])
elif RUN == "V":
    plt.scatter(exp["times"], (exp["EColiWithPhage"]).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(exp["times"], (exp["PhageWithEColiOnly"]).clip(lower=1), label="V", color=COLORS["V"])
elif RUN == "P":
    plt.scatter(exp["times"], (exp["EColiWithBd"]).clip(lower=1), label="N", color=COLORS["N"])
    plt.scatter(exp["times"], (exp["BdWithEColiOnly"]).clip(lower=1), label="P", color=COLORS["P"])
elif RUN == "N":
    plt.scatter(exp["times"], (exp["EColiOnly"]).clip(lower=1), label="N", color=COLORS["N"])
      

A = anim.FuncAnimation(fig, tick, interval=1)


plt.show()

# from matplotlib.animation import FuncAnimation, FFMpegWriter
# anim = FuncAnimation(fig, tick, frames=4800, interval=1/60, init_func=(lambda: None))
# writervideo = FFMpegWriter(fps=60)
# anim.save(f'anim_{RUN}_gray_scott.mp4', writer=writervideo)
# pd.DataFrame(history).to_csv(f'data_{RUN}_gray_scott.csv')
 

# while True:
#     tick(grid, color_grid, history, dt)
#     plt.pause(0.01)



