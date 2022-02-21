# Include modules
import src.pathmap as mp
import src.diffeq as dq
import src.sw_curve as sw

# Include default libraries
import numpy as np
# FAQ: Should we use a dynamic list here for x and y's?
# Pro: More flexible if we use a different cutoff condition than steps
# Cons: Performance could suffer.
class path():
    def __init__(self, label, gen, x_limits, y_limits, resolution, start = 0, color='black'):
        self.x = []
        self.y_i = []
        self.y_j = []
        self.mass = []
        self.start = start
        self.label = label
        self.gen = gen
        self.pathmap = mp.pathmap(x_limits, y_limits, resolution)
        self.color = '#247ad6'
        self.intersection_tracker = []
        self.active = True

    def evolve(self, dt, f, theta, expo):
        val_x = self.x[-1]
        val_y_i = self.y_i[-1]
        val_y_j = self.y_j[-1]
        dy = dq.rk4(f, [val_x, val_y_i, val_y_j], dt, theta, expo=expo)
        if dy[0] == 0:
             self.active = False
             return
        self.x.append(val_x + dy[0])
        self.y_i.append(val_y_i + dy[1])
        self.y_j.append(val_y_j + dy[2])
        self.mass.append(self.mass[-1] + abs(dy[0]))

    def update_map(self, step):
        self.pathmap.draw_line(self.x[step - 1], self.x[step], step)

    