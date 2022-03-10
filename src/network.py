#
# Network class
#

# Include libraries
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Include projectr files
import src.sw_curve as sw
import src.diffeq as dq
# import src.moebius as mb
import src.pathmap as pm
import src.path as pth
import src.intersection as scn

import time

colors = ['red','orange','green','purple', 'red', 'red', 'red']

# Probably this should go somewhere else...
def moebius(A, z):
    return (A[0, 0] * z + A[0, 1]) / (A[1, 0] * z + A[1, 1])


class network():
    def __init__(self, H, dt, theta, cutoff, expo=False, x_limits=(-5, 5), y_limits=(-3, 3), resolution=(2000, 2000)):
        self.curve = sw.sw_curve(H)
        # FAQ: Maybe remove that?
        self.expo = expo
        self.dt = dt
        self.theta = theta
        self.x_lim = x_limits
        self.y_lim = y_limits
        self.resolution = resolution
        self.cutoff = cutoff
        self.split = 1000

        # For a different termination condition
        # a dynamic list might work better here
        # -> Implemented in version 4
        self.time = []
        self.paths = []
        self.active_paths = []
        self.intersections = []
        self.all_intersections = []
        self.counter = 0
        self.pts = list(self.curve.branch_points) + list(self.curve.sing_points)
        

    def start_paths(self):
        self.time.append(0)
        self.time.append(self.dt)
        start_x = []
        start_y = []
        for j in range(len(self.curve.branch_points)):
            rts = self.curve.get_fiber(self.curve.branch_points[j])

            for i in range(len(rts) - 1):
                if abs(np.prod
                        ([rts[i] - rts[u] for u in range(i + 1, len(rts))])
                        ) < 0.0001:
                    start_x.append(self.curve.branch_points[j])
                    start_y.append(complex(rts[i]))
                    break
            self.time[1] = self.dt


        for j in range(len(self.curve.branch_points)):
            kappa = -1 / 2 * self.curve.d2Hy2(
                self.curve.branch_points[j], start_y[j]) / \
                self.curve.dHx(start_x[j], start_y[j])
            if self.expo:
                dx = ((3 / 4 * np.sqrt(kappa) * start_y[j] *
                       start_x[j] *
                       np.exp(1j * self.theta) * self.dt)**2)**(1 / 3)
            else:
                dx = ((3 / 4 * np.sqrt(kappa) * start_x[j]
                       * np.exp(1j * self.theta) * self.dt)**2)**(1 / 3)
        
            for k in range(3):
                # How can we find out if we have +- or -+.
                self.paths.append(pth.path("+-", 0, self.x_lim, self.y_lim, self.resolution))
                dx_k = dx*np.exp(2 * np.pi * k * 1j / 3)
                self.paths[-1].x.append(start_x[j])
                self.paths[-1].x.append(start_x[j] + dx_k)
                self.paths[-1].mass.append(0)
                self.paths[-1].mass.append(abs(dx_k))
                if self.expo:
                    self.paths[-1].y_i.append(np.log(start_y[j]))
                    self.paths[-1].y_j.append(self.paths[-1].y_i[0])
                    self.paths[-1].y_i.append(self.paths[-1].y_i[0] \
                            +  ((3 / 4 * np.sqrt(kappa) * self.paths[-1].x[0]
                                * np.exp(1j * self.theta) * self.dt))**(1 / 3)\
                            * np.exp(2 * np.pi * k * 1j / 3) / np.sqrt(kappa))
                    self.paths[-1].y_j.append(self.paths[-1].y_i[0] \
                            - ((3 / 4 * np.sqrt(kappa) * self.paths[-1].x[0]
                                * np.exp(1j * self.theta) * self.dt))**(1 / 3)\
                            * np.exp(2 * np.pi * k * 1j / 3) / np.sqrt(kappa))
                else:
                    self.paths[-1].y_i.append(start_y[j])
                    self.paths[-1].y_j.append(start_y[j])
                    self.paths[-1].y_i.append(self.paths[-1].y_i[0] \
                        + np.sqrt(dx_k) / np.sqrt(kappa))
                    self.paths[-1].y_j.append(self.paths[-1].y_j[0] \
                        - np.sqrt(dx_k) / np.sqrt(kappa))
        


        # What parameter should we take here?
        for j in range(len(self.paths)):
            for i in range(2, self.split): 
                self.paths[j].evolve(self.dt, self.curve.sw_differential, self.theta, self.expo)
                self.paths[j].update_map(i)
        self.time.append(0)
        self.time.append(self.time[-1] + self.dt)
        self.active_paths = [0,1,2]
        
        # This seems to be a good choice, but of course this is very ad hoc
        self.dt = 1e-3

    def evolve(self, steps=0):
        dt = self.dt

        # 3 is too much here!
        gen = 2
        for k in range(gen + 1):
            if k > 0:
                self.split = 1        
            for j in self.active_paths:
                i = 0
                t1 = 0
                t2 = 0
                while self.paths[j].mass[-1] < self.cutoff and self.paths[j].active and (i < steps or steps == 0):
                    if j == 0:
                        self.time.append(self.time[-1] + self.dt)
                    tic1 = time.time()
                    self.paths[j].evolve(dt, self.curve.sw_differential, self.theta, self.expo)
                    toc1 = time.time()
                    t1 += toc1 - tic1
                    tic2 = time.time()
                    self.paths[j].update_map(i)
                    toc2 = time.time()
                    t2 += toc2 - tic2
                    i += 1
                    if i > 50000:
                        print("Terminated by steps")
                        break
                print("path:", j)
                print("evolve:", t1)
                print("update:", t2)
                tic3 = time.time()
                for t in range(j + 1):
                    self.check_intersection(j, t)
                toc3 = time.time()
                t3 = toc3 - tic3
                print("check:", t3)
            tic4 = time.time()
            if k < gen:
                self.create_paths()
            toc4 = time.time()
            t4 = toc4 - tic4
            print("create:", t4)


    
    # Here I still need to implement linear interpolation between the x points of the paths
    # It should be possible to more or less copy this from the pathmap drawline method
    def check_intersection(self, j, t):
        # self.intersections = []
        # Move this to pathmap
        
        b_points = []
        for b in self.pts:
            p,q = self.paths[t].pathmap.number_to_coordinates(b)
            b_points.append((p,q))
        for i in range(self.split, len(self.paths[j].x)):
            c1 = self.paths[j].pathmap.number_to_coordinates(self.paths[j].x[i- 1])
            c2 = self.paths[j].pathmap.number_to_coordinates(self.paths[j].x[i])
            if c1 == c2:
                if len(self.intersections) != 0 and self.intersections[-1].steps[1] == i - 1 and (c1[0], c1[1]) in self.intersections[-1].coordinates:
                    self.intersections[-1].steps[1] = i
                continue
            s = abs(c2[0] - c1[0]) + abs(c2[1] - c1[1]) + 1
            m = None
            n = None
            for d in range(2 * s):
                new_m = int((c1[0] * d) / (2 * s - 1) + c2[0] * (1 - d / (2 * s - 1)))
                new_n = int((c1[1] * d) / (2 * s - 1) + c2[1] * (1 - d / (2 * s - 1)))
                at_branch = False
                for b in b_points:
                    if abs(new_m  - b[0]) + abs(new_n - b[1]) < 10:
                        at_branch = True
                if at_branch:
                    continue
                if new_m != m or new_n != n:
                    m = new_m
                    n = new_n
                    if (m >= 0 and m < self.paths[t].pathmap.resolution[0]) and (n >= 0 and n < self.paths[t].pathmap.resolution[1]):
                        if len(self.paths[t].pathmap.data[m,n]) != 0:
                            if len(self.intersections) == 0 or self.intersections[-1].target_map != t or self.intersections[-1].path_index != j or (self.intersections[-1].steps[1] != i - 1 and self.intersections[-1].steps[1] != i):
                                if t == j:
                                    for t_range in self.paths[t].pathmap.data[m,n]:
                                        if i > t_range[1] + 2*s:
                                            self.intersections.append(scn.intersection(i, j, t, (m,n)))
                                            break
                                else:
                                    self.intersections.append(scn.intersection(i, j, t, (m,n)))

                            else:
                                if len(self.intersections[-1].coordinates) > 20:
                                    print("Pop!")
                                    print(j, t)
                                    self.intersections.pop(-1)
                                    if t != j:
                                        return
                                    else:
                                        break
                                if j == t:
                                    for t_range in self.paths[t].pathmap.data[m,n]:
                                        if i > t_range[1] + 1:
                                            self.intersections[-1].steps[1] = i
                                        if (m,n) not in self.intersections[-1].coordinates:
                                                self.intersections[-1].coordinates.append((m,n))
                                        break
                                else:
                                    self.intersections[-1].steps[1] = i
                                    if (m,n) not in self.intersections[-1].coordinates:
                                        self.intersections[-1].coordinates.append((m,n))


    def create_paths(self):
        print("In create path!")
        for s in range(len(self.intersections)):
            self.all_intersections.append(self.intersections[s])

        print([self.intersections[i].coordinates[0] for i in range(len(self.intersections))])
        self.active_paths = []
        for i in range(len(self.intersections)):
            t1_range = [self.intersections[i].steps[0] - 1, self.intersections[i].steps[1] + 1]
            target_map = self.intersections[i].target_map
            path_index = self.intersections[i].path_index
            # This is probably not ideal.
            t_cand = []
            for c in self.intersections[i].coordinates:
                for d in self.paths[target_map].pathmap.data[c][0]:
                    t_cand.append(d)
            # I definetly have to change this
            t2_range = [min(t_cand), max(t_cand)]
            intersection_data = self.get_intersection(path_index, target_map, t1_range, t2_range)
            if intersection_data != None:
                pt = intersection_data[0]
                print(pt)
                self.pts.append(pt)
                print(np.log(self.curve.get_fiber(pt)))
                for c in self.intersections[i].coordinates:
                    print(self.paths[target_map].pathmap.coordinates_to_number(c))
                Y1 = [(self.paths[path_index].y_i[intersection_data[1]] + self.paths[path_index].y_i[intersection_data[1] + 1]) / 2, 
                      (self.paths[path_index].y_j[intersection_data[1]] + self.paths[path_index].y_j[intersection_data[1] + 1]) / 2,
                      (self.paths[target_map].y_i[intersection_data[2]] + self.paths[target_map].y_i[intersection_data[2] + 1]) / 2,
                      (self.paths[target_map].y_j[intersection_data[2]] + self.paths[target_map].y_j[intersection_data[2] + 1]) / 2] 
                Y2 = np.log(self.curve.match_fiber_point(pt, np.exp(Y1)))
                sheet = [round(((Y1[u] - Y2[u]).imag)/(2*np.pi)) for u in range(4)]
                Y = [Y2[u] + 2 * np.pi * 1j * sheet[u] for u in range(4)]
                print(self.intersections[i].coordinates)
                print(Y1)
                print(Y)
                for l in range(2):
                    for m in range(2):
                        if np.exp(Y[l]) == np.exp(Y[m + 2]):
                            print((Y[l] - Y[m + 2]) - (Y[(l + 1) % 2] - Y[((m + 1) % 2) + 2]))
                            if (Y[l] - Y[m + 2]) - (Y[(l + 1) % 2] - Y[((m + 1) % 2) + 2]) != 0:
                                self.paths.append(pth.path("+-", max(self.paths[path_index].gen, self.paths[target_map].gen) + 1, self.x_lim, self.y_lim, self.resolution))
                                self.paths[-1].color = colors[self.paths[-1].gen]
                                self.pts.append(pt)
                                self.active_paths.append(len(self.paths) - 1)
                                print("New path created!")
                                if Y[(l + 1) % 2] - Y[(m + 1) % 2] == 0:
                                    self.paths[-1].label = "+-{}".format(((Y[l] - Y[m]).imag)/(2 * np.pi))
                                else:
                                    self.paths[-1].label = "++/--{}".format(((Y[l] - Y[m]).imag)/(2 * np.pi))

                                self.paths[-1].x.append(pt)
                                self.paths[-1].y_i.append(Y[((l + 1) % 2)] + (Y[l] - Y[m + 2]))
                                self.paths[-1].y_j.append(Y[((m + 1) % 2) + 2])
                                self.paths[-1].mass.append(0)

        self.intersections = []
        print("Out create path!")

    def get_intersection(self, path_index, target_map, t1_range, t2_range):
        for t1 in range(t1_range[0], t1_range[1] - 1):
            for t2 in range(t2_range[0], t2_range[1] - 1):
                A0 = np.array([self.paths[path_index].x[t1].real, self.paths[path_index].x[t1].imag])
                A1 = np.array([self.paths[path_index].x[t1 + 1].real, self.paths[path_index].x[t1 + 1].imag])
                B0 = np.array([self.paths[target_map].x[t2].real, self.paths[target_map].x[t2].imag])
                B1 = np.array([self.paths[target_map].x[t2 + 1].real, self.paths[target_map].x[t2 + 1].imag])

                v = A1 - A0
                w = B1 - B0

                A = np.array([v, -w])
                if(np.linalg.det(A)) != 0:
                    A_inv = np.linalg.inv(A)
                    t = np.dot((B0 - A0),A_inv)
                    if (t[0] >= 0 and t[0] <= 1) and t[1] >= 0 and t[1] <= 1:
                        return [np.dot(A0 + t[0]*v, np.array([[1], [1j]]))[0], t1, t2]
        return None

    def plot_network(self, steps=100000, paths=[], fix_axis=True,
                     filename="Network.png"):
     

        fig = plt.figure(figsize=(15, 10), dpi=200)

        if fix_axis:
            plt.xlim(self.x_lim)
            plt.ylim(self.y_lim)

        if len(paths) == 0:
            paths = np.arange(len(self.paths)) 
        A = np.array([[1, 0], [-1, 1/4]])
        for j in paths:
            X = [self.paths[j].x[i].real for i in range(min(steps, len(self.paths[j].x)))]
            Y = [self.paths[j].x[i].imag for i in range(min(steps, len(self.paths[j].x)))]
            plt.plot(X, Y, '-', color=self.paths[j].color)
        plt.plot(self.curve.branch_points.real, self.curve.branch_points.imag,
                 'x', color='orange')
        plt.plot(self.curve.sing_points.real, self.curve.sing_points.imag,
                 'b.')
        # plt.plot([self.pts[i].real for i in range(len(self.pts))], [self.pts[i].imag for i in range(len(self.pts))], 'r.')
        fig.savefig("./graphics/" + filename + ".png", dpi=200)

        fig = plt.figure(figsize=(15, 10), dpi=200)
        for j in paths:
            x_tf = [moebius(A, self.paths[j].x[i]) for i in range(min(steps, len(self.paths[j].x)))]
            X = [x_tf[i].real for i in range(len(x_tf))]
            Y = [x_tf[i].imag for i in range(len(x_tf))]
            plt.plot(X, Y, '-', color=self.paths[j].color)
        bp_tf = [moebius(A, b) for b in self.curve.branch_points]
        bf_X = [bp_tf[i].real for i in range(len(bp_tf))]
        bf_Y = [bp_tf[i].imag for i in range(len(bp_tf))]
        plt.plot(bf_X, bf_Y, 'x', color='orange')
        sp_tf = [moebius(A, s) for s in self.curve.sing_points]
        sf_X = [sp_tf[i].real for i in range(len(sp_tf))] + [0, -1]
        sf_Y = [sp_tf[i].imag for i in range(len(sp_tf))] + [0, 0]
        plt.plot(bf_X, bf_Y, 'x', color='orange')
        plt.plot(sf_X, sf_Y, 'bo')
        fig.savefig("./graphics/" + filename + "_tf.png", dpi=200)

    def plot_all_networks(self, steps=0):
        for j in range(len(self.paths)):
            self.plot_network(steps=0, paths=[j], fix_axis=True, filename="Path_{}.png".format(j))