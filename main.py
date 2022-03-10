# Executable file
# This file should only contain function calls 
# imported from other files

# Include modules 
import src.sw_curve as sw
import src.path as pth
import src.network as nw
import src.pathmap as mp

# Include default libraries (ideally not much needed)
import numpy as np
import time

# Main function
def main():
    dt = 1e-8
    theta = -0.02
    cutoff = 100000
    expo_net = nw.network(sw.H_c3, dt, theta, cutoff, expo=True)
    print(expo_net.pts)
    expo_net.start_paths()
    expo_net.evolve()
    print("Evolution_finished!")
    expo_net.plot_network(paths=[], fix_axis=True, filename="expo_net_lq")
    expo_net.plot_all_networks()
    for i in range(len(expo_net.all_intersections)):
        print(expo_net.all_intersections[i].path_index)
        print(expo_net.all_intersections[i].target_map)
        print("Intersection {}".format(i))
        for j in range(len(expo_net.all_intersections[i].coordinates)):
            print(expo_net.all_intersections[i].steps, expo_net.all_intersections[i].coordinates[j])
    
    for j in range(len(expo_net.paths)):
        print(len(expo_net.paths[j].mass))
        print(expo_net.paths[j].mass[-1])

if __name__ == '__main__':
    tic = time.time()
    main()
    toc = time.time()
    print("Total time:", toc - tic)