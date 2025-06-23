import numpy as np
import math
import Configurations as config


def normpdf(x, mean, sd=0.1):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = np.exp(-(x - mean)**2 / (2 * var))
    return num/denom


class Graph:
    def __init__(self, max_size=config.MAX_EMB_SIZE):
        self.threshold = 5  # 2
        self.gdict = self.get_dgict(max_size, self.threshold)

    def walk(self, starts, steps):
        # starts are the steps computed by the actor
        starts = np.round(starts).astype(np.int32)
        paths = [starts]
        for _ in range(steps):
            neighbours = self.gdict[starts - config.MIN_EMB_SIZE]
            dist = 1 / np.abs(neighbours - np.expand_dims(starts, axis=1))
            # normalised the dist
            dist = dist / np.sum(dist, axis=1)[:, None]
            # randomly choose one element from each row
            assert len(dist) == len(neighbours) == len(starts)
            # pick one neighbour (computationally heavy)
            next_vertex = [np.random.choice(neighbours[i], 1, p=dist[i], replace=False)[0] for i in range(len(dist))]

            next_vertex = np.array(next_vertex)
            starts = next_vertex
            paths.append(next_vertex)
        return np.stack(paths).transpose()

    def get_dgict(self, max_size, threshold):
        sizes = list(range(config.MIN_EMB_SIZE, max_size + 1))
        gdict = []
        for i in sizes:
            edges = []
            for j in sizes:
                if i != j and abs(i - j) <= threshold:
                    edges.append(j)
            edges.extend([999999**2] * (self.threshold * 2 - len(edges)))
            gdict.append(np.array(edges))
        return np.stack(gdict)


