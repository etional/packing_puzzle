import numpy as np
from threading import Thread
import functools
from scipy import ndimage
import random

# Global variable
closest_sol = np.zeros((1, 1))
solution_found = False


class State:
    def __init__(self, grid, bag):
        self.grid = grid
        self.bag = bag

    def is_goal(self):
        if np.all(self.grid) and len(self.bag) == 0:
            return True
        else:
            return False

    def get_successors(self):
        tiles = []
        if self.bag.num_t is not 0:
            tile = T()
            tiles.append(tile)
        if self.bag.num_i is not 0:
            tile = I()
            tiles.append(tile)
        if self.bag.num_o is not 0:
            tile = O()
            tiles.append(tile)
        if self.bag.num_j is not 0:
            tile = J()
            tiles.append(tile)
        if self.bag.num_s is not 0:
            tile = S()
            tiles.append(tile)

        successors = []
        for tile in tiles:
            possible = self.can_fit(tile)
            bag = self.bag.copy()
            if type(tile) is T:
                bag.num_t -= 1
            if type(tile) is I:
                bag.num_i -= 1
            if type(tile) is O:
                bag.num_o -= 1
            if type(tile) is J:
                bag.num_j -= 1
            if type(tile) is S:
                bag.num_s -= 1

            for p in possible:
                pos = p[0]
                shp = p[1]
                grid = self.grid.copy()
                for row in range(shp.shape[0]):
                    for col in range(shp.shape[1]):
                        grid[pos[0] + row][pos[1] + col] += shp[row][col]
                successor = State(grid, bag)
                successors.append((successor, p[2]))
        return successors

    def can_fit(self, tile):
        loc_shape = []
        shp = tile.shape
        zeros = np.argwhere(self.grid == 0)
        for i in range(tile.num_reflect):
            shp = reflectTile(shp)
            for j in range(tile.num_rotate):
                shp = rotateTile(shp)
                for index in zeros:
                    if index[0] + shp.shape[0] > self.grid.shape[0]:
                        continue
                    if index[1] + shp.shape[1] > self.grid.shape[1]:
                        continue
                    overlap = False
                    for row in range(len(shp)):
                        for col in range(len(shp[row])):
                            if shp[row][col] != 0 and self.grid[index[0] + row][index[1] + col] != 0:
                                overlap = True
                                break
                        if overlap:
                            break
                    if not overlap:
                        min_dist = self.sum_min_dist(index, shp)
                        loc_shape.append((index, shp, min_dist))
        return loc_shape

    def num_hole(self):
        grid = self.grid.copy()
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if grid[row][col] == 0:
                    grid[row][col] = np.NaN
                else:
                    grid[row][col] = 0
        labels, holes = ndimage.measurements.label(np.isnan(grid))
        return holes

    def sum_dist_non_zeros(self):
        nonzero_idx = np.transpose(np.nonzero(self.grid))
        sum_dist = 0
        for idx_1 in nonzero_idx:
            for idx_2 in nonzero_idx:
                dist = abs(idx_2[0] - idx_1[0]) + abs(idx_2[1] - idx_1[1])
                sum_dist += dist
        return sum_dist

    def sum_min_dist(self, pos, shp):
        result = 0

        if not np.any(self.grid):
            for row in range(shp.shape[0]):
                for col in range(shp.shape[1]):
                    if shp[row][col] != 0:
                        x_dist = min(pos[0] + row, self.grid.shape[0] - (pos[0] + row))
                        y_dist = min(pos[1] + col, self.grid.shape[1] - (pos[1] + col))
                        min_dist = x_dist + y_dist
                        result += min_dist
            return result

        for row in range(shp.shape[0]):
            for col in range(shp.shape[1]):
                min_dist = self.grid.shape[0] + self.grid.shape[1]
                if shp[row][col] != 0:
                    for r in range(self.grid.shape[0]):
                        for c in range(self.grid.shape[1]):
                            if self.grid[r][c] != 0:
                                dist = abs(pos[0] + row - r) + abs(pos[1] + col - c)
                                if dist < min_dist:
                                    min_dist = dist
                    result += min_dist
        return result

    def print_grid(self):
        for row in self.grid:
            for col in row:
                print("%d" % col, end=" ")
            print("\n")


class Bag:
    def __init__(self, k):
        self.num_t = k
        self.num_i = k
        self.num_o = k
        self.num_j = k
        self.num_s = k

    def content(self):
        print("The bag contains:")
        print("%d T's" % self.num_t)
        print("%d I's" % self.num_i)
        print("%d O's" % self.num_o)
        print("%d J's" % self.num_j)
        print("%d S's" % self.num_s)

    def total(self):
        return self.num_t + self.num_i + self.num_o + self.num_j + self.num_s

    def copy(self):
        copied = Bag(0)
        copied.num_t = self.num_t
        copied.num_i = self.num_i
        copied.num_o = self.num_o
        copied.num_j = self.num_j
        copied.num_s = self.num_s
        return copied


class T:
    def __init__(self):
        self.shape = np.array([[1, 1, 1],
                               [0, 1, 0]])
        self.num_rotate = 4
        self.num_reflect = 1


class I:
    def __init__(self):
        self.shape = np.array([[2, 2, 2, 2]])
        self.num_rotate = 2
        self.num_reflect = 1


class O:
    def __init__(self):
        self.shape = np.array([[3, 3],
                               [3, 3]])
        self.num_rotate = 1
        self.num_reflect = 1


class J:
    def __init__(self):
        self.shape = np.array([[0, 4],
                               [0, 4],
                               [4, 4]])
        self.num_rotate = 4
        self.num_reflect = 2


class S:
    def __init__(self):
        self.shape = np.array([[0, 5, 5],
                               [5, 5, 0]])
        self.num_rotate = 2
        self.num_reflect = 2


def rotateTile(tile):
    return np.rot90(tile)


def reflectTile(tile):
    return np.flip(tile, 1)


def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded' % (func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('Error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                print("Function [%s] timeout [%s seconds] exceeded" % (func.__name__, timeout))
                print("The closest Solution:")
                closest_sol.print_grid()
                raise ret
            return ret
        return wrapper
    return deco


def is_valid_grid(height, width):
    area = height * width
    if area % 20 == 0:
        return int(area / 20)
    return 0


def random_climbing_hill(state):
    i = 1
    min_tiles = state.bag.total()
    while True:
        current = state
        while not current.is_goal():
            best_successors = []
            successors = current.get_successors()
            if len(successors) == 0:
                if i % 20 == 0:
                    print("Trial # %d failed to find a solution." % i)
                i = i + 1
                if current.bag.total() < min_tiles:
                    min_tiles = current.bag.total()
                    global closest_sol
                    closest_sol = current
                break
            min_dist = successors[0][1]
            for successor in successors:
                if successor[0].is_goal():
                    solution = successor[0]
                    global solution_found
                    solution_found = True
                    return solution
                dist = successor[1]
                if dist < min_dist:
                    min_dist = dist
            for successor in successors:
                if successor[1] == min_dist:
                    best_successors.append(successor[0])
            random_idx = random.randint(0, len(best_successors) - 1)
            current = best_successors[random_idx]


def local_beam_search(state, k=5):
    i = 1
    min_tiles = state.bag.total()
    while True:
        currents = [state]
        while True:
            all_successors = []
            for current in currents:
                successors = current.get_successors()
                for successor in successors:
                    if successor[0].is_goal():
                        solution = successor[0]
                        global solution_found
                        solution_found = True
                        return solution
                    sum_dist = successor[0].sum_dist_non_zeros()
                    all_successors.append((successor[0], sum_dist))
                if current.bag.total() < min_tiles:
                    min_tiles = current.bag.total()
                    global closest_sol
                    closest_sol = current
            if len(all_successors) == 0:
                if i % 5 == 0:
                    print("Trial # %d failed to find a solution." % i)
                i = i + 1
                break
            all_successors.sort(key=lambda x: x[1])
            min_value = all_successors[0][1]
            min_successors = list(filter(lambda x:min_value in x, all_successors))
            if len(all_successors) > k:
                currents = []
                idxes = []
                if len(min_successors) < k:
                    for successor in all_successors[:k]:
                        currents.append(successor[0])
                else:
                    for i in range(k):
                        random_idx = random.randint(0, len(min_successors) - 1)
                        while random_idx not in idxes:
                            random_idx = random.randint(0, len(min_successors) - 1)
                        idxes.append(random_idx)
                        currents.append(min_successors[random_idx][0])
            else:
                currents = []
                for successor in all_successors:
                    currents.append(successor[0])


def generate_solution(height, width, search):
    if height == 1 or width == 1:
        print("Absolutely there is no solution for %d X %d grid" % (height, width))
        return

    num_set = is_valid_grid(height, width)

    if num_set == 0:
        print("Absolutely there is no solution for %d X %d grid" % (height, width))
        return

    # create a grid height x width
    grid = np.zeros((height, width))
    # create a bag of k set of tetrominos
    bag = Bag(num_set)

    startState = State(grid, bag)
    global closest_sol
    closest_sol = startState
    if search == 0:
        solution = random_climbing_hill(startState)
        print("Found a solution!")
        solution.print_grid()
    if search == 1:
        solution = local_beam_search(startState, k=5)
        print("Found a solution!")
        solution.print_grid()


def main():
    print("Packing Puzzle Solver\n")

    while True:
        height = input("Please tell the height:")
        if height.isnumeric():
            if int(height) > 0:
                height = int(height)
                break
        print("Please type a positive integer\n")

    while True:
        width = input("Please tell the width:")
        if width.isnumeric():
            if int(width) > 0:
                width = int(width)
                break
        print("Please type a positive integer\n")

    print("Search algorithm:")
    print("0 - Random Hill Climbing")
    print("1 - Local Beam Search")
    algo = ["Random Hill Climbing", "Local Beam Search"]
    while True:
        search = input("Please tell the search algorithm:")

        if search.isnumeric():
            if int(search) in range(len(algo)):
                search = int(search)
                break
        print("Please type a valid selection\n")

    print("You typed: height = %d, width = %d, algorithm = %s\n" % (height, width, algo[search]))

    print("Start Solving...")

    func = timeout(timeout=100)(generate_solution)(height, width, search)
    try:
        func()
    except:
        pass


if __name__ == '__main__':
    main()