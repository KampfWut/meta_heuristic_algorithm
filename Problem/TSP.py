# !/usr/bin/env python3
"""
    author: Apeiria
    contact: apeiriaStar@outlook.com
    file: TSP.py
    function: Definition of traveling salesman problem
"""

# ---------------------------------- import --------------------------------- #
import ray
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------- Main Class ----------------------------- #
class TSP(object):
    
    """Traveling salesman problem class"""
    
    def __init__(self, 
                 node_num, 
                 square_length):
        """Class initialization

        Args:
            node_num (int): Number of TSP Node
            square_length (int): Side length of square map
        """
        self.__param_check(node_num, int, "node_num")
        self.__param_check(square_length, int, "square_length")
        self.node_num = node_num       
        self.square_length = square_length   
        self.square = np.zeros((square_length, square_length)) 
        self.node_info_list = []                 
        self.node_distance = np.zeros((node_num, node_num)) 
        self.solve, self.Fit = None, None
    
    
    def __param_check(self, 
                      param,
                      type,
                      param_name):
        """Parameter validity check 

        Args:
            param (Any): Param
            type (Any): Param type
            param_name (string): Param Name
        """
        assert isinstance(param, type), "[ERROR] {} need {}".format(param_name, type)
        return
    
    
    def generate(self):
        """Generate a random TSP problem map"""
        assert self.node_num < self.square_length ** 2
        # Build Node
        for i in range(0, self.node_num):
            x = np.random.randint(0, self.square_length - 1)
            y = np.random.randint(0, self.square_length - 1)
            self.node_info_list.append([x, y])
            self.square[x, y] = 1
        # Cal Distance
        for i in range(0, self.node_num):
            for j in range(i + 1, self.node_num):
                dis = self.calDistance(i, j)
                self.node_distance[i, j], self.node_distance[j, i] = dis, dis
        return


    def calDistance(self, idx, idy):
        """Distance calculation function

        Args:
            idx (int): Node x index
            idy (int): node y index

        Returns:
            float: Euclidean distance between X and Y
        """
        self.__param_check(idx, int, "idx")
        self.__param_check(idy, int, "idy")
        dis = (self.node_info_list[idx][0] - self.node_info_list[idy][0])**2
        dis += (self.node_info_list[idx][1] - self.node_info_list[idy][1])**2
        return dis ** 0.5


    def fitFunction(self, solve, auxiliary_input):
        """Calculate fitness function

        Args:
            solve (list): Now Seq
            auxiliary_input (list): auxiliary input(No use in this problem)

        Returns:
            float: fit value
        """
        self.__param_check(solve, list, "solve")
        self.__param_check(auxiliary_input, list, "auxiliary_input")
        fit_value = 0
        for i in range(0, self.node_num):
            if i != self.node_num - 1:
                fit_value += self.node_distance[solve[i], solve[i + 1]]
            else:
                fit_value += self.node_distance[solve[i], solve[0]]
        return -fit_value   


    def setForDraw(self, solve, Fit):
        """Packing function for draw

        Args:
            solve (list): Now Seq
            Fit (float): Now fit value
        """
        self.__param_check(solve, list, "solve")
        self.__param_check(Fit, float, "Fit")
        self.solve = solve
        self.Fit = Fit
        return


    def show(self, method_name):
        """Draw the figure

        Args:
            method_name (string): method name
        """
        x, y = [], []
        for i in range(0, self.node_num):
            x.append(self.node_info_list[i][0])
            y.append(self.node_info_list[i][1])
        f = plt.figure("TSP", figsize=(5, 5))
        ax1 = f.add_subplot(111)
        plt.title("TSP-{}, Fit={:.6f}".format(method_name, self.Fit))
        ax1.scatter(x, y)
        for i in range(0, self.node_num):
            if i != self.node_num - 1:
                ax1.plot([x[self.solve[i]], x[self.solve[i+1]]], [y[self.solve[i]], y[self.solve[i+1]]], 'g')
            else:
                ax1.plot([x[self.solve[i]], x[self.solve[0]]], [y[self.solve[i]], y[self.solve[0]]], 'g')
        plt.savefig("./Figure/TSP-{}.jpg".format(method_name))
        plt.show()
        return


    def save(self):
        """Class save"""
        dict = {"node_num": self.node_num,
                "square_length": self.square_length,
                "square": self.square,
                "node_info_list": self.node_info_list,
                "node_distance": self.node_distance}
        np.save('./TempData/TSPProblemDict.npy', dict, allow_pickle=True) 
        return


    def load(self):
        """Class load"""
        dict = np.load('./TempData/TSPProblemDict.npy', allow_pickle=True).item()
        self.node_num = dict["node_num"]
        self.square_length = dict["square_length"]
        self.square = dict["square"]
        self.node_info_list = dict["node_info_list"]
        self.node_distance = dict["node_distance"]
        return
        