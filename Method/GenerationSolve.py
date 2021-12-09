# !/usr/bin/env python3
"""
    author: Apeiria
    contact: apeiriaStar@outlook.com
    file: GenerationSolve.py
    function: Tool class for building a new solution.
"""
# ---------------------------------- import --------------------------------- #
import random

# ---------------------------------- Main Class ----------------------------- #
class GenerationSolve(object):
    """Tool class for building a new solution"""
    
    # Continuity transformation operator
    
    @classmethod
    def get_NewContinue_RandomFluctuation(cls, Solve):
        """Random fluctuation of continuous solutions
           1, 1, 0, 0.5 ,1 -> 0.78, 0.8, -0.2, -0.3, 1.3

        Args:
            Solve (list): Seq Now

        Returns:
            list: Changed Seq
        """
        NodeNumber = len(Solve)
        NewSolve = []
        for i in range(0, NodeNumber):
            NewSolve.append(Solve[i] + (random.random() * 2 - 1))
        return NewSolve
    
    # Sequence transformation operator
    
    @classmethod
    def __seqSolve_check(cls, Solve):
        """Validity check of sequence solution

        Args:
            Solve (list): Now Seq

        Raises:
            Exception: Duplicate Elements, Mismatch Data Type.
        """
        assert isinstance(Solve, list) == True, "[ERROR] Solve must be List."
        for i in range(0, len(Solve)):
            assert isinstance(Solve[i], int) == True, "[ERROR] Solve element must all be Int."
        inside_list = []
        for i in range(0, len(Solve)):
            if Solve[i] not in inside_list:
                inside_list.append(Solve[i])
            else:
                raise Exception("[ERROR] Duplicate element {} are not allowed in seq solve.".format(Solve[i]))
        return
        
        
    @classmethod
    def get_NewSeq_RandomSwitchOne(cls, Solve):
        """Random exchange of a sequence element
           1,(2),3,4,(5) -> 1,(5),3,4,(2)

        Args:
            Solve (list): Seq Now

        Returns:
            list: Changed Seq
        """
        cls.__seqSolve_check(Solve)
        NodeNumber = len(Solve)
        if NodeNumber - 1 >= 1:
            x, y = random.randint(0, NodeNumber - 1), random.randint(0, NodeNumber - 1)
            while x == y:
                x, y = random.randint(
                    0, NodeNumber - 1), random.randint(0, NodeNumber - 1)
            newSolve = []
            newSolve.extend(Solve)
            newSolve[x], newSolve[y] = newSolve[y], newSolve[x]
        else:
            newSolve = []
            newSolve.extend(Solve)
        return newSolve


    @classmethod
    def get_NewSeq_RandomSwitchPart(cls, Solve):
        """Random exchange of a part of sequence element
           (1,2),3,(4,5) -> (4,5),3,(1,2)
    
        Args:
            Solve (list): Now Seq

        Returns:
            list: Changed Seq
        """
        cls.__seqSolve_check(Solve)
        NodeNumber = len(Solve)
        n = random.randint(1, int(NodeNumber / 4))
        x = random.randint(0, int(NodeNumber / 4) - 1)
        y = random.randint(int(NodeNumber / 4) * 2, int(NodeNumber / 4) * 3 - 1)
        newSolve = []
        newSolve.extend(Solve)
        for i in range(0, n):
            newSolve[x + i], newSolve[y + i] = newSolve[y + i], newSolve[x + i]
        return newSolve


    @classmethod
    def get_NewSeq_PartReverse(cls, Solve):
        """Random reverse of a part of sequence element
           1,(2,3,4),5 -> 1,(4,3,2),5

        Args:
            Solve (list): Now Seq

        Returns:
            list: Changed Seq
        """
        cls.__seqSolve_check(Solve)
        NodeNumber = len(Solve)
        x, y = random.randint(0, NodeNumber - 1), random.randint(0, NodeNumber - 1)
        while x == y:
            x, y = random.randint(0, NodeNumber - 1), random.randint(0, NodeNumber - 1)
        if x > y:
            x, y = y, x
        temp = Solve[x: y + 1]
        temp.reverse()
        newSolve = []
        newSolve.extend(Solve[0: x])
        newSolve.extend(temp)
        newSolve.extend(Solve[y + 1: ])
        return newSolve


    @classmethod
    def get_NewSeq_PartJump(cls, Solve):
        """Cyclic random exchange of some sequence elements
           (1),2,(3),4,(5) -> (5),2,(1),4,(3)
           
        Args:
            Solve (list): Now Seq

        Returns:
            list: Changed Seq
        """
        cls.__seqSolve_check(Solve)
        NodeNumber = len(Solve)
        n = random.randint(2, 5)
        i, count = 0, 0
        part_1, part_2 = [], []
        while i < NodeNumber:
            end = i + n if i + n <= NodeNumber else NodeNumber 
            if count % 2 == 0:
                part_1.append(Solve[i:end])
            else:
                part_2.append(Solve[i:end])
            i = i + n
            count += 1
        NewSolve = []
        index, i = -1, 0
        while i < len(part_1):
            NewSolve.extend(part_1[i])
            if index < len(part_2) - 1:
                NewSolve.extend(part_2[index])
            index += 1
            i += 1
        return NewSolve
        
    