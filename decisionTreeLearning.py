import math

import copy

import pandas as p #for creating list of lists from excel

from treelib import Node, Tree #for printing tree

#defining node class
class Node :
    def __init__(self) :
        self.type = None
        self.attribute = None
        self.branches = None
        self.children = None

#creating decision tree
class DecisionTree :
    'class for decision tree creation'

    #initialising decision tree's algroithm
    def __init__(self, attributes, target) :
        self.root = Node()
        names = attributes[0]#getting names
        del attributes[0]#removing names
        self.recursiveInsertion(self.root, attributes, target, names)

    #defining ID3 algorithm
    def recursiveInsertion(self, currentNode, attributes, target, attributeNames) :
        
        #labeling tree elements   
        gain, removedAttributeName, removedAttributeValues, remainingAttributes, splittedAttributeTarget = self.maxGain(attributes, target)
        #gain is information gain value
        #removedAttributeName is the name that will be assigned to the node
        #reovedAttributeValues is the name for braches' node
        #remainingAttributes are arrays sorted by removed attribute's values
        #splittedAttributeTarget are lists splitted by removed attribute's values
        #remainingNames return remaining names when removed attribute's name with the highest information gain value

        if(gain == 0 and any(remainingAttributes)) : #any to check if a list is not empty
            currentNode.type = 'leaf'
            currentNode.children = [] # because this is a leaf
            currentNode.attribute = splittedAttributeTarget
        elif(any(remainingAttributes)) :
            currentNode.type = 'node'
            currentNode.children = []
            names = copy.deepcopy(attributeNames)
            currentNode.attribute =  names[removedAttributeName]#node -> attribute name
            currentNode.branches = removedAttributeValues #branches -> attribute's values
            del names[removedAttributeName]
            for i in range (0, len(remainingAttributes)) : 
                currentNode.children.append(Node())
            for i in range(0, len(remainingAttributes)) :
                self.recursiveInsertion(currentNode.children[i], remainingAttributes[i], splittedAttributeTarget[i], names)

    #defining max gain method               
    def maxGain(self, table, S) :

        #getting max information gain and values of the attribute
        max, attributeValues = self.gain ([row[0] for row in table], S)#row[n] for row in listOfLists for selecting a specific column
        indiceTracker = 0
        for i in range(1, len(table[0])) :#A[2] to get the length of dataset row
            if(max < self.gain([row[i] for row in table], S)[0]) :#i is the column
                max, attributeValues = self.gain([row[i] for row in table], S)
                indiceTracker = i

        attributeName = indiceTracker

        #splitting target attribute values and creating tables sorted by attribute values
        #note: we need to add copy.deepcopy because .append only refers to them (with only .append
        #will be changed also the original list)
        newTargets = []
        newTables = []
        for j in range(0, len(attributeValues)) :
            tmpTarget = []  
            tmpTable = []
            for k in range(0, len(S)) :
                if(table[k][indiceTracker] == attributeValues[j]) :
                    tmpTarget.append(S[k])# append points only from the original list
                                          # it doesnt copy lists elements! (so they can be modified from the original)  
                    tmpTable.append(table[k])

            #deep copying tmp table and target
            deepCopyTarget = copy.deepcopy(tmpTarget)
            deepCopyTable = copy.deepcopy(tmpTable)
                
            #removing highest attribute column
            for l in range(0, len(deepCopyTable)) :
                del deepCopyTable[l][indiceTracker]
            if(any(deepCopyTarget) and any(deepCopyTable)) :    
                newTargets.append(deepCopyTarget)
                newTables.append(deepCopyTable) 
                      

        return max, attributeName, attributeValues, newTables, newTargets

    #defining information gain algorithm        
    def gain(self, A, S) :
        result = self.entropy(S)
        variables = (list(set(A)))

        #removing from entropy "before split", the entropy "after split"
        for i in range(0, len(variables)) :
            tmp = [] #for single attribute's variables list 
            for j in range(0, len(S)) :
                if(A[j] == variables[i]) :
                    tmp.append(S[j])
            result -= (len(tmp)/len(S))*self.entropy(tmp)   

        return result, variables         

    #defining entropy
    def entropy(self, S) :
        #counting how many values the target has
        counter = 0 #to count every single values of single target attribute's value
        totalCounter = len(S) #to count total values of target
        targetValues = (list(set(S))) #to know which values are in target
        result = 0 #to get entropy

        for i in range (0, len(targetValues)) :
            #counting values
            for j in range(0, len(S)) :
                if(S[j] == targetValues[i]) :
                    counter += 1

            #getting entropy
            if(counter != 0) :
                result -= (counter/totalCounter)*math.log2(counter/totalCounter)

            counter = 0    

        return result 

#method for measuring test's accuracy ONLY FOR IRIS DATASET
def accuracy(decisionTree, testData) :
    
    if(decisionTree.attribute == 'sepal_length') :#switching attribute type name into a value
        value = 0
    elif(decisionTree.attribute == 'sepal_width') :
        value = 1
    elif(decisionTree.attribute == 'petal_length') :
        value = 2
    else :
        value = 3

    if(decisionTree.type == 'leaf'):
        #checking if the branch has only one value and we compare it test's target
        if(testData[len(testData)-1] == decisionTree.attribute[0][0]) :
            return True
        elif(testData[len(testData)-1] != decisionTree.attribute[0][0]) :
            return False     

    #exploring the tree
    for i in range(0, len(decisionTree.children)) :
        
        #continue thourgh tree's children
        if(testData[value] == decisionTree.branches[i]) : #checking which decisionTree attribute value is equal to test attribute value 
            return accuracy(decisionTree.children[i], testData)

    #else when node.type is None (means that lists are empty)
    #sometimes can happend that we dont have any information about an attribute branch (no array/targets)
    return False        


#method for printing tree
def printTree(currentNode, isIris) :

    #list of attributes
    if(isIris) :
        names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    else :
        names = ["Outlook", "Temperature", "Humidity", "Wind"]

    #saving lists of branch name, target name and attributes names
    attributeName = []
    targetName = []
    branchName = []

    for i in range(0, len(names)) :
        if(currentNode.attribute == names[i]) :
            attributeName.append(names[i])

    for i in range (0, len(currentNode.children)) :
        if(currentNode.children[i].type == 'leaf') :
            #appending branch, attribute and target name
            branchName.append(currentNode.branches[i])
            targetName.append(currentNode.children[i].attribute[0][0])#we are using list set to print only a value because they are all the same

        elif(currentNode.children[i].type == 'node') :
            branchName.append(currentNode.branches[i])
            #adding new node attribute name instead target name

            for j in range(0, len(names)) :
                if(currentNode.children[i].attribute == j) :
                    targetName.append(names[j])
                 
            tmp1, tmp2, tmp3, = printTree(currentNode.children[i], isIris)
            attributeName.append(tmp1)
            branchName.append(tmp2)
            targetName.append(tmp3)


        #if we dont any information about an attribute
        else:
            branchName.append(currentNode.branches[i])
            targetName.append("   ")

    
    return attributeName, branchName, targetName

#main
if __name__ == "__main__":
    
    #transforming excel table to list of lists
    #training dataset
    #attributes
    test = p.read_excel(".//dataset.xlsx", header = None, usecols = "A,B,C,D", nrows = 88)
    list_values = test.values.tolist()
    trainingAttributes = [list(filter(None, l)) for l in list_values]

    #target attributes
    test = p.read_excel(".//dataset.xlsx", header = None, usecols = "E", skiprows = 1, nrows = 87)
    trainingSpecies = test[4].tolist()

    #test dataset
    test = p.read_excel(".//dataset.xlsx", header = None, usecols = "G,H,I,J,K", skiprows = 1, nrows = 63)
    list_values = test.values.tolist()
    dataTest = [list(filter(None, l)) for l in list_values]
    
    #creating decision tree
    tree = DecisionTree(trainingAttributes, trainingSpecies)
    test1, test2, test3 = printTree(tree.root, True)
    result = 0
    #"testing" test data: 60% training 40% test
    for j in range(0, len(dataTest)) :
        if(accuracy(tree.root, dataTest[j])) :
            result += 1


    print("The accuracy of the decision tree is: ", result/len(dataTest))

    
    #create tree for this specific dataset: iris
    firstList = test2[0:6]
    firstList.extend(test2[7:16])
    firstList.extend(test2[17:21])
    firstList.extend(test2[22:24])

    secondList = test3[0:5]
    secondList.extend(test3[6:14])
    secondList.extend(test3[15:18])
    secondList.extend(test3[19:24])

    thirdList = []
    thirdList.append(test2[6])
    thirdList.append(test2[16])
    thirdList.append(test2[21])

    fourthList = []
    fourthList.append(test3[5])
    fourthList.append(test3[14])
    fourthList.append(test3[18])   

    #for tracking subtrees
    j = 0 #for leaves on second layer of nodes
    b = 0
    r = 0 #for leaves on the first layer of nodes
    
    #plotting tree for iris dataset
    plotTree = Tree()
    plotTree.create_node(test1[0], "radice")
    for i in range(0, len(firstList)) :
        plotTree.create_node(firstList[i], firstList[i], parent = "radice")
    for i in range(0, len(firstList)) :
        a = i*10 #to have different values
        if(i == 5 or i == 14 or i == 18) : 
            plotTree.create_node(test1[j+1], a, parent = firstList[i])
            for k in range(0, len(thirdList[j])) :
                b += 1
                c = (b*2)+1 #to have different values
                plotTree.create_node(thirdList[j][k], c, parent = a)
                plotTree.create_node(fourthList[j][k], parent = c)
            j += 1
        else : 
            plotTree.create_node(secondList[r], a, parent = firstList[i])
            r += 1
    plotTree.show()
    

    del tree
    
    #creating playTennis dataset
    #transforming excel table to list of lists
    #training dataset
    #attributes
    test = p.read_excel(".//dataset.xlsx", header = None, usecols = "A,B,C,D", skiprows = 89, nrows = 15)
    list_values = test.values.tolist()
    playTennisAttributes = [list(filter(None, l)) for l in list_values]

    #target attributes
    test = p.read_excel(".//dataset.xlsx", header = None, usecols = "E", skiprows = 90, nrows = 14)
    playTennisSpecies = test[4].tolist()
    print(playTennisSpecies)

    tennisTree = DecisionTree(playTennisAttributes, playTennisSpecies)
    tennis1, tennis2, tennis3 = printTree(tennisTree.root, False)
    #print("tennis1 ", tennis1)
    #print("tennis2 ", tennis2)
    #print("tennis3 ", tennis3)
    
    #plotting tree for playTennis
    tennisTree = Tree()
    tennisTree.create_node("Outlook", "root") #first level tree
    tennisTree.create_node("Sunny", "Sunny", parent = "root")#second level tree
    tennisTree.create_node("Overcast", "Overcast", parent = "root")
    tennisTree.create_node("Rain", "Rain", parent = "root")
    tennisTree.create_node("Humidity", "Humidity", parent = "Sunny")#third level tree
    tennisTree.create_node("Yes", parent = "Overcast")
    tennisTree.create_node("Wind", "Wind", parent = "Rain")
    tennisTree.create_node("Normal", "Normal", parent = "Humidity")#fourth level tree
    tennisTree.create_node("High", "High", parent = "Humidity")
    tennisTree.create_node("Strong", "Strong", parent = "Wind")
    tennisTree.create_node("Weak", "Weak", parent = "Wind")
    tennisTree.create_node("Yes", parent = "Normal")#fifth level tree
    tennisTree.create_node("No", parent = "High")
    tennisTree.create_node("No", parent = "Strong")
    tennisTree.create_node("Yes", parent = "Weak")

    tennisTree.show()

    print("success!")