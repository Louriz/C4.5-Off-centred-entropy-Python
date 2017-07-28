from __future__ import division
import math
import operator
import time
import random
import copy
import sys
import ast
import csv
from collections import Counter
from collections import OrderedDict
from numbers import Number

#================== for categorical entropy=================================
def dataset_to_table(dataset):
    table={}
    for k in range(len(dataset.attributes)):
        table[dataset.attributes[k]]=[]
        for example in dataset.examples:
            table[dataset.attributes[k]].append(example[k])
    return table

def deldup(li):
    """ Deletes duplicates from list _li_
        and return new list with unique values.
    """
    return list(OrderedDict.fromkeys(li))
################Crucial parameters################
#Off-centred entropy parameter
theta=0.8
big_theta=[]
##################################################
# data class to hold csv data
##################################################
class data():
    def __init__(self, classifier):
        self.examples = []
        self.attributes = []
        self.attr_types = []
        self.classifier = classifier
        self.class_index = None

##################################################
# function to read in data from the .csv files
##################################################
def read_data(dataset, datafile):
    print ("Reading data...")
    f = open(datafile)
    original_file = f.read()
    rowsplit_data = original_file.splitlines()
    dataset.examples = [rows.split(',') for rows in rowsplit_data]

    #list attributes
    dataset.attributes = dataset.examples.pop(0)
    #print(dataset.attributes)

    
    #create array that indicates whether each attribute is a numerical value or not
    #attr_type = open(datatypes) 
    #orig_file = attr_type.read()
    #dataset.attr_types = orig_file.split(',')
    #dataset.attr_types=['true','false','true','false','false','false','false','false','false','false','true','true','true','false','false']
    #datatypes=['false','false','false','false','true','false']
    #dataset.attr_types=['true','true','true','true','true','true','true','true','true','true','true','true','true','false']
    #dataset.attr_types=['false','true','false','false','true','false','false','true','false','false','true','false','true','false','false','true','false','true','false','false','false']
    #dataset.attr_types=[True, True,True,True,False] #  iris
    #dataset.attr_types=['true','true','true','true','true','true','true','true','false'] #  pima

    #dataset.attr_types=datatypes
    #dataset.attr_types=['true','true','true','true','true','true','true','true','true','false'] #  glass





##################################################
# Preprocess dataset
##################################################
def preprocess2(dataset):
    print ("Preprocessing data...")

    class_values = [example[dataset.class_index] for example in dataset.examples]
    
    class_mode = Counter(class_values)
    classes=class_mode.keys()
    
    class_mode = class_mode.most_common(1)[0][0]

    attr_modes = [0]*len(dataset.attributes)                 
    for attr_index in range(len(dataset.attributes)):

        ex_class={}
        values_class={}
        values={}
        mode={}

        for c in classes:

            ex_class[c] = filter(lambda x: x[dataset.class_index] == c, dataset.examples)
            values_class[c] = [example[attr_index] for example in ex_class[c]]  
                
            values[c] = Counter(values_class[c])
            
        
            mode[c] = values[c].most_common(1)[0][0]
            if mode[c] == '?':
                mode[c] = values.most_common(2)[1][0]

        
        for example in dataset.examples:
            if (example[attr_index] == '?'):
                for c in classes:
                    if (example[dataset.class_index] == c):
                        example[attr_index] = mode[c]
                    
                    else:
                        example[attr_index] = class_mode

        #convert attributes that are numeric to floats
        for example in dataset.examples:
            for x in range(len(dataset.examples[0])):
                if dataset.attr_types[x]==True:
                #if dataset.attributes[x] == 'true':
                    example[x] = float(example[x])

##################################################
# tree node class that will make up the tree
##################################################
class treeNode():
    def __init__(self, is_leaf, classification, attr_split_index, attr_split_value, parent, upper_child, lower_child, height):
        self.is_leaf = True
        self.classification = None
        self.attr_split = None
        self.attr_split_index = None
        self.attr_split_value = None
        self.parent = parent
        self.upper_child = None
        self.lower_child = None
        self.children = None
        self.height = None
###############################################
# compute tree for shannon
#############################################
def compute_tree_shannon(dataset, parent_node, classifier,datatypes):
    node = treeNode(True, None, None, None, parent_node, None, None, 0)
    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1

    count, classes = find_classes(dataset.examples, dataset.attributes, classifier)
    for c in classes:
        if (len(dataset.examples) == count[c]):
            node.classification = c
            node.is_leaf = True
            return node
        else:
            node.is_leaf = False
    attr_to_split = None # The index of the attribute we will split on
    max_gain = 0 # The gain given by the best attribute
    split_val = [] 
    min_gain = 0.01
    dataset_entropy = shannon_entropy(dataset, classifier)
    for attr_index in range(len(dataset.examples[0])):

        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in dataset.examples] # these are the values we can split on, now we must find the best one
            attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values
            if(datatypes[attr_index]==True):
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total/10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x*ten_percentile])
                attr_value_list = new_list

                for val in attr_value_list:
                    # calculate the gain if we split on this value
                    # if gain is greater than local_max_gain, save this gain and this value
                    local_gain = calc_gain_shannon(dataset, classifier, dataset_entropy, attr_index,datatypes, val) # calculate the gain if we split on this value
      
                    if (local_gain >= local_max_gain):
                        local_max_gain = local_gain
                        local_split_val = val
            else:
                local_max_gain=calc_gain_shannon(dataset, classifier, dataset_entropy, attr_index,datatypes)
                local_split_val=attr_value_list
            
            if (local_max_gain >= max_gain):
                max_gain = local_max_gain

                split_val = local_split_val
                attr_to_split = attr_index
            #print(max_gain,split_val)
    #attr_to_split is now the best attribute according to our gain metric
    if (split_val is None or attr_to_split is None):
        #print( "Something went wrong. Couldn't find an attribute to split on or a split value.")
        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node
    elif (max_gain <= min_gain or node.height >34):

        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)
        #print(dataset.examples)

        return node
    
    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_to_split]
    node.attr_split_value = split_val
    

    if (isinstance(split_val,Number)):
        # currently doing one split per node so only two datasets are created
        upper_dataset = data(classifier)
        lower_dataset = data(classifier)
        upper_dataset.attributes = dataset.attributes
        lower_dataset.attributes = dataset.attributes
        upper_dataset.attr_types = dataset.attr_types
        lower_dataset.attr_types = dataset.attr_types
        for example in dataset.examples:
            if (attr_to_split is not None and example[attr_to_split] >= split_val):
                upper_dataset.examples.append(example)
            elif (attr_to_split is not None):
                lower_dataset.examples.append(example)

        node.upper_child = compute_tree_shannon(upper_dataset, node, classifier,datatypes)
        node.lower_child = compute_tree_shannon(lower_dataset, node, classifier,datatypes)

        return node
    children_datasets=[data(classifier) for i in range(len(split_val))]
    for i in range(len(split_val)):
        children_datasets[i].attributes=dataset.attributes
        children_datasets[i].attr_types=dataset.attr_types
        for example in dataset.examples:
            if (example[node.attr_split_index]==split_val[i]):
                children_datasets[i].examples.append(example)
        node.children=[compute_tree_shannon(children_datasets[j],node,classifier,datatypes) for j in range(len(split_val))]
    #print(node.children)
    return node
##################################################
# compute tree recursively
##################################################

# initialize Tree
    # if dataset is pure (all one result) or there is other stopping criteria then stop
    # for all attributes a in dataset
        # compute information-theoretic criteria if we split on a
    # abest = best attribute according to above
    # tree = create a decision node that tests abest in the root
    # dv (v=1,2,3,...) = induced sub-datasets from D based on abest
    # for all dv
        # tree = compute_tree_oce(dv)
        # attach tree to the corresponding branch of Tree
    # return tree 

def compute_tree_oce(dataset, parent_node, classifier,datatypes):
    node = treeNode(True, None, None, None, parent_node, None, None, 0)
    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1

    count, classes = find_classes(dataset.examples, dataset.attributes, classifier)
    for c in classes:
        if (len(dataset.examples) == count[c]):
            node.classification = c
            node.is_leaf = True
            return node
        else:
            node.is_leaf = False
    attr_to_split = None # The index of the attribute we will split on
    max_gain = 0 # The gain given by the best attribute
    split_val = [] 
    min_gain = 0.01
    dataset_entropy = oce_entropy(dataset, classifier)
    for attr_index in range(len(dataset.examples[0])):

        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in dataset.examples] # these are the values we can split on, now we must find the best one
            attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values
            if(datatypes[attr_index]==True):
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total/10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x*ten_percentile])
                attr_value_list = new_list

                for val in attr_value_list:
                    # calculate the gain if we split on this value
                    # if gain is greater than local_max_gain, save this gain and this value
                    local_gain = calc_gain_oce(dataset, classifier, dataset_entropy, attr_index,datatypes, val) # calculate the gain if we split on this value
      
                    if (local_gain >= local_max_gain):
                        local_max_gain = local_gain
                        local_split_val = val
            else:
            	local_max_gain=calc_gain_oce(dataset, classifier, dataset_entropy, attr_index,datatypes)
            	local_split_val=attr_value_list
            
            if (local_max_gain >= max_gain):
                max_gain = local_max_gain

                split_val = local_split_val
                attr_to_split = attr_index
            #print(max_gain,split_val)
    #attr_to_split is now the best attribute according to our gain metric
    if (split_val is None or attr_to_split is None):
        #print( "Something went wrong. Couldn't find an attribute to split on or a split value.")
        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node
    elif (max_gain <= min_gain or node.height >34):

        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node
    
    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_to_split]
    node.attr_split_value = split_val
    

    if (isinstance(split_val,Number)):
        # currently doing one split per node so only two datasets are created
        upper_dataset = data(classifier)
        lower_dataset = data(classifier)
        upper_dataset.attributes = dataset.attributes
        lower_dataset.attributes = dataset.attributes
        upper_dataset.attr_types = dataset.attr_types
        lower_dataset.attr_types = dataset.attr_types
        for example in dataset.examples:
            if (attr_to_split is not None and example[attr_to_split] >= split_val):
                upper_dataset.examples.append(example)
            elif (attr_to_split is not None):
                lower_dataset.examples.append(example)

        node.upper_child = compute_tree_oce(upper_dataset, node, classifier,datatypes)
        node.lower_child = compute_tree_oce(lower_dataset, node, classifier,datatypes)

        return node
    children_datasets=[data(classifier) for i in range(len(split_val))]
    for i in range(len(split_val)):
        children_datasets[i].attributes=dataset.attributes
        children_datasets[i].attr_types=dataset.attr_types
        for example in dataset.examples:
            if (example[node.attr_split_index]==split_val[i]):
                children_datasets[i].examples.append(example)
        node.children=[compute_tree_oce(children_datasets[j],node,classifier,datatypes) for j in range(len(split_val))]
    #print(node.children)
    return node

############ compute root oae======================
def compute_tree_oae(dataset, parent_node, classifier,datatypes):
    node = treeNode(True, None, None, None, parent_node, None, None, 0)
    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1

    count, classes = find_classes(dataset.examples, dataset.attributes, classifier)
    for c in classes:
        if (len(dataset.examples) == count[c]):
            node.classification = c
            node.is_leaf = True
            return node
        else:
            node.is_leaf = False
    attr_to_split = None # The index of the attribute we will split on
    max_gain = 0 # The gain given by the best attribute
    split_val = [] 
    min_gain = 0.01
    dataset_entropy = oae_entropy(dataset, classifier)
    for attr_index in range(len(dataset.examples[0])):

        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in dataset.examples] # these are the values we can split on, now we must find the best one
            attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values
            if(datatypes[attr_index]==True):
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total/10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x*ten_percentile])
                attr_value_list = new_list

                for val in attr_value_list:
                    # calculate the gain if we split on this value
                    # if gain is greater than local_max_gain, save this gain and this value
                    local_gain = calc_gain_oae(dataset, classifier, dataset_entropy, attr_index,datatypes, val) # calculate the gain if we split on this value
      
                    if (local_gain >= local_max_gain):
                        local_max_gain = local_gain
                        local_split_val = val
            else:
            	local_max_gain=calc_gain_oae(dataset, classifier, dataset_entropy, attr_index,datatypes)
            	local_split_val=attr_value_list
            
            if (local_max_gain >= max_gain):
                max_gain = local_max_gain

                split_val = local_split_val
                attr_to_split = attr_index
            #print(max_gain,split_val)
    #attr_to_split is now the best attribute according to our gain metric
    if (split_val is None or attr_to_split is None):
        #print( "Something went wrong. Couldn't find an attribute to split on or a split value.")
        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node
    elif (max_gain <= min_gain or node.height >34):

        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node
    
    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_to_split]
    node.attr_split_value = split_val
    

    if (isinstance(split_val,Number)):
        # currently doing one split per node so only two datasets are created
        upper_dataset = data(classifier)
        lower_dataset = data(classifier)
        upper_dataset.attributes = dataset.attributes
        lower_dataset.attributes = dataset.attributes
        upper_dataset.attr_types = dataset.attr_types
        lower_dataset.attr_types = dataset.attr_types
        for example in dataset.examples:
            if (attr_to_split is not None and example[attr_to_split] >= split_val):
                upper_dataset.examples.append(example)
            elif (attr_to_split is not None):
                lower_dataset.examples.append(example)

        node.upper_child = compute_tree_oae(upper_dataset, node, classifier,datatypes)
        node.lower_child = compute_tree_oae(lower_dataset, node, classifier,datatypes)

        return node
    children_datasets=[data(classifier) for i in range(len(split_val))]
    for i in range(len(split_val)):
        children_datasets[i].attributes=dataset.attributes
        children_datasets[i].attr_types=dataset.attr_types
        for example in dataset.examples:
            if (example[node.attr_split_index]==split_val[i]):
                children_datasets[i].examples.append(example)
        node.children=[compute_tree_oae(children_datasets[j],node,classifier,datatypes) for j in range(len(split_val))]
    #print(node.children)
    return node

##################################################
# Classify dataset
##################################################
def classify_leaf(dataset, classifier):
    count, classes = find_classes(dataset.examples, dataset.attributes, classifier)  
    max=0
    for c in classes:
        if count[c]>=max:
            max=count[c]
            cfin=c
    return cfin
    
##########################################################
#### Find theta
############################################
def find_theta(dataset):
    count,classes=find_classes(dataset.examples, dataset.attributes, dataset.classifier)
    class_values = [example[dataset.class_index] for example in dataset.examples]
    class_mode = Counter(class_values)
    
    global theta , big_theta
    total_examples=len(dataset.examples)
    if len(classes)==2:
        theta=class_mode.most_common()[0][1]/total_examples
    else:
        big_theta=[count[c] / total_examples for c in classes]
    #print(theta,big_theta)
######  
############################################################

##################################################
# Calculate the entropy of the current dataset : Entropy(dataset)= - Sigma( pi*log2(pi)) i=1:k  , k is the number of modalities in the class
#########################"#############################"
def shannon_entropy(dataset,classifier):
    count,classes=find_classes(dataset.examples, dataset.attributes, classifier)
    total_examples=len(dataset.examples)
    entropy=0
    probabilities=[count[c]/total_examples for c in classes]
    #print(probabilities)
    for i in range(len(probabilities)):
        if(probabilities[i]==1 or probabilities[i]==0):
            entropy+=0
        else:
            entropy+=probabilities[i] * math.log(probabilities[i],2)
    entropy= -entropy
    return entropy
#### oAE entropy============================================
def oae_entropy(dataset,classifier):
    count,classes=find_classes(dataset.examples, dataset.attributes, classifier)
    total_examples=len(dataset.examples)
    entropy=0
    probabilities=[(count[c]*total_examples+1)/(total_examples+len(classes)) for c in classes]
    if(len(probabilities)==2):
        for i in range(len(classes)):
            entropy+=(probabilities[i]*(1-probabilities[i]))/(((-2*theta+1)*probabilities[i])+theta*theta)
    else:
        for i in range(len(classes)):
            entropy+=(probabilities[i]*(1-probabilities[i]))/(((-2*big_theta[i]+1)*probabilities[i])+big_theta[i]*big_theta[i])
    return entropy
#####====================== calc  gain oae
def calc_gain_oae(dataset,classifier, entropy, attr_index,datatypes, val=None):
    
    attr_entropy = 0
    total_examples = len(dataset.examples);
    #categorical variable
    if(datatypes[attr_index]==False):
        table=dataset_to_table(dataset)
        li=table[dataset.attributes[attr_index]]
        unique_li=deldup(li)
        datasets=[None for i in range(len(unique_li))]
        for i in range(len(unique_li)):
            datasets[i]=data(classifier)
            datasets[i].attributes=dataset.attributes
            for example in dataset.examples:
                if example[attr_index]==unique_li[i]:
                   datasets[i].examples.append(example)
            attr_entropy+=oae_entropy(datasets[i],classifier)*len(datasets[i].examples)/total_examples
        
    # numerical variable  
    else:
        
        gain_upper_dataset = data(classifier) # instanciate the class data with the parametr classifier
        gain_lower_dataset = data(classifier)
        gain_upper_dataset.attributes = dataset.attributes
        gain_lower_dataset.attributes = dataset.attributes
        gain_upper_dataset.attr_types = dataset.attr_types
        gain_lower_dataset.attr_types = dataset.attr_types
        for example in dataset.examples:
            if (example[attr_index] >= val):
                gain_upper_dataset.examples.append(example)
            elif (example[attr_index] < val):
                gain_lower_dataset.examples.append(example)

        if (len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0): #Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
            return -1

        attr_entropy += oae_entropy(gain_upper_dataset, classifier)*len(gain_upper_dataset.examples)/total_examples
        attr_entropy += oae_entropy(gain_lower_dataset, classifier)*len(gain_lower_dataset.examples)/total_examples

    return entropy - attr_entropy    # this is the IG ( Information Gain)






####### calc gain Shannon===================================

def calc_gain_shannon(dataset,classifier, entropy, attr_index,datatypes, val=None):
    
    attr_entropy = 0
    total_examples = len(dataset.examples);
    #categorical variable
    if(datatypes[attr_index]==False):
        table=dataset_to_table(dataset)
        li=table[dataset.attributes[attr_index]]
        unique_li=deldup(li)
        datasets=[None for i in range(len(unique_li))]
        for i in range(len(unique_li)):
            datasets[i]=data(classifier)
            datasets[i].attributes=dataset.attributes
            for example in dataset.examples:
                if example[attr_index]==unique_li[i]:
                   datasets[i].examples.append(example)
            attr_entropy+=shannon_entropy(datasets[i],classifier)*len(datasets[i].examples)/total_examples
        
    # numerical variable  
    else:
        
        gain_upper_dataset = data(classifier) # instanciate the class data with the parametr classifier
        gain_lower_dataset = data(classifier)
        gain_upper_dataset.attributes = dataset.attributes
        gain_lower_dataset.attributes = dataset.attributes
        gain_upper_dataset.attr_types = dataset.attr_types
        gain_lower_dataset.attr_types = dataset.attr_types
        for example in dataset.examples:
            if (example[attr_index] >= val):
                gain_upper_dataset.examples.append(example)
            elif (example[attr_index] < val):
                gain_lower_dataset.examples.append(example)

        if (len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0): #Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
            return -1

        attr_entropy += shannon_entropy(gain_upper_dataset, classifier)*len(gain_upper_dataset.examples)/total_examples
        attr_entropy += shannon_entropy(gain_lower_dataset, classifier)*len(gain_lower_dataset.examples)/total_examples

    return entropy - attr_entropy    # this is the IG ( Information Gain)

##################################################

# oce entropy
##################################################

def oce_entropy(dataset, classifier):  # off centered entropy
    count,classes=find_classes(dataset.examples, dataset.attributes, classifier)
    total_examples = len(dataset.examples);
    entropy = 0
    probabilities = [count[c] / total_examples for c in classes]
    q=len(probabilities)
    
    if q==2:
        if (probabilities[1]<=theta):
            x=probabilities[1]/(2*theta)
            if (x==1 or x==0):
                entropy+=0
            else:
                entropy+=-x*math.log(x,2)-(1-x)*math.log(1-x,2)
    
        if (probabilities[1]>theta):
            x=(probabilities[1]+1-2*theta)/(2*(1-theta))
            if (x==1 or x==0):
                entropy+=0
            else:
                entropy+=-x*math.log(x,2)-(1-x)*math.log(1-x,2)    
    else:
        for i in range(q):
            if (probabilities[i]<=big_theta[i]):
                x=probabilities[i]/(q*big_theta[i])
                if (x==1 or x==0):
                    entropy+=0
                else:
                    entropy+=-x*math.log(x,2)-(1-x)*math.log(1-x,2)
    
        if (probabilities[i]>big_theta[i]):
            x=(q*(probabilities[i]-big_theta[i])+1-probabilities[i])/(q*(1-big_theta[i]))
            if (x==1 or x==0):
                entropy+=0
            else:
                entropy+=-x*math.log(x,2)-(1-x)*math.log(1-x,2) 
   

    return entropy

##################################################
# Calculate the gain of a particular attribute split
##################################################
def calc_gain_oce(dataset,classifier, entropy, attr_index,datatypes, val=None):
    
    attr_entropy = 0
    total_examples = len(dataset.examples);
    #categorical variable
    if(datatypes[attr_index]==False):
        table=dataset_to_table(dataset)
        li=table[dataset.attributes[attr_index]]
        unique_li=deldup(li)
        datasets=[None for i in range(len(unique_li))]
        for i in range(len(unique_li)):
            datasets[i]=data(classifier)
            datasets[i].attributes=dataset.attributes
            for example in dataset.examples:
                if example[attr_index]==unique_li[i]:
                   datasets[i].examples.append(example)
            attr_entropy+=oce_entropy(datasets[i],classifier)*len(datasets[i].examples)/total_examples
        
    # numerical variable  
    else:
        
        gain_upper_dataset = data(classifier) # instanciate the class data with the parametr classifier
        gain_lower_dataset = data(classifier)
        gain_upper_dataset.attributes = dataset.attributes
        gain_lower_dataset.attributes = dataset.attributes
        gain_upper_dataset.attr_types = dataset.attr_types
        gain_lower_dataset.attr_types = dataset.attr_types
        for example in dataset.examples:
            if (example[attr_index] >= val):
                gain_upper_dataset.examples.append(example)
            elif (example[attr_index] < val):
                gain_lower_dataset.examples.append(example)

        if (len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0): #Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
            return -1

        attr_entropy += oce_entropy(gain_upper_dataset, classifier)*len(gain_upper_dataset.examples)/total_examples
        attr_entropy += oce_entropy(gain_lower_dataset, classifier)*len(gain_lower_dataset.examples)/total_examples

    return entropy - attr_entropy    # this is the IG ( Information Gain)

#########################################
# Determine classes
###################################
classes=[]
def find_classes(instances,attributes,classifier):
    global classes
    class_index = None
    #find index of classifier
    for a in range(len(attributes)):
        if attributes[a] == classifier:
            class_index = a
        else:
            class_index = len(attributes) - 1
    for i in instances:
        if i[class_index] not in classes :
            classes.append(i[class_index])
    count={}
    for c in classes:
        count[c]=0
    for i in instances:
        for c in classes:
            if i[class_index]==c:
                count[c]+=1
    #print (count)
    return  count, classes


##################################################
# Prune tree
##################################################
def prune_tree(root, node, dataset, best_score):
    # if node is a leaf
    if (node.is_leaf == True):
        # get its classification
        classification = node.classification
        # run validate_tree on a tree with the nodes parent as a leaf with its classification
        node.parent.is_leaf = True
        node.parent.classification = node.classification
        if (node.height < 34):
            new_score = validate_tree(root, dataset)
        else:
            new_score = 0
  
        # if its better, change it
        if (new_score >= best_score):
            return new_score
        else:
            node.parent.is_leaf = False
            node.parent.classification = None
            return best_score
    # if its not a leaf
    elif(node.children is None):
        # prune tree(node.upper_child)
        new_score = prune_tree(root, node.upper_child, dataset, best_score)
        # if its now a leaf, return
        if (node.is_leaf == True):
            return new_score
        # prune tree(node.lower_child)
        new_score = prune_tree(root, node.lower_child, dataset, new_score)
        # if its now a leaf, return
        if (node.is_leaf == True):
            return new_score
    else : 
        for i in range(len(node.children)):
            new_score = prune_tree(root, node.children[i], dataset, best_score)
            if (node.is_leaf == True):
                return new_score
    return new_score

##################################################
# Validate tree
##################################################
def validate_tree(node, dataset):
    total = len(dataset.examples)
    correct = 0
    for example in dataset.examples:
        # validate example
        correct += validate_example(node, example)
    return correct/total

##################################################
# Validate example
##################################################
def validate_example(node, example):
    if (node.is_leaf == True):
        projected = node.classification
        actual = example[-1]
        if (projected == actual): 
            return 1
        else:
            return 0
    value = example[node.attr_split_index]
    if (isinstance(node.attr_split_value,Number)):
        if (value >= node.attr_split_value):

            return validate_example(node.upper_child, example)
        else:
            return validate_example(node.lower_child, example)
    for i in range(len(node.attr_split_value)):
        if (value==node.attr_split_value[i]):
            #print(node.children)
            #print(value,node.attr_split_value[i])
            return validate_example(node.children[i], example)
    return 0
    	


##################################################
# Test example
##################################################
def test_example(example, node, class_index):
    if (node.is_leaf == True):
        return node.classification
    else:
        if (example[node.attr_split_index] >= node.attr_split_value):
            return test_example(example, node.upper_child, class_index)
        else:
            return test_example(example, node.lower_child, class_index)

##################################################
# Print tree
##################################################
def print_tree(node):
    if (node.is_leaf == True):
        for x in range(node.height):
            print ("\t"),
        print ("Classification: " + str(node.classification))
        return
    for x in range(node.height):
            print ("\t"),
    print ("Split index: " + str(node.attr_split))
    for x in range(node.height):
            print ("\t"),
    print ("Split value: " + str(node.attr_split_value))
    print_tree(node.upper_child)
    print_tree(node.lower_child)

##################################################
# Print tree in disjunctive normal form
##################################################
def print_disjunctive(node, dataset, dnf_string):
    if (node.parent == None):
        dnf_string = "( "
    if (node.is_leaf == True):
        if (node.classification == 1):
            dnf_string = dnf_string[:-3]
            dnf_string += ") ^^ "
            print (dnf_string,)
        else:
            return
    else:
        upper = dnf_string + str(dataset.attributes[node.attr_split_index]) + " >= " + str(node.attr_split_value) + " V "
        print_disjunctive(node.upper_child, dataset, upper)
        lower = dnf_string + str(dataset.attributes[node.attr_split_index]) + " < " + str(node.attr_split_value) + " V "
        print_disjunctive(node.lower_child, dataset, lower)
        return
##############################################
#isnumber function to use in datatypes

##################################################
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

def main():
    args = str(sys.argv)
    args = ast.literal_eval(args)
    if (len(args) < 2):
        print ("You have input less than the minimum number of arguments. Go back and read README.txt and do it right next time!")
    elif (args[1][-4:] != ".csv"):
        print ("Your training file (second argument must be a .csv!")
    else:
        datafile = args[1]
        dataset = data("")
        training_set=data("")
        test_set=data("")
        training_set = copy.deepcopy(dataset)
        training_set.rows = []
        test_set = copy.deepcopy(dataset)
        read_data(dataset, datafile)
        datatypes=[False for i in range(len(dataset.attributes))]
        #print(datatypes)
        for i in range(len(dataset.examples[0])-1): # we exclude the last column because we do classification so it is set to False by default
            if(is_number(dataset.examples[0][i])):
                datatypes[i]=True
        #print(datatypes)
        arg3 = args[2]
        if (arg3 in dataset.attributes):
            classifier = arg3
        else:
            classifier = dataset.attributes[-1]

        dataset.classifier = classifier
        for a in range(len(dataset.attributes)):
            if dataset.attributes[a] == dataset.classifier:
                dataset.class_index = a
            else:
                dataset.class_index = range(len(dataset.attributes))[-1]
                
        unprocessed = copy.deepcopy(dataset)
        dataset.attr_types=datatypes
        preprocess2(dataset)
        print ("Computing tree...")
        #============================ LOO: Leave One Out. This part will not be executed unless you pass the paramter "-l". Here you are not obliged to pass the K
        ##### It is fixed to the number of instances in your dataset.
        if ("-l" in args):   
            K=len(dataset.examples)

            prepruning_score_oce=0
            postpruning_score_oce=0
            prepruning_score_shannon=0
            postpruning_score_shannon=0
            prepruning_score_oae=0
            postpruning_score_oae=0

            for k in range(K):
                #print ("Doing fold ", k)
                training_set.examples = [x for i, x in enumerate(dataset.examples) if i % K != k]
                test_set.examples = [x for i, x in enumerate(dataset.examples) if i % K == k]
                training_set.attributes=dataset.attributes
                test_set.attributes=dataset.attributes
                training_set.class_index=dataset.class_index
                test_set.class_index=dataset.class_index
                find_theta(training_set)
                # computes roots for each method:
                root_oce = compute_tree_oce(training_set, None, classifier,datatypes)
                root_shannon=compute_tree_shannon(training_set,None,classifier,datatypes)
                root_oae=compute_tree_oae(training_set,None,classifier,datatypes)

                # best scores for each root
                best_score_oce = validate_tree(root_oce, test_set)
                prepruning_score_oce+=best_score_oce

                best_score_shannon = validate_tree(root_shannon, test_set)
                prepruning_score_shannon+=best_score_shannon

                best_score_oae=validate_tree(root_oae,test_set)
                prepruning_score_oae+=best_score_oae



                if ("-p" in args):
                    post_prune_accuracy_oce = 100*prune_tree(root_oce, root_oce, test_set, best_score_oce)
                    postpruning_score_oce+=post_prune_accuracy_oce

                    post_prune_accuracy_shannon = 100*prune_tree(root_shannon, root_shannon, test_set, best_score_shannon)
                    postpruning_score_shannon+=post_prune_accuracy_shannon

                    post_prune_accuracy_oae=100*prune_tree(root_oae,root_oae,test_set,best_score_oae)
                    postpruning_score_oae+=post_prune_accuracy_oce

            print("========================Result for OCE :==============================")
            print ("Initial (pre-pruning) validation set score for LOO is : " + str(100*(prepruning_score_oce/K)) +"%")
            print ("Post-pruning score on validation set for LOO with LOO is : " + str(postpruning_score_oce/K) + "%")


            print("=========================Result for Shannon :==========================")
            print ("Initial (pre-pruning) validation set score for LOO  is : " + str(100*(prepruning_score_shannon/K)) +"%")
            print ("Post-pruning score on validation set for LOO is : " + str(postpruning_score_shannon/K) + "%")

            print("=======================Result for OAE : =============================")
            print ("Initial (pre-pruning) validation set score for LOO is : " + str(100*(prepruning_score_oae/K)) +"%")
            print ("Post-pruning score on validation set for LOO with LOO is : " + str(postpruning_score_oae/K) + "%")


        #============================== Cross validation: will not be executed unless you pass the paramter "-k" followed by the number of folds K
        if ("-k" in args):   
            K=int(args[args.index("-k")+1])

            prepruning_score_oce=0
            postpruning_score_oce=0
            prepruning_score_shannon=0
            postpruning_score_shannon=0
            prepruning_score_oae=0
            postpruning_score_oae=0
            for k in range(K):
                #print ("Doing fold ", k)
                training_set.examples = [x for i, x in enumerate(dataset.examples) if i % K != k]
                test_set.examples = [x for i, x in enumerate(dataset.examples) if i % K == k]
                training_set.attributes=dataset.attributes
                test_set.attributes=dataset.attributes
                training_set.class_index=dataset.class_index
                test_set.class_index=dataset.class_index
                find_theta(training_set)

                # roots
                root_oce = compute_tree_oce(training_set, None, classifier,datatypes)
                root_shannon=compute_tree_shannon(training_set,None,classifier,datatypes)
                root_oae=compute_tree_oae(training_set,None,classifier,datatypes)

                #best scores for roots
                best_score_oce = validate_tree(root_oce, test_set)
                prepruning_score_oce+=best_score_oce

                best_score_shannon = validate_tree(root_shannon, test_set)
                prepruning_score_shannon+=best_score_shannon

                best_score_oae=validate_tree(root_oae,test_set)
                prepruning_score_oae+=best_score_oae

                if ("-p" in args):
                    post_prune_accuracy_oce = 100*prune_tree(root_oce, root_oce, test_set, best_score_oce)
                    postpruning_score_oce+=post_prune_accuracy_oce

                    post_prune_accuracy_shannon = 100*prune_tree(root_shannon, root_shannon, test_set, best_score_shannon)
                    postpruning_score_shannon+=post_prune_accuracy_shannon

                    post_prune_accuracy_oae=100*prune_tree(root_oae,root_oae,test_set,best_score_oae)
                    postpruning_score_oae+=post_prune_accuracy_oae


            print("========================Result for OCE :==============================")
            print ("Initial (pre-pruning) validation set score for", K,"-folds  is : " + str(100*(prepruning_score_oce/K)) +"%")
            print ("Post-pruning score on validation set for",K,"-folds is : " + str(postpruning_score_oce/K) + "%")


            print("=========================Result for Shannon :==========================")
            print ("Initial (pre-pruning) validation set score for",K,"-folds is : " + str(100*(prepruning_score_shannon/K)) +"%")
            print ("Post-pruning score on validation set for ",K,"-folds is : " + str(postpruning_score_shannon/K) + "%")

            print("========================Result for OAE :==============================")
            print ("Initial (pre-pruning) validation set score for", K,"-folds  is : " + str(100*(prepruning_score_oae/K)) +"%")
            print ("Post-pruning score on validation set for",K,"-folds is : " + str(postpruning_score_oae/K) + "%")



        #=========================== Train and validation set separated. This part of code will be executed where the paramters "-k" and "-l" not in args============================================================ 
        if not ("-k" in args or "-l" in args):	
            find_theta(dataset)
	        #if  ("-w" in args):
	         #   global theta
	          #  theta=float(args[args.index("-w")+1])
            prepruning_score_oce=0
            postpruning_score_oce=0
            prepruning_score_shannon=0
            postpruning_score_shannon=0
            prepruning_score_oae=0
            postpruning_score_oae=0
            K=1
            for i in range(K):
                root_oce = compute_tree_oce(dataset, None, classifier,datatypes)
                root_shannon=compute_tree_shannon(dataset,None,classifier,datatypes) 
                root_oae=compute_tree_oae(dataset,None,classifier,datatypes)

                if ("-s" in args):
                    print_disjunctive(root_oce, dataset, "")
                print( "\n")
                if ("-v" in args):
                    datavalidate = args[args.index("-v") + 1]
                    print("Test: "+str(i))
                    print ("Validating tree...")
	    
                    validateset = data(classifier)
                    read_data(validateset, datavalidate)
                    for a in range(len(dataset.attributes)):
                        if validateset.attributes[a] == validateset.classifier:
                            validateset.class_index = a
                        else:
                            validateset.class_index = range(len(validateset.attributes))[-1]
                    validateset.attr_types=datatypes

                    preprocess2(validateset)

                    best_score_oce = validate_tree(root_oce, validateset)
                    prepruning_score_oce+=best_score_oce

                    best_score_shannon = validate_tree(root_shannon, validateset)
                    prepruning_score_shannon+=best_score_shannon

                    best_score_oae=validate_tree(root_oae,validateset)
                    prepruning_score_oae+=best_score_oae
                    if ("-p" in args):
                        post_prune_accuracy_oce = 100*prune_tree(root_oce, root_oce, validateset, best_score_oce)
                        postpruning_score_oce+=post_prune_accuracy_oce

                        post_prune_accuracy_shannon = 100*prune_tree(root_shannon, root_shannon, validateset, best_score_shannon)
                        postpruning_score_shannon+=post_prune_accuracy_shannon

                        post_prune_accuracy_oae=100*prune_tree(root_oae,root_oae,validateset,best_score_oae)
                        postpruning_score_oae+=post_prune_accuracy_oae
            print("========================Result for OCE :==============================")
            print ("Initial (pre-pruning) validation set score is : " + str(100*(prepruning_score_oce/K)) +"%")
            print ("Post-pruning score on validation set  is : " + str(postpruning_score_oce/K) + "%")


            print("=========================Result for Shannon :==========================")
            print ("Initial (pre-pruning) validation set score  is : " + str(100*(prepruning_score_shannon/K)) +"%")
            print ("Post-pruning score on validation set f is : " + str(postpruning_score_shannon/K) + "%")


            print("========================Result for OAE :==============================")
            print ("Initial (pre-pruning) validation set score is : " + str(100*(prepruning_score_oae/K)) +"%")
            print ("Post-pruning score on validation set  is : " + str(postpruning_score_oae/K) + "%")
            """
	                best_score = validate_tree(root, validateset)
	                prepruning_score+=best_score
	            if ("-p" in args):
	                if("-v" not in args):
	                    print ("Error: You must validate if you want to prune")
	                else:
	                    post_prune_accuracy = 100*prune_tree(root, root, validateset, best_score)
	                    print(post_prune_accuracy)
	                    postpruning_score+=post_prune_accuracy 

	        print ("Initial (pre-pruning) validation set score: " + str(100*(prepruning_score/10)) +"%")
	        print ("Post-pruning score on validation set: " + str(postpruning_score/10) + "%")
        """
            
if __name__ == "__main__":
	main()