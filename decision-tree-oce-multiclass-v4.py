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


def is_mono(t):
    """ Returns True if all values of _t_ are equal
        and False otherwise.
    """
    for i in t:
        if i != t[0]:
            return False
    return True


def get_indexes(table, col, v):
    """ Returns indexes of values _v_ in column _col_
        of _table_.
    """
    li = []
    start = 0
    for row in table[col]:
        if row == v:
            index = table[col].index(row, start)
            li.append(index)
            start = index + 1
    return li


def get_values(t, col, indexes):
    """ Returns values of _indexes_ in column _col_
        of the table _t_.
    """
    return [t[col][i] for i in range(len(t[col])) if i in indexes]


def del_values(t, ind):
    """ Creates the new table with values of _ind_.
    """
    return {k: [v[i] for i in range(len(v)) if i in ind] for k, v in t.items()}


def print_list_tree(tree, tab=''):
    """ Prints list of nested lists in
        hierarchical form.
    """
    print('%s[' % tab)
    for node in tree:
        if isinstance(node, basestring):
            print('%s  %s' % (tab, node))
        else:
            print_list_tree(node, tab + '  ')
    print('%s]' % tab)


def formalize_rules(list_rules):
    """ Gives an list of rules where
        facts are separeted by coma.
        Returns string with rules in
        convinient form (such as
        'If' and 'Then' words, etc.).
    """
    text = ''
    for r in list_rules:
        t = [i for i in r.split(',') if i]
        text += 'If %s,\n' % t[0]
        for i in t[1:-1]:
            text += '   %s,\n' % i
        text += 'Then: %s.\n' % t[-1]
    return text


def get_subtables(t, col):
    """ Returns subtables of the table _t_
        divided by values of the column _col_.
    """
    return [del_values(t, get_indexes(t, col, v)) for v in deldup(t[col])]



def freq(table, col, v):
    """ Returns counts of variant _v_
        in column _col_ of table _table_.
    """
    return table[col].count(v)


def info(table, res_col):
    """ Calculates the entropy of the table _table_
        where res_col column = _res_col_.
    """
    s = 0 # sum
    for v in deldup(table[res_col]):
        p = freq(table, res_col, v) / float(len(table[res_col]))
        s += p * math.log(p, 2)
    return -s


def infox(table, col, res_col):
    """ Calculates the entropy of the table _table_
        after dividing it on the subtables by column _col_.
    """
    s = 0 # sum
    for subt in get_subtables(table, col):
        s += (float(len(subt[col])) / len(table[col])) * info(subt, res_col)
    return s
    


def gain(table, x, res_col):
    """ The criterion for selecting attributes for splitting.
    """
    return info(table, res_col) - infox(table, x, res_col)

# ======================== end categorical entropy
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
def read_data(dataset, datafile, datatypes):
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
    dataset.attr_types=['false','true','false','false','true','false','false','true','false','false','true','false','true','false','false','true','false','true','false','false','false']


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
                if dataset.attr_types[x]=='true':
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
        self.height = None

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
        # tree = compute_tree(dv)
        # attach tree to the corresponding branch of Tree
    # return tree 

def compute_tree(dataset, parent_node, classifier,datatypes):
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
    split_val = None 
    min_gain = 0.01
    dataset_entropy = calc_dataset_entropy(dataset, classifier)
    for attr_index in range(len(dataset.examples[0])):

        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in dataset.examples] # these are the values we can split on, now we must find the best one
            attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values
            if(len(attr_value_list) > 100):
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
                local_gain = calc_gain(dataset, classifier, dataset_entropy, val, attr_index,datatypes) # calculate the gain if we split on this value
  
                if (local_gain >= local_max_gain):
                    local_max_gain = local_gain
                    local_split_val = val

            if (local_max_gain >= max_gain):
                max_gain = local_max_gain
                split_val = local_split_val
                attr_to_split = attr_index

    #attr_to_split is now the best attribute according to our gain metric
    if (split_val is None or attr_to_split is None):
        print( "Something went wrong. Couldn't find an attribute to split on or a split value.")
    elif (max_gain <= min_gain or node.height > 20):

        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node

    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_to_split]
    node.attr_split_value = split_val
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

    node.upper_child = compute_tree(upper_dataset, node, classifier,datatypes)
    node.lower_child = compute_tree(lower_dataset, node, classifier,datatypes)

    return node

##################################################
# Classify dataset
##################################################
def classify_leaf(dataset, classifier):
    count, classes = find_classes(dataset.examples, dataset.attributes, classifier)  # count number of classied as '1'
    max=0
    for c in classes:
        if count[c]>=max:
            max=count[c]
            cfin=c
    return cfin
    #total = len(dataset.examples)
    #zeroes = total - ones
    #if (ones >= zeroes):
    #    return 1
    #else:
    #    return 0
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
    print(theta,big_theta)
######  


##################################################
# Calculate the entropy of the current dataset : Entropy(dataset)= - Sigma( pi*log2(pi)) i=1:k  , k is the number of modalities in the class
##################################################
def calc_dataset_entropy(dataset, classifier):  # off centered entropy
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
def calc_gain(dataset,classifier, entropy, val, attr_index,datatypes):
    table=dataset_to_table(dataset)
    #classifier = dataset.attributes[attr_index]
    attr_entropy = 0
    if(datatypes[attr_index]=='false'):
        attr_entropy+=infox(table,dataset.attributes[attr_index],dataset.attributes[len(dataset.attributes)-1])
        #datasets=[]
        #li=table[datatset.attributes[attr_index]]
        #unique_li=deldup(li)
        #for i in range(len(unique_li)):
        #	datasets[i]=data()

        
        
    else:
        total_examples = len(dataset.examples);
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

        attr_entropy += calc_dataset_entropy(gain_upper_dataset, classifier)*len(gain_upper_dataset.examples)/total_examples
        attr_entropy += calc_dataset_entropy(gain_lower_dataset, classifier)*len(gain_lower_dataset.examples)/total_examples

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
        if (node.height < 20):
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
    else:
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
    if (value >= node.attr_split_value):
        return validate_example(node.upper_child, example)
    else:
        return validate_example(node.lower_child, example)

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

##################################################
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
        test_set.rows = []

        if ("-d" in args):
            datatypes = args[args.index("-d") + 1]
        else:
            datatypes = 'datatypes.csv'
        datatypes=['false','false','false','false','true','false']  # yellow
        #datatypes=['true','false','true','false','false','false','false','false','false','false','true','true','true','false','false'] # income
        #datatypes=['true','true','true','true','true','true','true','true','true','true','true','true','true','false'] # wine
        #datatypes=['true','true','true','true','true','true','true','true','true','false'] #  glass
        #datatypes=['false','true','false','false','true','false','false','true','false','false','true','false','true','false','false','true','false','true','false','false','false']





        read_data(dataset, datafile, datatypes)
        print("hereeeee", datatypes[4])
        print(isinstance(dataset.examples[0][4],str))
        print(dataset.examples[0][4].isdigit())
        #print(dataset.examples)
        #print(dataset_to_table(dataset))
        #table=dataset_to_table(dataset)
        #print(" here it is ",info(table,"class"))

        arg3 = args[2]
        if (arg3 in dataset.attributes):
            classifier = arg3
        else:
            classifier = dataset.attributes[-1]

        dataset.classifier = classifier
        #print("here again",calc_dataset_entropy(dataset,classifier))
        #table=dataset_to_table(dataset)
        #attr_entropy=infox(table,dataset.attributes[1],dataset.attributes[len(dataset.attributes)-1])
        #print("I am here ",attr_entropy)
        #attr_value_list = [example[0] for example in dataset.examples] # these are the values we can split on, now we must find the best one
        #attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values
        #print(attr_value_list)
        #find index of classifier
        for a in range(len(dataset.attributes)):
            if dataset.attributes[a] == dataset.classifier:
                dataset.class_index = a
            else:
                dataset.class_index = range(len(dataset.attributes))[-1]
                
        unprocessed = copy.deepcopy(dataset)
        print(dataset.examples[0])
        dataset.attr_types=datatypes
        preprocess2(dataset)
        print ("Computing tree...")
        #============================ LOO: Leave One Out. This part will not be executed unless you pass the paramter "-l". Here you are not obliged to pass the K
        ##### It is fixed to the number of instances in your dataset.
        if ("-l" in args):   
            K=len(dataset.examples)

            prepruning_score=0
            postpruning_score=0
            for k in range(K):
                print ("Doing fold ", k)
                training_set.examples = [x for i, x in enumerate(dataset.examples) if i % K != k]
                test_set.examples = [x for i, x in enumerate(dataset.examples) if i % K == k]
                training_set.attributes=dataset.attributes
                test_set.attributes=dataset.attributes
                training_set.class_index=dataset.class_index
                test_set.class_index=dataset.class_index
                find_theta(training_set)
                #print(find_theta(training_set))
                print ("Number of training records: %d" % len(training_set.examples))
                #print(training_set.examples)
                print ("Number of test records: %d" % len(test_set.examples))
                #print(test_set.examples)
                root = compute_tree(training_set, None, classifier,datatypes)
                best_score = validate_tree(root, test_set)
                prepruning_score+=best_score
                if ("-p" in args):
                    post_prune_accuracy = 100*prune_tree(root, root, test_set, best_score)
                    postpruning_score+=post_prune_accuracy

            print ("Initial (pre-pruning) validation set score for LOO: " + str(100*(prepruning_score/K)) +"%")
            print ("Post-pruning score on validation set for LOO: " + str(postpruning_score/K) + "%")

        #============================== Cross validation: will not be executed unless you pass the paramter "-k" followed by the number of folds K
        if ("-k" in args):   
            K=int(args[args.index("-k")+1])

            prepruning_score=0
            postpruning_score=0
            for k in range(K):
                print ("Doing fold ", k)
                training_set.examples = [x for i, x in enumerate(dataset.examples) if i % K != k]
                test_set.examples = [x for i, x in enumerate(dataset.examples) if i % K == k]
                training_set.attributes=dataset.attributes
                test_set.attributes=dataset.attributes
                training_set.class_index=dataset.class_index
                test_set.class_index=dataset.class_index
                find_theta(training_set)
                #print(find_theta(training_set))
                print ("Number of training records: %d" % len(training_set.examples))
                #print(training_set.examples)
                print ("Number of test records: %d" % len(test_set.examples))
                #print(test_set.examples)
                root = compute_tree(training_set, None, classifier,datatypes)
                best_score = validate_tree(root, test_set)
                prepruning_score+=best_score
                if ("-p" in args):
                    post_prune_accuracy = 100*prune_tree(root, root, test_set, best_score)
                    postpruning_score+=post_prune_accuracy

            print ("Initial (pre-pruning) validation set score for cross validatio: " + str(100*(prepruning_score/K)) +"%")
            print ("Post-pruning score on validation set for cross validation: " + str(postpruning_score/K) + "%")

        #=========================== Train and validation set separated. This part of code will be executed where the paramters "-k" and "-l" not in args============================================================ 
        if not ("-k" in args or "-l" in args):	
	        find_theta(dataset)
	        if  ("-w" in args):
	            global theta
	            theta=float(args[args.index("-w")+1])
	        prepruning_score=0
	        postpruning_score=0
	        for i in range(1):

	            root = compute_tree(dataset, None, classifier,datatypes) 
	    
	            if ("-s" in args):
	                print_disjunctive(root, dataset, "")
	            print( "\n")
	            if ("-v" in args):
	                datavalidate = args[args.index("-v") + 1]
	                print("Test: "+str(i))
	                print ("Validating tree...")
	    
	                validateset = data(classifier)
	                read_data(validateset, datavalidate, datatypes)
	                for a in range(len(dataset.attributes)):
	                    if validateset.attributes[a] == validateset.classifier:
	                        validateset.class_index = a
	                    else:
	                        validateset.class_index = range(len(validateset.attributes))[-1]
	                preprocess2(validateset)
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
        
            
if __name__ == "__main__":
	main()