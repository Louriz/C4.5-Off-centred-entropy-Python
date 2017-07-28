# This is a modification of the code C4.5 made by : Rayan Madden and Ally Cody.
# The main in changes in the code are: Adding the OCE ( Off-Centred Entropy), OAE, Cross validation, LOO(Leave-One-Out),multiclass dealing and finding datatypes using dataset. The purpose of those changes is to deal with imbalance of modalities in the class. Here we have done it just for a binary class, but the code can be extended to multiple values class.
# This modification is made by:,Ali MesbahiRochd Maliki and Riahi Louriz during the internship at IMT Atlantique.
#
# decision-tree
A C4.5 Decision Tree python implementation with validation, pruning, and attribute multi-splitting
Contributors: Ryan Madden and Ally Cody

## Requirements
python 2.7.6 [Download](https://www.python.org/download/releases/2.7.6/)

## Files
* many files (.csv) are available to test our solution.
* oce-shannon-v6.py - The decision tree program

## How to run
oce-shannon-v6.py accepts parameters passed via the command line. The possible paramters are:
* Filename for training (Required, must be the first argument after 'python oce-shannon-v6.py')
* Classifier name (Optional, by default the classifier is the last column of the dataset)
* Print flag (-s) (Optional, causes the dataset)
* Validate flag (-v) followed by validate filename (Optional, specifies file to use for validation)
* Pruning flag (-p) (Optional, you must include a validation file in order to prune)
* Cross validation flag (-k) you must add the fold size 
* LOO(leave one out) flag (-l) ( you do not need to specify the folds, it is equal to size of data)

#### Examples

#####Example 1
```
python oce-shannon-v6.py btrain.csv -v bvalidate.csv -p 
```
This command runs oce-shannon-v6.py with btrain.csv as the training set, bvalidate.csv as the validation set and pruning enabled. The classifier is not specified so it defaults to the last column in the training set. Printing is not enabled.
#####Example 2
```
python oce-shannon-v6.py yeast.csv -k 10 -p

```

This command runs oce-shannon-v6.py with yeast.csv dataset. Here we apply the cross validation with 10 folds. Pruning is enabled. 

#####Example 3
```
python oce-shannon-v6.py yellow-small.csv -l -p
```
This command runs oce-shannon-v6.py  with yellow-small.csv dataset. Here we apply LOO( leave one out). Pruning is enabled 

