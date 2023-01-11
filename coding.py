import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Part (a) #
pathreal = 'https://raw.githubusercontent.com/dabandaidai/Data-Storage/main/clean_real.txt'
pathfake = 'https://raw.githubusercontent.com/dabandaidai/Data-Storage/main/clean_fake.txt'


def helper(doc):
    return doc


vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=helper,
    preprocessor=helper,
    token_pattern=None,
    max_features=1000)


def load_data(path1, path2):
    df1 = pd.read_csv(path1, names=['Text'], header=None)
    df1['y'] = 1
    df2 = pd.read_csv(path2, names=['Text'], header=None)
    df2['y'] = 0
    df = pd.concat([df1, df2])
    df['tokens'] = df['Text'].str.split()
    X = vectorizer.fit_transform(df['tokens'].to_numpy())
    y = df['y'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=15 / 85, random_state=1)

    return (X_train, X_val, X_test, y_train, y_val, y_test)


X_train, X_val, X_test, y_train, y_val, y_test = load_data(pathreal, pathfake)

# Part (b) #
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *


# Helper1
def Treepro(depth, c):
    d = DecisionTreeClassifier(criterion=c, splitter='best', random_state=0, max_depth=depth)
    t = d.fit(X_train, y_train)
    return t


# Helper2
def evaluate(tree):
    pred = tree.predict(X_val)
    threshold = 0.5
    as_pred = (pred > threshold).astype(int)
    as_true = y_val
    accuracy = accuracy_score(as_true, as_pred)
    return accuracy


def select_model():
    Trees = []
    max_depth = [3, 5, 12, 25, 48]
    cri = ['gini', 'entropy', 'log_loss']
    for depth in max_depth:
        for c in cri:
            Trees.append((Treepro(depth, c), depth, c))
    acc = []
    max = []
    cri = []
    trees = []
    output = ""
    for tree in Trees:
        accuracy = evaluate(tree[0])
        trees.append(tree[0])
        acc.append(accuracy)
        max.append(tree[1])
        cri.append(tree[2])
        output += "The tree built with criterion {cri} and maximum depth {max} has accuracy {acc} \n".format(
            cri=tree[2],
            max=tree[1],
            acc=accuracy)
    return trees, acc, max, cri, output


trees, acc, max, cri, output = select_model()
print(output)

## Plot ##
plt.scatter(max, acc, alpha=0.5)
plt.title("Scatterplot between accuracy and maximum depth of decision trees")
plt.xlabel("maximum depth")
plt.ylabel("accuracy")
plt.show()

## Part (c) ##

from sklearn import tree


def max_value(lst):
    maximum = 0
    index = 0
    for i in range(len(lst)):
        if lst[i] > maximum:
            maximum = lst[i]
            index = i
    return index


# Finding Maximum accuracy tree and plotting #
mi = max_value(acc)
best = trees[mi]
fig = plt.figure(figsize=(12, 8))
_ = tree.plot_tree(best, max_depth=2, feature_names=vectorizer.get_feature_names(), class_names=str(y_train),
                   filled=True, rounded=True)
plt.title("Graph of the best accuracy decision trees")
plt.show()

## Part (d) ##

import math


def entropycalc(py):
    py1 = py
    py0 = 1 - py1
    loga0 = math.log(py0, 2)
    loga1 = math.log(py1, 2)
    hy = -(py0 * loga0 + py1 * loga1)
    return hy


def compute_information_gain(Y, X, x: str):
    global index
    words = vectorizer.get_feature_names()
    ## Calculate H(y) ##
    total = len(Y)
    num1 = sum(Y)
    p_y1 = num1 / total
    hy = entropycalc(p_y1)
    ## Calculate H(y|x) ##
    # Find index of x #
    for i in range(len(words)):
        if words[i] == x:
            index = i
    dense = X.todense()
    # Find the column for x #
    all_x = dense[:, index]
    # Find rows where x present and miss #
    l1 = y_train[np.where(all_x <= 0.5)[0]]
    l2 = y_train[np.where(all_x > 0.5)[0]]
    p_x1 = len(l1) / total  # probability of containing x #
    p_x0 = len(l2) / total  # probability of not containing x #
    prob_y1 = sum(l1) / len(l1)
    prob_y0 = sum(l2) / len(l2)
    hy0 = entropycalc(prob_y0)
    hy1 = entropycalc(prob_y1)
    hy_x = p_x1 * hy1 + p_x0 * hy0
    return hy - hy_x


print("The information gain by splitting on 'the' is {the}, \nthe information gain by splitting on 'donald' is "
      "{donald}, \n the information gain by splitting on 'trumps' is {trumps}, \nthe information gain by splitting on "
      "'hillary' is {hillary}, \nthe information gain by splitting on "
      "'le' is {le}, \nthe information gain by splitting on "
      "'market' is {market}.".format(
    the=compute_information_gain(y_train, X_train, 'the'),
    donald=compute_information_gain(y_train, X_train, 'donald'),
    trumps=compute_information_gain(y_train, X_train, 'trumps'),
    hillary=compute_information_gain(y_train, X_train, 'hillary'),
    le=compute_information_gain(y_train, X_train, 'le'),
    market=compute_information_gain(y_train, X_train, 'market'),
    )
)
