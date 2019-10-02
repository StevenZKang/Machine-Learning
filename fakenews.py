import numpy
import matplotlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
import math

#Load data into a tokenized matrix 
def load_data(fakenewsfile: str, realnewsfile: str):
    fakenews = open(fakenewsfile, 'r')
    realnewsfile = open(realnewsfile, 'r')
    fakelist = list(fakenews)
    reallist = list(realnewsfile)

    vector = CountVectorizer()
    x = vector.fit_transform(reallist + fakelist)
    y = (len(reallist) * [0]) + (len(fakelist) * [1])

    x_train, x_other, y_train, y_other = train_test_split(x, y, train_size = 0.7) 
    x_test, x_validate, y_test, y_validate = train_test_split(x_other, y_other, train_size = 0.5)
    
    return [x_train, y_train, x_validate, y_validate, x_test, y_test]
    

def select_model(x_train, y_train, x_validate, y_validate):
    """
    Train multiples models at different depths using both 
    information gain and Gini coefficient. 
    Evaluate performance of each on the validation set. 
    """
    #Train models
    scores = []
    for i in range(40, 220, 20):
        clf = DecisionTreeClassifier(criterion = 'gini', max_depth = i, splitter = 'best')
        clf.fit(x_train, y_train)

        #validate
        score = clf.score(x_validate, y_validate)
        param = (score , 'gini', i)
        print(param)
        scores.append(param)

        clf2 = DecisionTreeClassifier(criterion = 'entropy', max_depth = i, splitter = 'best')
        clf2.fit(x_train, y_train)

        #validate
        score = clf.score(x_validate, y_validate)
        param = (score, 'entropy', i)
        print(param)
        scores.append(param)
    
    best_tuple = max(scores)
    print("Best model:", best_tuple)

    return [best_tuple[1], best_tuple[2]]

def visualize_tree(x_train, y_train, splitter, depth):
    """ """
    #Build Best Tree
    model = DecisionTreeClassifier(criterion=splitter, max_depth=depth)
    model.fit(x_train, y_train)
    #Visualize
    dot_data = StringIO()
    
    tree.export_graphviz(model, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    graph.write_png("treeplot.png")


def compute_information_gain(x_train, y_train, index):
    """Compute IG of a split"""
    x_train = x_train.toarray()
    total = len(y_train)
    num_fake = y_train.count(1)
    num_real = y_train.count(0)
    
    root_H = (num_fake/total* -1) * math.log(num_fake/total) - (num_real/total) * math.log(num_real/total) 


    true_fake, true_real, false_fake, false_real = 0, 0, 0, 0
    for i in range(len(y_train)):
        #True(Left, Doesn't exist)
        if x_train[i][index] <= 0.5:
            if y_train[i] == 1: 
                true_fake += 1
            else:
                true_real += 1
        #False(right, Exists)
        else:
            if y_train[i] == 1: 
                false_fake += 1
            else:
                false_real += 1
    true_total = true_fake + true_real
    false_total = false_fake + false_real 

    left_H = (true_fake/true_total * -1) * math.log(true_fake/true_total) - (true_real/true_total) * math.log(true_real/true_total) 
    left_H = left_H * (true_total/total)
    right_H = (false_fake/false_total* -1) * math.log(false_fake/false_total) - (false_real/false_total) * math.log(false_real/false_total) 
    right_H = right_H * (false_total/total)

    return root_H - (left_H + right_H)

if __name__== "__main__":
    x_train, y_train, x_validate, y_validate, x_test, y_test = load_data('clean_fake.txt', 'clean_real.txt')

    splitter, depth = select_model(x_train, y_train, x_validate, y_validate)
    print("Information gain on 5142", compute_information_gain(x_train, y_train, 5143))
    print("Information gain on 2405", compute_information_gain(x_train, y_train, 2405))
    print("Information gain on 1598", compute_information_gain(x_train, y_train, 1598))
    print("Information gain on 5324", compute_information_gain(x_train, y_train, 5324))
    
    clf = DecisionTreeClassifier(criterion=splitter, max_depth=depth)
    clf.fit(x_train,y_train)
    #print(tree.plot_tree(clf))
    
    #visualize_tree(x_train, y_train, splitter, depth)
    

