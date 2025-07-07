import csv
import numpy as np
import random
import unittest
import time
import seaborn as sns 
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import combinations

class MyTest(unittest.TestCase):

    def setUp(self):
        
        self.X = np.array([
            [1, 10, 5, 100],
            [2, 20, 10, 200],
            [2, 20, 5, 100],
            [3, 30, 15, 300],
            [3, 30, 15, 300],
            [3, 30, 5, 100]
        ])

        self.y = np.array([0, 0, 1, 1, 0, 1])

        self.X_1 = np.array([[2, 20, 10, 200],
                            [2, 20, 10, 200],
                            [2, 20, 10, 200]]) 
        self.y_1 = np.array([0, 0, 1])
    
    def test_tree_splits(self):

        t = Tree(rand = None, get_candidate_columns=all_columns)
        tree_model = t.build(self.X, self.y)
        
        # check first split should be feature_1 value:1
        self.assertEqual(tree_model.feature, 0)
        self.assertEqual(tree_model.threshold, 1.5)

        # check left_subtree is pure and class 0 
        left_subtree = tree_model.left 
        self.assertEqual(left_subtree.prediction, 0)
        
        # check right_subtree is splited by feature_3 value:5
        right_subtree = tree_model.right
        self.assertEqual(right_subtree.feature, 2)
        self.assertEqual(right_subtree.threshold, 7.5)

        # check right_left_subtree is pure and class 1 
        right_left_subtree = right_subtree.left
        self.assertEqual(right_left_subtree.prediction, 1)

        # check right_right_subtree 
        right_right_subtree = right_subtree.right
        self.assertEqual(right_right_subtree.feature, 0)
        self.assertEqual(right_right_subtree.threshold, 2.5)

        # check right_right_left subtree is pure and class 0 
        right_right_left_subtree = right_right_subtree.left
        self.assertEqual(right_right_left_subtree.prediction, 0)

        # check right_right_right subtree predicts class 0
        right_right_right_subtree = right_right_subtree.left
        self.assertEqual(right_right_left_subtree.prediction, 0)

    
    def test_tree_edge_case1(self):
         
         # properly handle when multiple rows have same feature values, but different classes 
         # avoid infinite loop
         t = Tree(rand=None, get_candidate_columns=all_columns)
         tree_model = t.build(self.X_1, self.y_1)
         self.assertEqual(tree_model.prediction, 0)
    

    
def all_columns(X, rand):
    return range(X.shape[1])

def random_sqrt_columns(X, rand):
    num_features = int(np.sqrt(X.shape[1]))
    return  rand.sample(range(X.shape[1]), num_features)

class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples
        self.tree_features = []
        self.features_depth = []
        
    def build(self, X, y, depth = 0):
        
        if len(y) < self.min_samples or len(np.unique(y)) == 1: # stop conditions: pure node or not enough samples
            # return the majority class
            #if len(y) == 0:
                #return TreeModel(prediction=0) # this should not happen ??
            return TreeModel(prediction=np.argmax(np.bincount(y)))

    
        if np.all(X == X[0]):  
            return TreeModel(prediction = np.argmax(np.bincount(y)))
        
        ix_samples = self.get_candidate_columns(X, self.rand)
        # print(ix_samples)
        X_sampled = X[:, ix_samples]
            
        relative_feature_index, threshold = self.find_best_split(X_sampled, y)
        
        if relative_feature_index is None: 
            return TreeModel(prediction=np.argmax(np.bincount(y)))
        
        feature_index = ix_samples[relative_feature_index]
        
        self.tree_features.append(feature_index)
        self.features_depth.append(depth)
        
        X_left, y_left = X[X[:, feature_index] <= threshold], y[X[:, feature_index] <= threshold]
        X_right, y_right = X[X[:, feature_index] > threshold], y[X[:, feature_index] > threshold]
       
        left_tree = self.build(X_left, y_left, depth + 1)
        right_tree = self.build(X_right, y_right, depth + 1)
        
        return TreeModel(feature = feature_index, threshold= threshold, left= left_tree, right= right_tree, tree_features= self.tree_features, depth = depth + 1, feature_depths=self.features_depth)  # return an object that can do prediction
        

    def find_best_split(self, X, y): 


        n_samples, n_features = X.shape 
        best_feature = None
        best_threshold = None 
        best_gini = float('inf')

        for i in range(n_features):

            feature = X[:, i]
            sort_idx = np.argsort(feature)
            feature_sorted = feature[sort_idx]
            y_sorted = y[sort_idx]

            cumulative_ones = np.cumsum(y_sorted)
            total_ones = cumulative_ones[-1]

            left_counts = np.arange(1, n_samples + 1)
            right_counts = n_samples - left_counts

            p_left_ones = cumulative_ones / left_counts
            p_left_zeros = 1 - p_left_ones
            gini_left = 1 - p_left_ones**2 - p_left_zeros**2

            valid_right_ix = right_counts > 0
            p_right_ones = np.zeros(n_samples)
            p_right_ones[valid_right_ix] = (total_ones - cumulative_ones[valid_right_ix]) / right_counts[valid_right_ix]
            p_right_zeros = 1 - p_right_ones
            gini_right = 1 - p_right_ones**2 - p_right_zeros**2
            
            gini_right[right_counts == 0] = 0
            
            gini = left_counts / n_samples * gini_left +  right_counts / n_samples * gini_right
    
            valid = np.where(feature_sorted[:-1] != feature_sorted[1:])[0]

            if valid.size > 0:
                candidate_gini = gini[valid]
                idx = valid[np.argmin(candidate_gini)]
                if candidate_gini[np.argmin(candidate_gini)] < best_gini:
                    best_gini = candidate_gini[np.argmin(candidate_gini)]
                    best_feature = i
                    # choose threshold as the midpoint between neighboring unique values
                    # better generalization, to unseen data
                    best_threshold = (feature_sorted[idx] + feature_sorted[idx + 1]) / 2.0

        return best_feature, best_threshold

    
 
class TreeModel:

    def __init__(self, feature = None, threshold = None, left = None, right = None, prediction=None, tree_features = None, depth = None, feature_depths = None):
        
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.tree_features = tree_features
        self.features_depth = feature_depths
        self.depth = depth 
        
    def predict(self, X):
        y_pred = [self.predict_single(x) for x in X]
        return np.array(y_pred)
    
    def predict_single(self, x):
        if self.prediction is not None:
            return self.prediction
        if x[self.feature] < self.threshold:
            return self.left.predict_single(x)
        else:
            return self.right.predict_single(x)

class RandomForest:

    def __init__(self, rand=None, n=100, get_candidate_columns = random_sqrt_columns):
        self.n = n
        self.rand = rand
        self.trees = []
        self.oob_indicies = []
        self.get_candidate_columns = get_candidate_columns

    def build(self, X, y):

        for _ in range(self.n):
            
            
            idx = self.rand.choices(range(len(y)), k=len(y)) # sample with replacement len(y) instances 
            X_boot = X[idx, :]
            y_boot = y[idx]
            
            oob_indicies_tree = np.nonzero(~np.isin(np.arange(len(y)), idx))[0]


            self.oob_indicies.append(oob_indicies_tree)

            tree = Tree(rand=self.rand, get_candidate_columns = self.get_candidate_columns)
            tree = tree.build(X_boot, y_boot)
            self.trees.append(tree)
        
        return RFModel(self.trees, self.oob_indicies, X, y, self.rand)  # return an object that can do prediction

class RFModel:

    def __init__(self, trees, oob_indicies, X, y, rand=None):
        self.trees = trees
        self.oob_indicies = oob_indicies
        self.X = X
        self.y = y
        self.rand = rand
        self.baseline_accs = []

    def predict(self, X):
        
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # majority voting
        y_pred = np.array([np.bincount(col).argmax() for col in tree_preds.T])
        return y_pred
        

    def importance(self):
        
        n_features = self.X.shape[1]
        importances = np.zeros(n_features)        
        
        # for each tree caluclate baseline accuracy 
        for i, (tree, oob_ix) in enumerate(zip(self.trees, self.oob_indicies)):
        
            if len(oob_ix) == 0:
                continue
            
            X_oob = self.X[oob_ix]
            y_oob = self.y[oob_ix]
            y_pred = tree.predict(X_oob)
            acc = np.mean(y_pred == y_oob)
            self.baseline_accs.append(acc)
        
        self.baseline_accs = np.array(self.baseline_accs)
        
        # drop in accuracy by permuting single feature 
        # OPTIMIZATION: for each tree permute only the values of the feature that construct the tree 
        # features_occ = np.zeros(n_features)
        for i, (tree, oob_ix) in enumerate(zip(self.trees, self.oob_indicies)): 
            
            if len(oob_ix) == 0:
                continue
            X_oob = self.X[oob_ix]
            y_oob = self.y[oob_ix]
            
            
            for feature_ix in tree.tree_features:
               
               X_oob_permuted = X_oob.copy()
               self.rand.shuffle(X_oob_permuted[:, feature_ix])
               y_pred_permuted = tree.predict(X_oob_permuted)
               acc_permuted = np.mean(y_pred_permuted == y_oob)
               importances[feature_ix] += self.baseline_accs[i] - acc_permuted

        importances /= len(self.trees)
        return importances
    
    
    def importance3(self):

        n_features = self.X.shape[1]
        importances = np.zeros((n_features, n_features, n_features))
        baseline_accs = []
        
        # estimate baseline accuracies 
        for i, (tree, oob_ix) in enumerate(zip(self.trees, self.oob_indicies)):
            
            if len(oob_ix) == 0:
                continue
            
            X_oob = self.X[oob_ix]
            y_oob = self.y[oob_ix]
            y_pred = tree.predict(X_oob)
            acc = np.mean(y_pred == y_oob)
            baseline_accs.append(acc)
        
        baseline_accs = np.array(baseline_accs)
        
        # drop in accuracy by permuting triplets
        for i, (tree, oob_ix) in enumerate(zip(self.trees, self.oob_indicies)):
            
            if len(oob_ix) == 0:
                continue
                
            X_oob = self.X[oob_ix]
            y_oob = self.y[oob_ix]
            # make triplets from all features in the tree
            tree_features = list(set(tree.tree_features))  
            feature_triplets = list(combinations(tree_features, 3))

            for f1, f2, f3 in feature_triplets:
                
                    X_oob_permuted = X_oob.copy()
                    self.rand.shuffle(X_oob_permuted[:, f1])
                    self.rand.shuffle(X_oob_permuted[:, f2])
                    self.rand.shuffle(X_oob_permuted[:, f3])
                    y_pred_permuted = tree.predict(X_oob_permuted)
                    acc_permuted = np.mean(y_pred_permuted == y_oob)
                    importances[f1, f2, f3] += baseline_accs[i] - acc_permuted  

        importances /= len(self.trees) 
        return importances

            
    def importance3_structure(self): 
        
        triplet_count = {}
        triplet_depth = {}
        max_count = 0
        best_comb = tuple()
        avg_min_depth = 0

        for tree in self.trees: 
            curr_features = tree.tree_features
            curr_depths = tree.features_depth
            feature_to_index = {feature: i for i, feature in enumerate(curr_features)}

            for comb in combinations(curr_features, 3):
                # comb = tuple(sorted(comb))  
                comb_ix = [feature_to_index[f] for f in comb]
                depths = [curr_depths[ix] for ix in comb_ix]

                triplet_count[comb] = triplet_count.get(comb, 0) + 1
                triplet_depth[comb] = triplet_depth.get(comb, 0) +  sum(depths) / 3
                    
     
        triplets = np.array(list(triplet_count.keys()))
        counts = np.array(list(triplet_count.values()))
        depths = np.array(list(triplet_depth.values()))
        depths = depths / counts
        max_count = counts.max()
        avg_max_depth = depths.max()
 
        count_norm = counts / max_count 
        depth_norm = 1 - depths / avg_max_depth
        scores = 0.2 *count_norm + 0.8* depth_norm 
        best_ix = np.argmax(scores)
        
        
        #print(best_ix)
        #print(depths[best_ix])
        #print(counts[best_ix])
        #print(triplets[best_ix])
        
        return triplets[best_ix]

    

def read_tab(fn, adict):
    
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))
    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y

 
def tki():
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


def hw_tree_full(learn_data, test_data):
    
    X_learn, y_learn = learn_data
    X_test, y_test = test_data
    
    tree = Tree(rand=random.Random(0))
    model = tree.build(X_learn, y_learn)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_learn)
    
    # calculate misclassification rate
    mean_msr_train, se_msr_train = binomial_se(y_pred=y_pred_train, y_true = y_learn)
    mean_msr_test, se_msr_test = binomial_se(y_pred = y_pred_test, y_true = y_test)

    return (mean_msr_train, se_msr_train), (mean_msr_test, se_msr_test)
    

def hw_randomforests(learn_data, test_data):
    
    X_learn, y_learn = learn_data
    X_test, y_test = test_data
    
    rf = RandomForest(rand=random.Random(0), n=100, get_candidate_columns= random_sqrt_columns)
    model = rf.build(X_learn, y_learn)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_learn)
    
    # calculate misclassification rate
    mean_msr_train, se_msr_train = bootstrap_se(y_pred=y_pred_train, y_true = y_learn)
    mean_msr_test, se_msr_test = bootstrap_se(y_pred = y_pred_test, y_true = y_test)

    return (mean_msr_train, se_msr_train), (mean_msr_test, se_msr_test)
# qunatiy the uncertainty of misslcalssification
    
def binomial_se(y_pred, y_true ): 
    # assuming each error_prediction is Benroulli i.i.d 
    n_samples = len(y_pred)
    mean_msr = np.sum(y_pred != y_true) / n_samples
    se_msr = np.sqrt((mean_msr * (1 - mean_msr)) / n_samples)
    return mean_msr, se_msr

def bootstrap_se(y_pred, y_true):
    
    # bootstrapping misclassification rate to estimate standard error
    np.random.seed(0)
    error_vec = (y_pred-y_true)**2
    num_bootsamples = 100
    msr_estimates = []
    n_samples = len(y_pred)
    for _ in range(num_bootsamples):
        
        error_boot = np.random.choice(error_vec, size=n_samples, replace=True)
        msr_i = np.mean(error_boot)
        msr_estimates.append(msr_i)
    
    mean_msr = np.mean(msr_estimates)
    se_msr = np.sqrt(np.sum((msr_estimates - mean_msr)**2) / len(msr_estimates))
    
    return mean_msr, se_msr

def plot_misclassification_rates(learn_data, test_data, tree_range = 100): 
    
    
    X_learn, y_learn = learn_data
    X_test, y_test = test_data
    misclassification_rates = []
    standard_errors = []

    for i in range(1, tree_range + 1):

        rf = RandomForest(rand=random.Random(0), n = i)
        model = rf.build(X_learn, y_learn)
        y_pred = model.predict(X_test)
        misclassification_rate_i = np.mean(y_pred != y_test)
        misclassification_rates.append(misclassification_rate_i)
        se_i = binomial_se(y_pred, y_test)
        standard_errors.append(se_i)
        
        
    tree_range = np.arange(1, 101, 1)

    plt.figure(figsize=(14, 8))
    plt.plot(tree_range, misclassification_rates, linestyle = '-', color = 'royalblue', linewidth=2)
    plt.xlabel("Number of Trees", fontsize=16)
    plt.ylabel("Misclassification Rate", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("misslassification_rates.pdf")
    plt.show()


def plot_variable_importances(learn_data, legend):

    
    X_learn, y_learn = learn_data
    rf = RandomForest(rand=random.Random(0), n=100)
    model = rf.build(X_learn, y_learn)
    importances = model.importance()
    
    # feature_indices = np.arange(len(importances))

    rf_1 = RandomForest(rand=random.Random(0), n = 100, get_candidate_columns=all_columns)
    model_1 = rf_1.build(X_learn, y_learn)
    
    root_features = [tree.feature for tree in model_1.trees]
    root_feature_counts = np.zeros(len(importances))
    for feature in root_features:
        if feature < len(root_feature_counts):  
            root_feature_counts[feature] += 1

    plt.figure(figsize=(14,8))
    
    feature_ix = np.linspace(0, len(legend) - 1, 10, dtype = int)
    selected_features = [legend[ix] for ix in feature_ix]
    plt.bar(range(len(legend)), importances, color='royalblue', alpha = 0.8)
    
    feature_indices = np.argwhere(root_feature_counts > 0).flatten()
    for i in feature_indices:
        plt.text(i, importances[i] + 0.0002, f"{int(root_feature_counts[i])}", fontsize=10, ha='center', color='black', fontweight='bold')

        
    plt.scatter(feature_indices, importances[feature_indices] , s=25, color='red', alpha=0.7)
    plt.xlabel("Variable Name", fontsize=14)
    plt.ylabel("Variable Importance", fontsize=14)
    plt.xticks(fontsize = 12)
    plt.xticks(ticks=feature_ix, labels=selected_features, rotation=30)
    plt.yticks(fontsize = 12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("variable_importance.pdf")
    plt.show()
    

def compare_performance(learn, test): 

    X_learn, y_learn = learn
    X_test, y_test = test
    rf = RandomForest(rand=random.Random(0), n = 1000)
    model = rf.build(X_learn, y_learn)
    importances = model.importance()
    importances_3 = model.importance3()
    # get top 3 single most important features and build a tree 
    importances = np.argsort(importances)[::-1][:10]
    top3_single = importances[:3]    
    # print(top3_single)
    
    X_learn_single = X_learn[:, top3_single]
    X_test_single = X_test[:, top3_single]
    
    tree = Tree(rand=random.Random(0), get_candidate_columns=all_columns)
    model = tree.build(X_learn_single, y_learn)
    y_pred_test = model.predict(X_test_single)
    y_pred_train = model.predict(X_learn_single)
    
    mean_msr_train, se_msr_train = bootstrap_se(y_pred=y_pred_train, y_true = y_learn)
    mean_msr_test, se_msr_test = bootstrap_se(y_pred = y_pred_test, y_true = y_test)
    print(f"3 single features with highest importance, train set: MSR: {mean_msr_train}, SE: {se_msr_train}")
    print(f"3 single features with highest importance, test set: MSR: {mean_msr_test}, SE: {se_msr_test}")
    
    # get most important triplets and build a tree
    top3_combined = np.unravel_index(np.argmax(importances_3), importances_3.shape)
    tree_3 = Tree(rand=random.Random(0), get_candidate_columns=all_columns)
    
    print("---------------------------------------------------------------------------")

    
    X_learn_triplet = X_learn[:, top3_combined]
    X_test_triplet = X_test[:, top3_combined]
    
    model_3 = tree_3.build(X_learn_triplet, y_learn)
    y_pred_test_triplet = model_3.predict(X_test_triplet)
    y_pred_train_triplet = model_3.predict(X_learn_triplet)
    mean_msr_train, se_msr_train = bootstrap_se(y_pred=y_pred_train_triplet, y_true = y_learn)
    mean_msr_test, se_msr_test = bootstrap_se(y_pred = y_pred_test_triplet, y_true = y_test)
    
    print(f"Triplet with highest importance, train set: MSR: {mean_msr_train}, SE: {se_msr_train}")
    print(f"Triplet with highest importance, test set: MSR: {mean_msr_test}, SE: {se_msr_test}")
    
def top3_features_unknown_data(learn, test):
    
    X_learn, y_learn = learn
    X_test, y_test = test
    rf = RandomForest(rand=random.Random(0), n = 1000)
    model = rf.build(X_learn, y_learn)
    features = model.importance3_structure()   
    
    X_learn = X_learn[:, features]
    X_test = X_test[:, features]
    tree =  Tree(rand=random.Random(0), get_candidate_columns=all_columns)
    model = tree.build(X_learn, y_learn)
    y_pred_test = model.predict(X_test)
    y_pred_learn = model.predict(X_learn)
    mean_msr_train, se_msr_train = bootstrap_se(y_pred=y_pred_learn, y_true = y_learn)
    mean_msr_test, se_msr_test = bootstrap_se(y_pred = y_pred_test, y_true = y_test)
    
    print(f"Importance3_structure, 3 top features train set : MSR: {mean_msr_train}, SE: {se_msr_train}")
    print(f"Importance3_structure, 3 top features test set : MSR: {mean_msr_test}, SE: {se_msr_test}")

 
if __name__ == "__main__":

    # TEST for manually calculated tree  
    # unittest.main()
    learn, test, legend = tki()

    ### PART1 
    print("---------------------------------------------------------------------------")
    #begin = time.time()
    train_est_dt, test_est_dt = hw_tree_full(learn, test)
    #end = time.time()
    print(f"Train data, single tree: {train_est_dt}")
    print(f"Test data, single tree: {test_est_dt}")
    print("---------------------------------------------------------------------------")
    train_est_rf, test_est_rf = hw_randomforests(learn, test)
    print(f"Train data, random forest n = 100: {train_est_rf}")
    print(f"Test data, random forest n = 100: {test_est_rf}")
    # plot misclassifacation rates for n features 
    plot_misclassification_rates(learn, test)
    
    
    ### PART2
    plot_variable_importances(learn, legend)
    print("---------------------------------------------------------------------------")

    compare_performance(learn, test)
    print("---------------------------------------------------------------------------")
    top3_features_unknown_data(learn, test)
    print("---------------------------------------------------------------------------")
    

