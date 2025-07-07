from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from scipy.special import logsumexp, expit
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.utils import resample
from matplotlib.gridspec import GridSpec
from sklearn.metrics import log_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import unittest


class MyTests(unittest.TestCase):
    def setUp(self):
        # for ordinal test
        X_ord, y_ord = self.simple_ordinal_dataset()
        train_idx = [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
        test_idx = [i for i in range(len(X_ord)) if i not in train_idx]

        self.X_ord_train = X_ord[train_idx]
        self.y_ord_train = y_ord[train_idx]
        self.X_ord_test = X_ord[test_idx]
        self.y_ord_test = y_ord[test_idx]

        # for multinomial test
        X_multi, y_multi = self.synthetic_multinomial_data()
        split = int(0.8 * len(X_multi))
        self.X_multi_train = X_multi[:split]
        self.y_multi_train = y_multi[:split]
        self.X_multi_test = X_multi[split:]
        self.y_multi_test = y_multi[split:]

    def test_multinomial_predictions(self):
       
        custom_model = MultinomialLogReg().build(self.X_multi_train, self.y_multi_train)
        custom_probs = custom_model.predict(self.X_multi_test)
        custom_preds = np.argmax(custom_probs, axis=1)
        acc_custom = accuracy_score(self.y_multi_test, custom_preds)

        sklearn_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        sklearn_model.fit(self.X_multi_train, self.y_multi_train)
        sklearn_preds = sklearn_model.predict(self.X_multi_test)
        acc_sklearn = accuracy_score(self.y_multi_test, sklearn_preds)

        print("Custom acc:", acc_custom)
        print("Sklearn acc:", acc_sklearn)

        self.assertAlmostEqual(acc_custom, acc_sklearn, delta=0.1)


    def test_ordinal_predictions(self):
        model = OrdinalLogReg().build(self.X_ord_train, self.y_ord_train)
        probs = model.predict(self.X_ord_test)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == self.y_ord_test)

        self.assertEqual(acc, 1.0)
    
    def synthetic_multinomial_data(self, seed=0):
        np.random.seed(seed)
        n_samples = 300
        X = np.random.randn(n_samples, 2)
        W = np.random.randn(2, 3)
        logits = X @ W
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # handle large values 
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        y = np.array([np.random.choice(3, p=p) for p in probs])
        return X, y
    
    def simple_ordinal_dataset(self):
        X = np.array([
            [1], [2], [3], [4], [5],   # class 0
            [6], [7], [8], [9], [10],  # class 1
            [11], [12], [13], [14], [15]  # class 2
        ])
        y = np.array([
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 
            2, 2, 2, 2, 2])
        return X, y


class MultinomialLogReg:
    def __init__(self):
        self.B = None       # weight matrix 
        self.k = None       # number of features
        self.m = None       # number of classes
        self.n = None       # number of samples

    def neg_log_likelihood(self, theta, X, y):

        B = theta.reshape(self.k, self.m - 1)        # reshape flat theta to k x (m - 1) matrix
        Z = X @ B                                    # compute logits 
        Z_full = np.hstack([Z, np.zeros((self.n, 1))])  # add ref class with logit 

        log_probs = Z_full - logsumexp(Z_full, axis=1, keepdims=True)  # logsoftmax
        Y_onehot = np.zeros_like(log_probs)
        Y_onehot[np.arange(self.n), y] = 1

        log_likelihood = np.sum(Y_onehot * log_probs)
        return -log_likelihood  

    def build(self, X, y):

        X = np.hstack([X, np.ones((X.shape[0], 1))]) 
        self.n, self.k = X.shape  # number of samples, number of features
        self.m = np.unique(y).size  # total number of classes
        m1 = self.m - 1             # number of non-reference classes

        np.random.seed(0)
        theta = 0.001 * np.random.randn(self.k * m1)  # initial weights

        result = fmin_l_bfgs_b(
            func=self.neg_log_likelihood,
            x0=theta,
            args=(X, y),
            approx_grad=True
        )

        theta_opt = result[0]
        self.B = theta_opt.reshape(self.k, m1)  # save learned weights
        return self

    def predict_proba(self, X):

        X = np.hstack([X, np.ones((X.shape[0], 1))]) 
        n = X.shape[0]
        Z = X @ self.B
        Z_full = np.hstack([Z, np.zeros((n, 1))])  # add logits for reference class
        log_probs = Z_full - logsumexp(Z_full, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs

    def predict(self, X):
        return self.predict_proba(X)
        # return np.argmax(probs, axis=1)

class OrdinalLogReg:

    def __init__(self):
        self.beta = None        # weight vector
        self.thresholds = None  # thresholds
        self.k = None           # number of features
        self.m = None           # number of classes
        self.n = None           # number of samples

    def neg_log_likelihood(self, theta, X, y):
        # parameters
        beta = theta[:self.k].reshape(-1, 1)            
        thresh = theta[self.k:]  # m - 2 thresholds
        thresholds = np.sort(np.concatenate([[0.0], thresh])) # fix one of the thresholds to be 0
        
        eta = X @ beta                                     
        log_likelihood = 0

        for j in range(self.m):
            if j == 0:
                p = expit(thresholds[0] - eta)
            elif j == self.m - 1:
                p = 1 - expit(thresholds[-1] - eta)
            else:
                p = expit(thresholds[j] - eta) - expit(thresholds[j - 1] - eta)

            mask = (y == j).reshape(-1, 1) 
            log_likelihood += np.sum(mask * np.log(p + 1e-12))

        return -log_likelihood

    def build(self, X, y):

        X = np.hstack([X, np.ones((X.shape[0], 1))]) # include the intercept
        self.n, self.k = X.shape  
        self.m = np.unique(y).size # get number of categories

        beta_init = np.zeros(self.k) 
        thresholds_init = np.linspace(-1, 1, self.m - 2) # only learn m-2 thresholds
        theta_init = np.concatenate([beta_init, thresholds_init])

        result = fmin_l_bfgs_b(func=self.neg_log_likelihood,x0=theta_init,args=(X, y), approx_grad=True)

        theta_opt = result[0]
        self.beta = theta_opt[:self.k].reshape(-1, 1)
        thresh = theta_opt[self.k:] # learned thresholds
        self.thresholds = np.sort(np.concatenate([[0.0], thresh]))  # ensure ordering
        return self

    def predict(self, X):

        X = np.hstack([X, np.ones((X.shape[0], 1))]) 
        eta = X @ self.beta  # (n, 1)
        n = X.shape[0]
        probs = np.zeros((n, self.m))

        for j in range(self.m):
            if j == 0:
                probs[:, j] = expit(self.thresholds[0] - eta[:, 0])
            elif j == self.m - 1:
                probs[:, j] = 1 - expit(self.thresholds[-1] - eta[:, 0])
            else:
                probs[:, j] = expit(self.thresholds[j] - eta[:, 0]) - expit(self.thresholds[j - 1] - eta[:, 0])
        return probs


    
def transform_variables(df): 

    # explain why dropping one feature
    cat_cols = ["Competition", "PlayerType", "Movement"]
    numerical_cols = ["Transition","TwoLegged","Angle" ,"Distance"]

    # print(df["ShotType"])
    
    # y = df["ShotType"]
    class_names = ['above head', 'dunk', 'hook shot', 'layup', 'tip-in', 'other'] 
    y = pd.Categorical(df['ShotType'], categories= class_names, ordered=True)
    y = y.codes

    df = df.drop(columns = ["ShotType"])
    for col in cat_cols: 
        df[col] = df[col].astype("category")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ]
    )

    X = preprocessor.fit_transform(df)
    feat_names = preprocessor.get_feature_names_out()
    return X, y, feat_names, class_names


def coefficient_importance(X, y,  n_bootstraps = 100):

    # X, y, _, _ = transform_variables(df)
    coef_list = []
    
    for i in range(n_bootstraps):

        X_sample, y_sample = resample(X, y, replace=True, random_state=i)
        print(i)

        mr = MultinomialLogReg()
        mr.build(X_sample, y_sample)

        coef_list.append(mr.B.flatten())

    coef_list = np.array(coef_list)
    means = coef_list.mean(axis=0)
    stds = coef_list.std(axis=0)

    lower_bounds = np.percentile(coef_list, 2.5, axis = 0)
    upper_bounds = np.percentile(coef_list, 97.5, axis = 0)

    np.savez("bootstrap_coefs.npz", means=means, stds=stds, 
             lower_bounds = lower_bounds, upper_bounds = upper_bounds)

    return means, stds, lower_bounds, upper_bounds

def plot_coefficients(feature_names, class_labels, B, stds, lower_bounds, upper_bounds, reference_class='other'):
    
    #nprint(feat_names)
    numerical_shades = {
        'num__Transition': 'royalblue',
        'num__TwoLegged': 'deepskyblue',
        'num__Angle': 'dodgerblue',
        'num__Distance': 'steelblue'
    }
    feature_groups = {
        'cat__Competition': 'orange',
        'cat__PlayerType': 'green',
        'cat__Movement': 'red'
    }

    feature_colors = []
    for f in feature_names:
        if f in numerical_shades:
            feature_colors.append(numerical_shades[f])
        else:
            for prefix, color in feature_groups.items():
                if f.startswith(prefix):
                    feature_colors.append(color)
                    break
            else:
                feature_colors.append("gray")

    feature_labels = [f.split('__')[1] if '__' in f else f for f in feature_names]
    B = np.array(B)
    stds = stds.reshape(B.shape)

    num_features, num_classes = B.shape
    y = np.arange(num_features)

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, num_classes + 1, width_ratios=[0.6] + [1] * num_classes, wspace=0.1)

    # external feature labels column
    ax_labels = fig.add_subplot(gs[0, 0])
    ax_labels.set_ylim(num_features - 0.5, -0.5)
    ax_labels.set_xlim(0, 1)
    ax_labels.axis("off")
    for i, label in enumerate(feature_labels):
        ax_labels.text(1, i, label, ha="right", va="center", fontsize=12)

    # class plots
    for idx in range(num_classes):
        ax = fig.add_subplot(gs[0, idx + 1], sharey=ax_labels)
        coefs = B[:, idx]
        # errors = stds[:, idx]
        errors_lower = coefs - lower_bounds[:, idx]
        errors_upper = upper_bounds[:, idx] - coefs

        ax.barh(y, coefs, xerr=[errors_lower, errors_upper], color=feature_colors, capsize=3, alpha=0.8, height=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
        ax.set_title(f"{class_labels[idx]} vs {reference_class}", fontsize=12)
        ax.set_yticks([])
        ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    plt.subplots_adjust(left=0.001, right=0.97, top=0.88, bottom=0.1)
    plt.savefig("coeff_importance.pdf")
    plt.show()


def multinomial_bad_ordinal_good(n_samples=1000, random_state=0):
    
    np.random.seed(random_state)
    x = np.random.normal(0, 1, size=(n_samples, 1))  # (n, 1)

    beta = 5.0              
    noise = 0.25           

    z = beta * x + np.random.normal(0, noise, size=(n_samples, 1))

    #  thresholds for 5  classes
    thresholds = [-2, -1,  0, 1]
    y = np.digitize(z, thresholds)

    return x, y.ravel()


def calculate_vif(X, feature_names):

    # high vif value tells how well a single feature is linearly predictable by all the other features
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def plot_vif_comparison(vif_before, vif_after):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # before removing 
    vif_labels_before = [f.split('__')[1] if '__' in f else f for f in vif_before['Feature']]
    axs[0].barh(vif_labels_before, vif_before['VIF'], color='red', alpha=0.6)
    axs[0].axvline(10, color='black', linestyle='--', linewidth=1, label="VIF = 10")
    axs[0].set_xlabel("VIF")
    axs[0].tick_params(axis='y', labelsize=12)
    axs[0].set_title("Before Removing Multicollinear Feature", fontsize = 12)
    axs[0].invert_yaxis()
    axs[0].legend()

    #  after removing movement
    vif_labels_after = [f.split('__')[1] if '__' in f else f for f in vif_after['Feature']]
    axs[1].barh(vif_labels_after, vif_after['VIF'], color='blue', alpha=0.6, )
    axs[1].axvline(10, color='black', linestyle='--', linewidth=1, label="VIF = 10")
    axs[1].set_xlabel("VIF")
    axs[1].set_title("After Removing Multicollinear Feature", fontsize = 14)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("handle_multicolinearity.pdf")
    plt.show()

def test_mul_bad_ord_good(n_bootstrap=100):

    X_test, y_test = multinomial_bad_ordinal_good(n_samples=30, random_state=42)
    X_pool, y_pool = multinomial_bad_ordinal_good(n_samples=100, random_state=42)

    logloss_multi = []
    logloss_ord = []

    for i in range(n_bootstrap):
        X_train, y_train = resample(X_pool, y_pool, replace=True, random_state=i)

        # multinomial model
        multinomial = MultinomialLogReg()
        multinomial.build(X_train, y_train)
        probs_multi = multinomial.predict(X_test)
        loss_multi = log_loss(y_test, probs_multi)
        logloss_multi.append(loss_multi)

        # ordinal model
        ordinal = OrdinalLogReg()
        ordinal.build(X_train, y_train)
        probs_ord = ordinal.predict(X_test)
        loss_ord = log_loss(y_test, probs_ord)
        logloss_ord.append(loss_ord)

    # compute means and standard errors
    logloss_multi = np.array(logloss_multi)
    logloss_ord = np.array(logloss_ord)

    mean_multi = logloss_multi.mean()
    stderr_multi = logloss_multi.std(ddof=1)

    mean_ord = logloss_ord.mean()
    stderr_ord = logloss_ord.std(ddof=1)

    print("Log-loss:")
    print(f"Multinomial: {mean_multi:.4f} ± {stderr_multi:.4f}")
    print(f"Ordinal    : {mean_ord:.4f} ± {stderr_ord:.4f}")


def get_standardized_residuals(X, residuals):

    n, p = X.shape
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H).reshape(-1, 1)
    MSE = np.sum(residuals**2) / (n - p)
    std_res = residuals / np.sqrt(MSE * (1 - h))
    return std_res

def plot_qq(std_residuals, filename="qqplot_residuals.pdf"):
    res_sorted = np.sort(std_residuals.flatten())
    n = len(res_sorted)
    probs = np.array([(i - 0.5) / n for i in range(1, n + 1)])
    z_scores = norm.ppf(probs)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(z_scores, res_sorted, color="#b0b0b0", alpha=0.8, s=14)
    ax.plot(z_scores, z_scores, color="#e74c3c", linestyle='--', linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles (Z-scores)", fontsize=14)
    ax.set_ylabel("Studentized Residuals", fontsize=14)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=1)
    plt.savefig(filename)
    plt.show()


def plot_residuals_vs_fitted(y_hat, std_res, filename="residuals_vs_fitted.pdf"):
    
    std_flat = std_res.flatten()
    outliers = np.abs(std_flat) > 2
    normal = ~outliers

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_hat[normal], std_flat[normal], color="#b0b0b0", alpha=0.6, s=14)
    ax.scatter(y_hat[outliers], std_flat[outliers], color="#e74c3c", alpha=0.9, s=40)
    ax.axhline(0, color="#e74c3c", linestyle='--', linewidth=1.3)
    ax.axhline(2, color="#f39c12", linestyle=':', linewidth=1)
    ax.axhline(-2, color="#f39c12", linestyle=':', linewidth=1)
    ax.set_xlabel("Fitted Values", fontsize=14)
    ax.set_ylabel("Studentized Residuals", fontsize=14)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=1)
    plt.savefig(filename)
    plt.show()

def plot_cooks_distance(X, residuals, filename="cooks_distance.pdf"):
    n, p = X.shape
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H).reshape(-1, 1)
    MSE = np.sum(residuals**2) / (n - p)
    cooks_d = (residuals**2 / (p * MSE)) * (h / (1 - h)**2)

    cooks_flat = cooks_d.flatten()
    threshold = 4 / n
    influential = np.where(cooks_flat > threshold)[0]
    non_influential = np.where(cooks_flat <= threshold)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(non_influential, cooks_flat[non_influential], color="#b0b0b0", alpha=0.4, s=12)
    ax.scatter(influential, cooks_flat[influential], color="#e74c3c", alpha=0.9, s=50)
    for i in influential:
        ax.text(i, cooks_flat[i], str(i), fontsize=14, ha='center', va='bottom', color="#e74c3c")
    ax.axhline(threshold, color="#f39c12", linestyle='--', linewidth=1.5)
    ax.set_xlim([0, n])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sample Index", fontsize=14)
    ax.set_ylabel("Cook's Distance", fontsize=14)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=1)
    plt.savefig(filename)
    plt.show()

def glm_diagnostics():
    data = pd.read_csv("dataset.csv", sep=";")
    x = data['Angle'].values
    y = data['Distance'].values.reshape(-1, 1)

    X = np.column_stack((np.ones(len(x)), x))
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat
    residuals = y - y_hat
    std_residuals = get_standardized_residuals(X, residuals)

    plot_qq(std_residuals)
    plot_residuals_vs_fitted(y_hat, std_residuals)
    plot_cooks_distance(X, residuals)

def multinomial_application():

    df = pd.read_csv("dataset.csv", sep = ";")
    X, y, feat_names, class_names = transform_variables(df)

    # estimate uncorrelated features
    X = pd.DataFrame(X, columns=feat_names)
    vif_before = calculate_vif(X, feat_names)
    # print(vif_df.sort_values("VIF", ascending=False))
    X_red = X.drop(columns=["cat__Movement_no"])
    vif_after = calculate_vif(X_red, feat_names)
    # plot_vif_comparison(vif_before, vif_after)

    X_red = X_red.to_numpy()
    mr = MultinomialLogReg()
    mr.build(X_red, y)

    # bootstrap estimate cis
    # means, stds, lower_bounds, upper_bounds = coefficient_importance(X_red, y)
    feat_names_red = [f for f in feat_names if "Movement_no" not in f]
    feat_names_red.append("Intercept")  # Add intercept at the end
    
    data = np.load("bootstrap_coefs.npz")
    stds = data["stds"]
    means = data["means"]
    lower_bounds = data["lower_bounds"].reshape(mr.B.shape)
    upper_bounds = data["upper_bounds"].reshape(mr.B.shape)

    plot_coefficients(feat_names_red, class_names, mr.B, stds, lower_bounds, upper_bounds)

if __name__ ==  "__main__":

    # 1. Test Implementation
    # unittest.main()

    # 2.1 Application of Multinomial Regression
    # multinomial_application()
    
    # 2.2 Application of Ordinal Regerssion
    # test_mul_bad_ord_good()

    # 3. GLM Diagnostics 
    glm_diagnostics()
