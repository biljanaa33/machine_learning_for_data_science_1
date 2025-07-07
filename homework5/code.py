import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

    
class Polynomial:
    """An example of a kernel."""

    def __init__(self, M):
        # here a kernel could set its parameters
        self.degree = M

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        A = np.asarray(A)
        B = np.asarray(B)

        # both 1D return scalar value
        if A.ndim == 1 and B.ndim == 1:
            return (1 + A.dot(B))**self.degree
        
        # A is vector, B is matrix
        if A.ndim == 1 and B.ndim == 2:
                return (1 + B.dot(A))**self.degree
        
        # B is vector, A is matrix
        if B.ndim == 1 and A.ndim == 2:
                return (1 + A.dot(B))**self.degree
        
        # both 2D matrices return gram matirx
        if A.ndim == 2 and B.ndim == 2:
            return (1 + A.dot(B.T))**self.degree
                         
class RBF:
    """An example of a kernel."""

    def __init__(self, sigma):

        self.sigma2 = sigma**2

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        A = np.asarray(A)
        B = np.asarray(B)

        if A.ndim == 1 and B.ndim == 1: 
             
            d2 = A.dot(A) - 2*A.dot(B) + B.dot(B)
            return np.exp(-d2 / (2*self.sigma2))
        
        elif A.ndim == 1 and B.ndim == 2:
             
             d2 = (A.dot(A) - 2*B.dot(A) + np.sum(B**2, axis=1))
             return np.exp(-d2 / (2*self.sigma2))

        elif A.ndim == 2 and B.ndim == 1:
            d2 = (np.sum(A**2, axis=1) - 2*A.dot(B) + B.dot(B))
            return np.exp(-d2 / (2*self.sigma2))
        
        elif A.ndim == 2 and B.ndim == 2:
           
            AA = np.sum(A**2, axis=1)[:, np.newaxis]  
            BB = np.sum(B**2, axis=1)[np.newaxis, :]   #
            D2 = AA - 2*A.dot(B.T) + BB              
            return np.exp(-D2 / (2*self.sigma2))

class TemporalRBF:
   
    def __init__(self, sigma_f, sigma_t):
        self.sigma_f2 = sigma_f**2
        self.sigma_t2 = sigma_t**2

    def __call__(self, A, B):
        N, M = len(A), len(B)
        K = np.zeros((N, M))
        for i in range(N):
            fi, ti = A[i]
            for j in range(M):
                fj, tj = B[j]
                df2 = np.sum((fi[:, None] - fj[None, :])**2, axis=2)
                dt2 = (ti[:, None] - tj[None, :])**2
                K[i, j] = np.exp(-df2 / self.sigma_f2 - dt2 / self.sigma_t2).sum()
        return K
        
class KernelizedRidgeRegression:

    def __init__(self, kernel, lambda_):

        self.kernel   = kernel  
        self.lambda_  = lambda_ 
    
    def fit(self, X, y):
         
        K = self.kernel(X, X)
        n = K.shape[0]
        A = K + self.lambda_ * np.eye(n)
        self.alpha   = np.linalg.solve(A, y)
        self.X_train = X.copy()
        return self

    def predict(self, X_new):
         
        K_new = self.kernel(X_new, self.X_train) 
        return K_new.dot(self.alpha)
         
class SVR:
    def __init__(self, kernel, lambda_, epsilon):
        
        #print("HELLLOOOO")
        self.kernel  = kernel
        self.C       = 1.0 / lambda_  
        self.epsilon = epsilon


    def fit(self, X, y):

        #X = np.array(X).reshape(-1, 1)
        #y = np.array(y).flatten() 

        n = len(y)
        K = self.kernel(X, X)

        # P = np.vstack([ np.hstack([ K, -K]), np.hstack([-K,  K])])
        # print("X shape:", np.shape(X))
        # print("K shape:", np.shape(K))

        P = np.block([[ K,   -K],[-K,    K]])

        q_full = np.hstack([ self.epsilon - y, self.epsilon + y ])

        # enusre proper ordering
        order = np.vstack([np.arange(n), np.arange(n, 2*n)]).T.flatten()

        P = P[np.ix_(order, order)]
        q = q_full[order]

        G_full = np.vstack([ np.eye(2*n), -np.eye(2*n) ]) 
        h_full = np.hstack([ self.C * np.ones(2*n), np.zeros(2*n) ])  

        G = G_full[:, order]    
        h = h_full             

        A_full = np.hstack([ np.ones(n), -np.ones(n) ]).reshape(1,2*n)
        A = A_full[:, order]
        b_eq = 0.0
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b_eq))

        z = np.array(sol['x']).flatten()  
        self.alpha      = z[0::2] # alphas
        self.alpha_star = z[1::2] # alpha star
        self.alpha_diff = self.alpha - self.alpha_star

        # get bias term 
        self.b = float(sol['y'][0])
        self.X_train = X
        return self
    

    def predict(self, X_new):

        K_new = self.kernel(X_new, self.X_train)
        return K_new.dot(self.alpha_diff) + self.b

    def get_alpha(self):
       
        return np.column_stack((self.alpha, self.alpha_star))
    
    def get_b(self):

        return self.b

#################################### FIRST PART ###########################################
def fit_sine():
    
    df = pd.read_csv('sine.csv')
    print(df.head())

    #plt.figure(figsize=(10, 5))
    #plt.scatter(df['x'], df['y'])
    #plt.show()

    # fit SVR poly kernel 
    scaler = StandardScaler()
    X = df['x'].values.reshape(-1, 1)
    X_scaled = scaler.fit_transform(X)
    y = np.array(df['y']).flatten()  
    fit_poly_kernel_svr(X_scaled, y)
    fit_rbf_kernel_svr(X, y)
    fit_poly_kernel_krr(X_scaled, y)
    fit_rbf_kernel_krr(X, y)

def fit_poly_kernel_svr(X_scaled, y): 

    poly_kernel = Polynomial(M = 9)
    svr_poly = SVR(kernel=poly_kernel, lambda_=0.0001, epsilon=1.0)
    svr_poly = svr_poly.fit(X_scaled, y)

    alpha_vals = svr_poly.get_alpha() # support vectors where alpha is different then 0
    support_indices = np.where((alpha_vals[:, 0] > 1e-5) | (alpha_vals[:, 1] > 1e-5))[0]
    X_support = X_scaled[support_indices]
    y_support = y[support_indices]
    print(f"Number of support vectors: {len(X_support)}, ratio: {len(X_support)/len(X_scaled)}")

    x_test = np.linspace(X_scaled.min(), X_scaled.max(), 300)
    x_test = np.array(x_test).reshape(-1, 1)
    y_pred = svr_poly.predict(x_test)

    epsilon = svr_poly.epsilon # plot margin
    plt.plot(x_test, y_pred + epsilon, linestyle='--', color='#a0a0a0', linewidth=1, label='Epsilon Margin')
    plt.plot(x_test, y_pred - epsilon, linestyle='--', color='#a0a0a0', linewidth=1)
    # original data
    plt.scatter(X_support, y_support, facecolors='none', edgecolors="#e74c3c", label='Support Vectors', s=80, linewidth=1.5)
    plt.scatter(X_scaled, y, label='Data', color='#e74c3c', alpha=0.3)
    plt.plot(x_test, y_pred, label='SVR Prediction', color="#a0a0a0") # plot svr prediction

    plt.xlabel('x value')
    plt.ylabel('y = sin(x)')
    # plt.legend()
    # plt.savefig("svr_poly.pdf", bbox_inches='tight')
    plt.show()

def fit_rbf_kernel_svr(X, y):

    rbf_kernel = RBF(sigma= 5) 
    svr_rbf = SVR(kernel=rbf_kernel, lambda_=0.0001, epsilon=1.0)
    svr_rbf = svr_rbf.fit(X, y)

    alpha_vals = svr_rbf.get_alpha() 
    support_indices = np.where((alpha_vals[:, 0] > 1e-5) | (alpha_vals[:, 1] > 1e-5))[0]
    X_support = X[support_indices]
    y_support = y[support_indices]
    print(f"Number of support vectors: {len(X_support)}, ratio: {len(X_support)/len(X)}")

    x_test = np.linspace(X.min(), X.max(), 300)
    x_test = np.array(x_test).reshape(-1, 1)
    y_pred = svr_rbf.predict(x_test)

    epsilon = svr_rbf.epsilon
    plt.plot(x_test, y_pred + epsilon, linestyle='--', color='#a0a0a0', linewidth=1, label='Epsilon Margin')
    plt.plot(x_test, y_pred - epsilon, linestyle='--', color='#a0a0a0', linewidth=1)
    plt.scatter(X_support, y_support, facecolors='none', edgecolors="#e74c3c", label='Support Vectors', s=80, linewidth=1.5)
    plt.scatter(X, y, label='Data', color='#e74c3c', alpha=0.3)
    plt.plot(x_test, y_pred, label='SVR Prediction', color="#a0a0a0")

    plt.xlabel('x value')
    plt.ylabel('y = sin(x)')
    plt.savefig("svr_rbf_1.pdf", bbox_inches='tight')
    plt.show()

def fit_rbf_kernel_krr(X, y): 

    rbf_kernel = RBF(sigma= 5)
    krr_rbf = KernelizedRidgeRegression(kernel=rbf_kernel, lambda_=0.0001)
    krr_rbf = krr_rbf.fit(X, y)

    x_test = np.linspace(X.min(), X.max(), 300)
    x_test = np.array(x_test).reshape(-1, 1)
    y_pred = krr_rbf.predict(x_test)

    plt.scatter(X, y, label='Data', color='#e74c3c', alpha=0.3)
    plt.plot(x_test, y_pred, label='KRR Prediction', color="#a0a0a0")
    plt.xlabel('x value')
    plt.ylabel('y = sin(x)')
    # plt.savefig("krr_rbf.pdf", bbox_inches='tight')
    plt.show()

def fit_poly_kernel_krr(X_scaled, y): 

    poly_kernel = Polynomial(M = 9)
    krr_poly = KernelizedRidgeRegression(kernel = poly_kernel, lambda_= 0.0001)
    krr_poly = krr_poly.fit(X_scaled, y)

    x_test = np.linspace(X_scaled.min(), X_scaled.max(), 300)
    x_test = np.array(x_test).reshape(-1, 1)
    y_pred = krr_poly.predict(x_test)

    plt.scatter(X_scaled, y, label='Data', color='#e74c3c', alpha=0.3)
    plt.plot(x_test, y_pred, label='KRR Prediction', color="#a0a0a0")
    plt.xlabel('x value')
    plt.ylabel('y = sin(x)')
    # plt.savefig("krr_poly.pdf", bbox_inches='tight')
    plt.show()

    
#################################### SECOND PART ###########################################

def housing_data():

    df = pd.read_csv("housing2r.csv")
    M_values = list(range(1, 11))
    X, y = df.drop(columns='y').values,  df['y'].values 

    # poly kernel 
    plot_results_krr_poly(X, y, M_values)
    plot_results_svr_poly(X, y, M_values)

    # rbf kernel
    sigma_values = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
    plot_results_krr_rbf(X, y, sigma_values)
    plot_results_svr_rbf(X, y, sigma_values)

def fit_svr_rbf_fixed(X, y, sigma_values, lambda_ = 1.0, k_folds = 10):

    print("Fitting svr with rbf kernel")
    mse_vals = []
    se_vals = []
    sv_counts = []

    for sigma in sigma_values:

        kernel = RBF(sigma=sigma)
        fold_mse_fixed = []
        fold_sv_counts = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X): 

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            svr = SVR(kernel, lambda_, epsilon=10.0)
            svr = svr.fit(X_train, y_train)
            y_pred = svr.predict(X_test)
            mse_fold_ix = mean_squared_error(y_test, y_pred)
            fold_mse_fixed.append(mse_fold_ix)

            alpha_vals = svr.get_alpha()
            support_indices = np.where((alpha_vals[:, 0] > 1e-5) | (alpha_vals[:, 1] > 1e-5))[0]
            fold_sv_counts.append(len(support_indices))
        
        mean_mse = np.mean(fold_mse_fixed)
        std_mse = np.std(fold_mse_fixed) / np.sqrt(k_folds)
        print(f"Sigma: {sigma}, MSE: {mean_mse}, SE: {std_mse}")
        mse_vals.append(mean_mse)
        se_vals.append(std_mse)
        sv_counts.append(np.mean(fold_sv_counts))

    return mse_vals, se_vals, sv_counts

def fit_svr_rbf_fine_tune(X, y, sigma_values, lambda_ = 1.0, k_folds = 10):

    lambda_values = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    outer_kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
 
    mse_vals = []
    se_vals = []
    sv_counts = []

    for sigma in sigma_values:
        best_lambda = None
        best_mse = np.inf
        best_se = None
        
        for lambda_ in lambda_values: 
            fold_mses = []

            for train_idx, test_idx in outer_kf.split(X):

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                kernel = RBF(sigma=sigma)
                svr = SVR(kernel, lambda_, epsilon=10.0)
                svr.fit(X_train, y_train)
                y_pred = svr.predict(X_test)

                fold_mses.append(mean_squared_error(y_test, y_pred))


            mean_mse = np.mean(fold_mses)

            if mean_mse < best_mse:
                best_mse    = mean_mse
                best_lambda = lambda_
                best_se     = np.std(fold_mses, ddof=1) / np.sqrt(k_folds)

        kernel = RBF(sigma=sigma)
        svr = SVR(kernel, best_lambda, epsilon=10.0)
        svr.fit(X, y)
        alpha_vals = svr.get_alpha()
        support_indices = np.where((alpha_vals[:, 0] > 1e-5) | (alpha_vals[:, 1] > 1e-5))[0]
        sv_counts.append(len(support_indices))

        print(f"Simga = {sigma}: best lambda = {best_lambda}, "f"MSE = {best_mse:.4f}, SE = {best_se:.4f}")
        mse_vals.append(best_mse)
        se_vals.append(best_se)
    
    return mse_vals, se_vals, sv_counts

def fit_svr_poly_fixed(X, y, M_values, lambda_ = 1.0, k_folds = 10):

    print("Fitting svr with poly kernel")

    mse_vals = []
    se_vals = []
    sv_counts = []

    for M in M_values:

        kernel = Polynomial(M=M)
        fold_mse_fixed = []
        fold_sv_counts = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X): 

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train) 
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #K_train = (X_train_scaled @ X_train_scaled.T)**M
            #print("M =", M,
            #s"K_train min/max =", K_train.min(), "/", K_train.max())
            svr = SVR(kernel, lambda_, epsilon=10.0)
            svr = svr.fit(X_train_scaled, y_train)
            y_pred = svr.predict(X_test_scaled)
            mse_fold_ix = mean_squared_error(y_test, y_pred)
            fold_mse_fixed.append(mse_fold_ix)

            alpha_vals = svr.get_alpha()
            support_indices = np.where((alpha_vals[:, 0] > 1e-5) | (alpha_vals[:, 1] > 1e-5))[0]
            fold_sv_counts.append(len(support_indices))
        
        mean_mse = np.mean(fold_mse_fixed)
        std_mse = np.std(fold_mse_fixed) / np.sqrt(k_folds)
        print(f"Polynomial: {M}, MSE: {mean_mse}, SE: {std_mse}")
        mse_vals.append(mean_mse)
        se_vals.append(std_mse)
        mean_sv = np.mean(fold_sv_counts)
        sv_counts.append(mean_sv)

    return mse_vals, se_vals, sv_counts

def fit_svr_poly_fine_tune(X, y, M_values, lambda_ = 1.0, k_folds = 10):

    lambda_values = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    outer_kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
 
    mse_vals = []
    se_vals = []
    sv_counts = []

    for M in M_values:
        best_lambda = None
        best_mse = np.inf
        best_se = None
        
        for lambda_ in lambda_values: 
            fold_mses = []

            for train_idx, test_idx in outer_kf.split(X):

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
                X_tr = scaler.transform(X_train)
                X_te = scaler.transform(X_test)

                kernel = Polynomial(M=M)
                krr = SVR(kernel, lambda_, epsilon=10.0)
                krr.fit(X_tr, y_train)
                y_pred = krr.predict(X_te)

                fold_mses.append(mean_squared_error(y_test, y_pred))

            mean_mse = np.mean(fold_mses)

            if mean_mse < best_mse:
                best_mse    = mean_mse
                best_lambda = lambda_
                best_se     = np.std(fold_mses, ddof=1) / np.sqrt(k_folds)

        print(f"Degree M = {M}: best lambda = {best_lambda}, "f"MSE = {best_mse:.4f}, SE = {best_se:.4f}")
        mse_vals.append(best_mse)
        se_vals.append(best_se)

        scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
        X_scaled = scaler.transform(X)

        kernel = Polynomial(M=M)
        svr = SVR(kernel, best_lambda, epsilon=10.0)
        svr.fit(X_scaled, y)
        alpha_vals = svr.get_alpha()
        support_indices = np.where((alpha_vals[:, 0] > 1e-5) | (alpha_vals[:, 1] > 1e-5))[0]
        sv_counts.append(len(support_indices))
    
    return mse_vals, se_vals, sv_counts

def fit_krr_poly_fixed(X, y, M_values, lambda_ = 1.0, k_folds = 10):

    print("Fitting krr with poly kernel")

    mse_vals = []
    se_vals = []

    for M in M_values:

        kernel = Polynomial(M=M)
        fold_mse_fixed = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X): 

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train) 
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #K_train = (X_train_scaled @ X_train_scaled.T)**M
            #print("M =", M,
            #s"K_train min/max =", K_train.min(), "/", K_train.max())
            krr = KernelizedRidgeRegression(kernel, lambda_)
            krr = krr.fit(X_train_scaled, y_train)
            y_pred = krr.predict(X_test_scaled)
            mse_fold_ix = mean_squared_error(y_test, y_pred)
            fold_mse_fixed.append(mse_fold_ix)
        
        mean_mse = np.mean(fold_mse_fixed)
        std_mse = np.std(fold_mse_fixed) / np.sqrt(k_folds)
        print(f"Polynomial: {M}, MSE: {mean_mse}, SE: {std_mse}")
        mse_vals.append(mean_mse)
        se_vals.append(std_mse)

    return mse_vals, se_vals

def fit_krr_rbf_fixed(X, y, sigma_values, lambda_ = 1.0, k_folds = 10):

    print("Fitting krr with rbf kernel")

    mse_vals = []
    se_vals = []

    for sigma in sigma_values:

        kernel = RBF(sigma=sigma)
        fold_mse_fixed = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X): 

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            krr = KernelizedRidgeRegression(kernel, lambda_)
            krr = krr.fit(X_train, y_train)
            y_pred = krr.predict(X_test)
            mse_fold_ix = mean_squared_error(y_test, y_pred)
            fold_mse_fixed.append(mse_fold_ix)
        
        mean_mse = np.mean(fold_mse_fixed)
        std_mse = np.std(fold_mse_fixed) / np.sqrt(k_folds)
        print(f"Sigma: {sigma}, MSE: {mean_mse}, SE: {std_mse}")
        mse_vals.append(mean_mse)
        se_vals.append(std_mse)

    return mse_vals, se_vals

def fit_krr_poly_fine_tune(X, y, M_values, k_folds = 10):

    lambda_values = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    outer_kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
 
    mse_vals = []
    se_vals = []

    for M in M_values:
        best_lambda = None
        best_mse    = np.inf
        best_se     = None
        
        for lambda_ in lambda_values: 
            fold_mses = []
            for train_idx, test_idx in outer_kf.split(X):

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
                X_tr = scaler.transform(X_train)
                X_te = scaler.transform(X_test)

                kernel = Polynomial(M=M)
                krr = KernelizedRidgeRegression(kernel, lambda_)
                krr.fit(X_tr, y_train)
                y_pred = krr.predict(X_te)

                fold_mses.append(mean_squared_error(y_test, y_pred))

            mean_mse = np.mean(fold_mses)

            if mean_mse < best_mse:
                best_mse    = mean_mse
                best_lambda = lambda_
                best_se     = np.std(fold_mses, ddof=1) / np.sqrt(k_folds)

        print(f"Degree M = {M}: best lambda = {best_lambda}, "f"MSE = {best_mse:.4f}, SE = {best_se:.4f}")
        mse_vals.append(best_mse)
        se_vals.append(best_se)
    
    return mse_vals, se_vals

def fit_krr_rbf_fine_tune(X, y, sigma_values, k_folds = 10):

    lambda_values = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    outer_kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
 
    mse_vals = []
    se_vals = []

    for sigma in sigma_values:
        best_lambda = None
        best_mse    = np.inf
        best_se     = None
        
        for lambda_ in lambda_values: 
            fold_mses = []
            for train_idx, test_idx in outer_kf.split(X):

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                kernel = RBF(sigma=sigma)
                krr = KernelizedRidgeRegression(kernel, lambda_)
                krr.fit(X_train, y_train)
                y_pred = krr.predict(X_test)

                fold_mses.append(mean_squared_error(y_test, y_pred))

            mean_mse = np.mean(fold_mses)

            if mean_mse < best_mse:
                best_mse    = mean_mse
                best_lambda = lambda_
                best_se     = np.std(fold_mses, ddof=1) / np.sqrt(k_folds)

        print(f"Sigma = {sigma}: best lambda = {best_lambda}, "f"MSE = {best_mse:.4f}, SE = {best_se:.4f}")
        mse_vals.append(best_mse)
        se_vals.append(best_se)
    
    return mse_vals, se_vals
        
def plot_results_krr_rbf(X, y, sigma_values):

    mse_fixed, se_fixed   = fit_krr_rbf_fixed(X, y, sigma_values, lambda_=1.0, k_folds=10)
    mse_tuned, se_tuned   = fit_krr_rbf_fine_tune(X, y, sigma_values, k_folds=10)
    
    plt.plot(sigma_values, mse_fixed,linestyle='-',marker='o',color='#b0b0b0',   label='Fixed λ')
    plt.errorbar(sigma_values, mse_fixed, yerr=se_fixed,fmt='',ecolor='#b0b0b0', color = '#b0b0b0' , elinewidth=0, capsize=5,capthick=1)
    plt.plot(sigma_values, mse_tuned,linestyle='--',marker='s',color='#e74c3c',label='Tuned λ')
    plt.errorbar(sigma_values, mse_tuned,yerr=se_tuned,fmt='',ecolor='#e74c3c', color='#e74c3c',elinewidth=0,capsize=5,capthick=1)

    plt.xlabel('Sigma')
    plt.ylabel('MSE')
    plt.xticks(sigma_values)
    plt.legend()
    plt.savefig("results_krr_rbf.pdf")
    plt.show()

def plot_results_svr_poly(X, y, M_values):

    mse_fixed, se_fixed, sv_fixed   = fit_svr_poly_fixed(X, y, M_values, lambda_=1.0, k_folds=10)
    mse_tuned, se_tuned, sv_tuned   = fit_svr_poly_fine_tune(X, y, M_values, k_folds=10)
    
    plt.plot(M_values, mse_fixed,linestyle='-',marker='o',color='#b0b0b0',   label='Fixed λ')
    plt.errorbar(M_values, mse_fixed, yerr=se_fixed,fmt='',ecolor='#b0b0b0', color = '#b0b0b0' , elinewidth=0, capsize=5,capthick=1)
    plt.plot(M_values, mse_tuned,linestyle='--',marker='s',color='#e74c3c',label='Tuned λ')
    plt.errorbar(M_values, mse_tuned,yerr=se_tuned,fmt='',ecolor='#e74c3c', color='#e74c3c',elinewidth=0,capsize=5,capthick=1)

    for i, sigma in enumerate(M_values):

        plt.text(sigma, mse_fixed[i] + 0.01, f"{int(sv_fixed[i])}", color='#808080', fontsize=9, ha='center', va='bottom')
        plt.text(sigma, mse_tuned[i] - 0.01, f"{int(sv_tuned[i])}", color='#e74c3c', fontsize=9, ha='center', va='top')

    plt.xlabel('Polynomial degree M')
    plt.ylabel('MSE')
    plt.xticks(M_values)
    plt.legend()
    plt.savefig("results_svr_poly.pdf")
    plt.show()

def plot_results_svr_rbf(X, y, sigma_values):

    mse_fixed, se_fixed, sv_fixed   = fit_svr_rbf_fixed(X, y, sigma_values, lambda_=1.0, k_folds=10)
    mse_tuned, se_tuned, sv_tuned   = fit_svr_rbf_fine_tune(X, y, sigma_values, k_folds=10)
    
    plt.plot(sigma_values, mse_fixed,linestyle='-',marker='o',color='#b0b0b0',   label='Fixed λ')
    plt.errorbar(sigma_values, mse_fixed, yerr=se_fixed,fmt='',ecolor='#b0b0b0', color = '#b0b0b0' , elinewidth=0, capsize=5,capthick=1)
    plt.plot(sigma_values, mse_tuned,linestyle='--',marker='s',color='#e74c3c',label='Tuned λ')
    plt.errorbar(sigma_values, mse_tuned,yerr=se_tuned,fmt='',ecolor='#e74c3c', color='#e74c3c',elinewidth=0,capsize=5,capthick=1)

    for i, sigma in enumerate(sigma_values):

        plt.text(sigma, mse_fixed[i] + 0.01, f"{int(sv_fixed[i])}", color='#808080', fontsize=9, ha='center', va='bottom')
        plt.text(sigma, mse_tuned[i] - 0.01, f"{int(sv_tuned[i])}", color='#e74c3c', fontsize=9, ha='center', va='top')

    plt.xlabel('Sigma value')
    plt.ylabel('MSE')
    plt.xticks(sigma_values)
    plt.legend()
    plt.savefig("results_svr_rbf.pdf")
    plt.show()

def plot_results_krr_poly(X, y, M_values ):

    mse_fixed, se_fixed   = fit_krr_poly_fixed(X, y, M_values, lambda_=1.0, k_folds=10)
    mse_tuned, se_tuned   = fit_krr_poly_fine_tune(X, y, M_values, k_folds=10)
    
    plt.plot(M_values, mse_fixed,linestyle='-',marker='o',color='#b0b0b0',   label='Fixed λ')
    plt.errorbar(M_values, mse_fixed, yerr=se_fixed,fmt='',ecolor='#b0b0b0', color = '#b0b0b0' , elinewidth=0, capsize=5,capthick=1)
    plt.plot(M_values, mse_tuned,linestyle='--',marker='s',color='#e74c3c',label='Tuned λ')
    plt.errorbar(M_values, mse_tuned,yerr=se_tuned,fmt='',ecolor='#e74c3c', color='#e74c3c',elinewidth=0,capsize=5,capthick=1)


    plt.xlabel('Polynomial degree M')
    plt.ylabel('MSE')
    plt.xticks(M_values)
    plt.legend()
    plt.savefig("results_krr_poly.pdf")
    plt.show()

#################################### THIRD PART ###########################################

def make_synthetic_var(N=5000, w=24, noise=0.5, seed=42):
    
    rng = np.random.RandomState(seed)
    X = np.zeros((N, w))
    T = np.zeros(N) # bump location for each sequence
    y = np.zeros(N)
    ts = np.linspace(0, 2*np.pi, w)

    for i in range(N):

        A     = rng.uniform(0.5, 3.5)
        phase = rng.uniform(0, 2*np.pi)
        base  = A * np.sin(ts + phase)

        # bump position and normalize it to [0,1]
        k       = rng.randint(w)
        T[i]    = k / (w - 1)
        bump_h  = rng.uniform(0.5, 2.0)
        bump    = np.zeros(w); 
        bump[k] = bump_h

        # noise_i = rng.uniform(0.1, 1.0)
        seq = base + bump + rng.randn(w)*noise # add noise to each sequence
        X[i] = seq
        y[i] = T[i]

    return X, T, y

def synthetic_data_test():

    w = 48 # window length 

    X, T, y = make_synthetic_var(N=1000, w=w, noise=0.5, seed=42)
    split = int(0.7 * len(y))
    X_tr, X_te = X[:split], X[split:]
    T_tr, T_te = T[:split], T[split:]
    y_tr, y_te = y[:split], y[split:]

    rbf_kernel = RBF(sigma=1.0)
    temporal_kernel = TemporalRBF(sigma_f=1.0, sigma_t=1.0)

    svr_rbf = SVR(kernel=rbf_kernel, lambda_= 1e-2, epsilon = 0.05)
    svr_rbf.fit(X_tr, y_tr)
    y_pred_rbf = svr_rbf.predict(X_te)

    # build temporal input
    Xs_tr = np.empty(len(X_tr), dtype=object)
    Xs_te = np.empty(len(X_te), dtype=object)
    for i in range(len(X_tr)): Xs_tr[i] = (X_tr[i].reshape(w,1), np.full(w, T_tr[i]))
    for i in range(len(X_te)): Xs_te[i] = (X_te[i].reshape(w,1), np.full(w, T_te[i]))

    svr_temporal = SVR(kernel=temporal_kernel, lambda_= 1e-2, epsilon = 0.05)
    svr_temporal.fit(Xs_tr, y_tr)
    y_pred_temporal = svr_temporal.predict(Xs_te)

    print_metrics(y_pred_rbf, y_te, "SVR RBF")
    print_metrics(y_pred_temporal, y_te, "SVR RBF temporal")
    """
    SVR RBF MSE: 0.0841, SE: 0.0044
    SVR RBF temporal MSE: 0.0003, SE: 0.0000
    """
    err_flat = (y_pred_rbf - y_te)**2
    err_temp = (y_pred_temporal - y_te)**2

    improvement = err_flat - err_temp
    top_ix = np.argsort(improvement)[-6:][::-1]
    plot_predictions_synthetic(top_ix, y_pred_rbf, y_pred_temporal, X_te, T_te, y_te,  w = w)

def plot_predictions_synthetic(top_ix, y_pred_rbf, y_pred_temporal, X_te, T_te, y_te,  w = 48):

    top6 = top_ix 
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False, sharey=False)
    for ax, idx in zip(axes.ravel(), top6):
        seq = X_te[idx]
        ax.plot(seq, color='k', label='Sequence')

        bump_idx = int(np.floor(T_te[idx] * (w-1)))
        ax.scatter(bump_idx, y_te[idx], color='k', marker='x', s=80, label='True loc')
        ax.axvline(bump_idx, color='gray', linestyle='--')

        flat_idx = int(round(w * y_pred_rbf[idx]))
        ax.scatter(flat_idx, y_pred_rbf[idx], color='#b0b0b0', marker='o', s=60, label='Flat pred')

        temp_idx = int(round(w * y_pred_temporal[idx]))
        ax.scatter(temp_idx, y_pred_temporal[idx], color='#e74c3c', marker='s', s=60, label='Temp pred')
        ax.set_title(f"Idx {idx}")
        ax.set_xlim(0, w-1)
        ax.set_ylim(seq.min() - 1, seq.max() + 1)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
    fig.text(0.5, 0.04, 'Time index', ha='center')
    fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig("synthetic_data_svr_1.pdf")
    plt.show()

def print_metrics(y_true, y_pred, model_name):
    errors = (y_true-y_pred) ** 2
    mse = errors.mean()
    se = errors.std(ddof=1) / np.sqrt(len(errors))
    print(f"{model_name} MSE: {mse:.4f}, SE: {se:.4f}")

def build_temporal_input(X_array, period_array, w = 10):
    temporal_input = []
    for x, end_date in zip(X_array, period_array):
        # get month data bsed on w 
        t = pd.date_range(end=end_date, periods=w, freq='M').to_pydatetime()
        # convert to days
        t_days = np.array([(d - t[0]).days for d in t], dtype=int).reshape(-1, 1)
        # group together with feature values
        temporal_input.append((x.reshape(-1, 1), t_days))
    return temporal_input

def process_data(): 

    df = pd.read_csv("Month_Value_1.csv")
    df = df.dropna().reset_index(drop=True)
    df['Period'] = pd.to_datetime(df['Period'])
    df = df.sort_values('Period').reset_index(drop=True)

    w = 10 # define month period 
    y = df['Revenue'].values.astype(float)
    periods = df['Period'].values

    # group data by length w 
    X_raw = []
    y_raw = []
    period_raw = []

    for i in range(w, len(y)):
        X_raw.append(y[i-w:i])      
        y_raw.append(y[i])            # raw target
        period_raw.append(periods[i]) # actual period

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)
    period_raw = np.array(period_raw)

    # split and keep ordering 
    split_idx = int(0.8 * len(X_raw))

    X_train_raw = X_raw[:split_idx]
    y_train = y_raw[:split_idx]
    period_train = period_raw[:split_idx]

    X_test_raw = X_raw[split_idx:]
    y_test = y_raw[split_idx:]
    period_test = period_raw[split_idx:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)     
    X_test = scaler.transform(X_test_raw)    

    X_train_temporal = build_temporal_input(X_train, period_train)
    X_test_temporal = build_temporal_input(X_test, period_test)

    target_scaler = MinMaxScaler()
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return X_train, X_test, X_train_temporal, X_test_temporal, y_train, y_test

def predict_revenue():

    X_train, X_test, X_train_temporal, X_test_temporal, y_train, y_test = process_data()

    rbf_kernel = RBF(sigma=1.0)
    temporal_kernel = TemporalRBF(sigma_f=1.0, sigma_t=1.0)

    # krr with rbf
    krr_rbf = KernelizedRidgeRegression(kernel=rbf_kernel, lambda_=100)
    krr_rbf.fit(X_train, y_train)
    y_pred_rbf = krr_rbf.predict(X_test)

    # krr with temporal rbf
    krr_temporal = KernelizedRidgeRegression(kernel=temporal_kernel, lambda_=100)
    krr_temporal.fit(X_train_temporal, y_train)
    y_pred_temporal = krr_temporal.predict(X_test_temporal)

    # svr with rbf
    svr_rbf = SVR(kernel=rbf_kernel, lambda_= 100, epsilon = 0.1)
    svr_rbf.fit(X_train, y_train)
    y_pred_svr_rbf = svr_rbf.predict(X_test)

    # svr wtih temporal rbf
    svr_temporal = SVR(kernel=temporal_kernel, lambda_= 100, epsilon=0.1)
    svr_temporal.fit(X_train_temporal, y_train)
    y_pred_svr_temporal = svr_temporal.predict(X_test_temporal)

    print_metrics(y_test, y_pred_rbf, "KRR RBF")
    print_metrics(y_test, y_pred_temporal, "KRR Temporal")
    print_metrics(y_test, y_pred_svr_rbf, "SVR RBF")
    print_metrics(y_test, y_pred_svr_temporal, "SVR Temporal")
    """
    KRR RBF MSE: 0.6077, SE: 0.0872
    KRR Temporal MSE: 0.1295, SE: 0.0365
    SVR RBF MSE: 0.1675, SE: 0.0421
    SVR Temporal MSE: 0.0861, SE: 0.0273
    """

if __name__ == "__main__":

    # PART 1
    fit_sine()
    # PART 2
    housing_data()
    # PART 3
    synthetic_data_test()
    predict_revenue()