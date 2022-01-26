# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 09:33:20 2022

@author: Administrator
"""
import pandas as pd
import math

class LaplacianNB():
    """
    Laplacian naive bayes
    """
    def __init__(self):
        """
        """

    def train(self, X, y):
        """
        Training laplacian naive bayes classifier with traning set (X, y).
        Input:
            X: list of instances. Each instance is represented by ()
            y: list of labels. 0 represents bad, 1 represents good.
        """
        N = len(y)
        self.classes = self.count_list(y)
        self.class_num = len(self.classes)
        self.classes_p = {}
        # p_{hat}(c)
        for c, n in self.classes.items():
            self.classes_p[c] = float(n+1) / (N+self.class_num)

        self.discrete_attris_with_good_p = []
        self.discrete_attris_with_bad_p = []
        for i in range(6):
            attr_with_good = []
            attr_with_bad = []
            for j in range(N):
                if y[j] == "是":
                     attr_with_good.append(X[j][i])
                else:
                    attr_with_bad.append(X[j][i])
            unique_with_good = self.count_list(attr_with_good)
            unique_with_bad = self.count_list(attr_with_bad)
            self.discrete_attris_with_good_p.append(self.discrete_p(unique_with_good, self.classes["是"]))
            self.discrete_attris_with_bad_p.append(self.discrete_p(unique_with_bad, self.classes["否"]))

        self.good_mus = []
        self.good_vars = []
        self.bad_mus = []
        self.bad_vars = []
        for i in range(2):
            attr_with_good = []
            attr_with_bad = []
            for j in range(N):
                if y[j] == "是":
                    attr_with_good.append(X[j][i+6])
                else:
                    attr_with_bad.append(X[j][i+6])
            good_mu, good_var = self.mu_var_of_list(attr_with_good)
            bad_mu, bad_var = self.mu_var_of_list(attr_with_bad)
            self.good_mus.append(good_mu)
            self.good_vars.append(good_var)
            self.bad_mus.append(bad_mu)
            self.bad_vars.append(bad_var)

    def predict(self, x):
        """
        """
        p_good = self.classes_p["是"]
        p_bad = self.classes_p["否"]
        for i in range(6):
            p_good  *= self.discrete_attris_with_good_p[i][x[i]]
            p_bad *= self.discrete_attris_with_bad_p[i][x[i]]
        for i in range(2):
            p_good *= self.continuous_p(x[i+6], self.good_mus[i], self.good_vars[i])
            p_bad *= self.continuous_p(x[i+6], self.bad_mus[i], self.bad_vars[i])
        if p_good >= p_bad:
            return p_good, p_bad, "是"
        else:
            return p_good, p_bad, "否"

    def count_list(self, l):
        """
        Get unique elements in list and corresponding count.
        """
        unique_dict = {}
        for e in l:
            if e in unique_dict:
                unique_dict[e] += 1
            else:
                unique_dict[e] = 1
        return unique_dict


    def discrete_p(self, d, N_class):
        """
        Compute discrete attribution probability
        """
        new_d = {}
        #print d
        for a, n in d.items():
            new_d[a] = float(n+1) / (N_class + len(d))
        return new_d

    def continuous_p(self, x, mu, var):
        p = 1.0 / (math.sqrt(2*math.pi) * math.sqrt(var)) * math.exp(- (x-mu)**2 /(2*var))
        return p

    def mu_var_of_list(self, l):
        mu = sum(l) / float(len(l))
        var = 0
        for i in range(len(l)):
            var += (l[i]-mu)**2
        var = var / float(len(l))
        return mu, var

if __name__=="__main__":
    lnb = LaplacianNB()
    workbook = pd.read_csv("./watermelon_3.csv", encoding="gb18030")
    X = workbook.values[:, 1:9]
    y = workbook.values[:, 9]
    #print X, y
    lnb.train(X, y)  # generate dictionary
    #print lnb.discrete_attris_with_good_p
    label = lnb.predict(["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460])
    print ("predict result: ", label)