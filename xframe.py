import pandas as pd
from os import cpu_count
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

"""
    Extended DataFrame: This is an extension for Pandas DataFrame that adds
    some specific data cleansing functionality not included in the original
    class.
"""
class XFrame(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        super(XFrame, self).__init__(*args, **kwargs)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*allow columns to be created.*")
            self.numeric_cols = list(self.columns)
            

        self.col_num = len(self.numeric_cols)

        """
            To be used in parallelizable methods, identifying whether
            or not it is possible to use multiple CPUs in the current
            machine.
        """
        self.really_parallel = cpu_count() > 1

    def report(self):
        N, n_valid, sums, sums_sq, sums_xy, mins, maxes, consts, n_bins = self.__first_pass__()
        sum_devs, sum_sq_devs, sum_prod_devs, bins = self.__second_pass__(n_valid, sums, consts, n_bins, mins, maxes)
        corr = self.__calculate__(consts, sum_sq_devs, sum_prod_devs)
        info = self.__create_info__(N, n_valid, mins, maxes, consts, sums, sum_sq_devs, corr, bins)

        self.exhibit_info(info)
        self.exhibit_correlogram(corr, bins)
        
    def exhibit_info(self, info):
        print(info)
        print()

    def exhibit_correlogram(self, corr, bins, invert_alpha=False):
        n = len(self.columns)
        fig, axes = plt.subplots(n, n)

        for x in range(n):
            for y in range(n):
                ax = axes[x, y]
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())

                if y > x:
                    alpha = abs(corr[x,y])
                    if invert_alpha:
                        alpha = 1 - alpha

                    alpha = alpha * 0.8 + 0.2 

                    font = {'size': 20, 'alpha': alpha}
                    plt.text(0.5, 0.5, f'{corr[x,y]:.2f}', ha='center', va='center', fontdict=font, transform=ax.transAxes)
                elif x == y:
                    ax.bar(range(len(bins[x])), bins[x])
                elif y < x:
                    sns.kdeplot(self[self.numeric_cols[x]], y=self[self.numeric_cols[y]], ax=ax, fill=True)

        plt.show()

    def __first_pass__(self):
        sums = np.zeros(self.col_num, dtype=float)
        sums_sq = np.zeros(self.col_num, dtype=float)
        sums_xy = np.zeros((self.col_num,self.col_num), dtype=float)

        mins = np.ones(self.col_num, dtype=np.float) * math.inf
        maxes = np.ones(self.col_num, dtype=np.float) * -math.inf
        
        n_valid = np.zeros(self.col_num)
        N = 0

        values_0 = None
        consts = np.array([True] * self.col_num)

        for _, row in self.iterrows():
            N += 1

            values = np.array(row[self.numeric_cols].values, dtype=float)

            if values_0 is None:
                values_0 = values

            consts = np.logical_and(consts, (values == values_0))

            sums += values
            sums_sq += values ** 2

            mins = np.minimum(mins, values)
            maxes = np.maximum(maxes, values)

            for x in range(self.col_num):
                if not np.isnan(values[x]):
                    n_valid[x] += 1
                
                for y in range(x, self.col_num):
                    if (not np.isnan(values[x])) and (not np.isnan(values[y])):
                        sums_xy[x, y] += values[x] * values[y]

        n_bins = np.ceil(1 + np.log2(n_valid)).astype(int)

        return N, n_valid, sums, sums_sq, sums_xy, mins, maxes, consts, n_bins

    def __second_pass__(self, n_valid, sums, consts, n_bins, mins, maxes):
        means = sums / n_valid

        bins = []

        for n in n_bins:
            bins.append(np.zeros(n))

        sum_devs = np.zeros(self.col_num, dtype=float)
        sum_sq_devs = np.zeros(self.col_num, dtype=float)

        sum_prod_devs = np.zeros((self.col_num, self.col_num))

        for _, row in self.iterrows():
            values = np.array(row[self.numeric_cols].values, dtype=float)

            devs = values - means
            sq_devs = devs ** 2

            sum_devs += devs
            sum_sq_devs += sq_devs

            for x in range(self.col_num):
                if (np.isnan(values[x])) or (consts[x]):
                    continue

                mx = maxes[x] + abs(maxes[x]) * 0.01
                mn = mins[x] - abs(mins[x]) * 0.01

                bin = math.floor((values[x] - mn) / (mx - mn) * n_bins[x])
                bins[x][bin] += 1
                
                for y in range(x, self.col_num):
                    if (np.isnan(values[x])) or (consts[y]):
                        continue

                    sum_prod_devs[x, y] += devs[x] * devs[y]

        return sum_devs, sum_sq_devs, sum_prod_devs, bins

    def __calculate__(self, consts, sum_sq_devs, sum_prod_devs):
        corr = np.zeros((self.col_num, self.col_num))

        for x in range(self.col_num):
            if consts[x]:
                continue
            
            for y in range(x, self.col_num):
                if consts[y]:
                    continue

                corr[x, y] = sum_prod_devs[x, y] / np.sqrt(sum_sq_devs[x] *(sum_sq_devs[y]))

        return corr

    def __create_info__(self, N, n_valid, mins, maxes, consts, sums, sum_sq_devs, corr, bins):
        means = sums / n_valid
        stds = np.sqrt(sum_sq_devs / n_valid)
        faulting = (N - n_valid).astype(int)
        faulting_percent = faulting.astype(float) / N * 100
        ns = n_valid.astype(int)

        cols = ['Variable', 'Mean', 'Std', 'N', 'Faulting', 'Faulting%', 'Min', 'Max']

        info = [self.columns, means, stds, ns, faulting, faulting_percent, mins, maxes]
        info = np.array(info)
        info = info.transpose()
        info = pd.DataFrame(info, columns=cols)

        return info