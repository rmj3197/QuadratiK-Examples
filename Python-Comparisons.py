##############################################################################
#       COMPARISON OF MMD AND ENERGY STATISTICS ON THE WINE DATASET          #
##############################################################################
from QuadratiK.datasets import load_wine_data

X = load_wine_data()
X1 = X[X["Class"] == 1].drop(columns=["Class"]).values
X2 = X[X["Class"] == 2].drop(columns=["Class"]).values
X3 = X[X["Class"] == 3].drop(columns=["Class"]).values

from hyppo.ksample import KSample

# K-Sample Energy Test
stat, pvalue = KSample("Dcorr").test(X1, X2, X3)
print("K-Sample Energy Test: stat = ", stat, ", pvalue = ", pvalue)

# K-Sample MMD Test
stat, pvalue = KSample("HSic").test(X1, X2, X3)
print("K-Sample MMD Test: stat = ", stat, ", pvalue = ", pvalue)

##############################################################################
#                    END OF COMPARISON ON WINE DATASET                       #
##############################################################################
