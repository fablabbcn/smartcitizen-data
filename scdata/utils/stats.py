from scipy import stats

def spearman(x, y):
     # correlation to distance: range 0 to 2
    r = stats.pearsonr(x, y)[0]
    return 1 - r