from scipy.stats import pearsonr
from numpy import mean, std, power
from sklearn.metrics import r2_score
from math import sqrt

def spearman(x, y):
    # correlation to distance: range 0 to 2
    r = stats.pearsonr(x, y)[0]
    return 1 - r

def get_metrics(reference, estimation):
    '''
        Makes a dictionary with important prediction quality metrics

        Parameters
        ----------
            reference: numpy.array
                Reference, ground truth or measurand
            estimation: numpy.array
                Estimation of that reference
        Returns
        -------
            Metrics dictionary containing:
                - reference's and estimation's average (avg)
                - reference's and estimation's std deviation (sigma)
                - bias (difference between averages)
                - normalised bias: bias/sigma_ref
                - normalised std deviation
                - r2 score
                - root mean squared deviation (rmsd)
                - rmsd normalised and ubiased
        '''
    metricsd = dict()
    
    # Average
    avg_ref = mean(reference)
    avg_est = mean(estimation)
    metricsd['avg_ref'] = avg_ref
    metricsd['avg_est'] = avg_est

    # Standard deviation
    sigma_ref = std(reference)
    sigma_est = std(estimation)
    metricsd['sig_ref'] = sigma_ref
    metricsd['sig_est'] = sigma_est
    
    # Bias
    bias = avg_est-avg_ref
    normalised_bias = float((avg_est-avg_ref)/sigma_ref)
    metricsd['bias'] = bias
    metricsd['normalised_bias'] = normalised_bias
    
    # Normalized std deviation
    sigma_norm = sigma_est/sigma_ref
    sign_sigma = (sigma_est-sigma_ref)/(abs(sigma_est-sigma_ref))
    metricsd['sigma_norm'] = sigma_norm
    metricsd['sign_sigma'] = sign_sigma

    # R2
    SS_Residual = sum((estimation-reference)**2)
    SS_Total = sum((reference-mean(reference))**2)
    rsquared = max(0, 1 - (float(SS_Residual))/SS_Total)
    metricsd['rsquared'] = rsquared
    metricsd['r2_score_sklearn'] = r2_score(estimation, reference) 
    # RMSD
    RMSD = sqrt((1./len(reference))*SS_Residual)
    RMSD_norm_unb = sqrt(1+power(sigma_norm,2)-2*sigma_norm*sqrt(rsquared))
    metricsd['RMSD'] = RMSD
    metricsd['RMSD_norm_unb'] = RMSD_norm_unb
    
    return metricsd