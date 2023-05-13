import numpy as np
import scipy.optimize as optim
import scipy.interpolate as polate

def CDS_bootstrap(cds_spreads, yield_curve, cds_tenor, yield_tenor, prem_per_year, R):
    '''
    Bootstraps a credit curve from CDS spreads of varying maturities. Returns the hazard
    rate values and survival probabilities corresponding to the CDS maturities.

    Args:
        cds_spreads :   vector of CDS spreads
        yield_curve :   vector of risk-free bond yields
        cds_tenor :     vector of maturities corresponding to the given CDS spreads
        yield_tenor :   vector of risk-free bond yield tenor matching yield_curve
        prem_per_year : premiums paid per year on the CDS (i.e. annualy=1, semiannually=2, quarterly=4, monthly=12)
        R :             recovery rate
    '''
    # Checks
    if len(cds_spreads) != len(cds_tenor):
        print("CDS spread array does match CDS tenor array.")
        return None

    if len(yield_curve) != len(yield_tenor):
        print("Yield curve array does not match yield tenor.")
        return None

    # Interpolation/Extrapolation function
    interp = polate.interp1d(yield_tenor, yield_curve, 'linear', fill_value='extrapolate')

    # The bootstrap function
    def bootstrap(h, given_haz, s, cds_tenor, yield_curve, prem_per_year, R):
        '''
        Returns the difference between values of payment leg and default leg.
        '''
        a = 1 / prem_per_year
        maturities = [0] + list(cds_tenor)
        pmnt = 0;
        dflt = 0;
        auc = 0
        # 1. Calculate value of payments for given hazard rate curve values
        for i in range(1, len(maturities) - 1):
            num_points = int((maturities[i] - maturities[i - 1]) * prem_per_year + 1)
            t = np.linspace(maturities[i - 1], maturities[i], num_points)
            r = interp(t)

            for j in range(1, len(t)):
                surv_prob_prev = np.exp(-given_haz[i - 1] * (t[j - 1] - t[0]) - auc)
                surv_prob_curr = np.exp(-given_haz[i - 1] * (t[j] - t[0]) - auc)
                pmnt += s * a * np.exp(-r[j] * t[j]) * 0.5 * (surv_prob_prev + surv_prob_curr)
                dflt += np.exp(-r[j] * t[j]) * (1 - R) * (surv_prob_prev - surv_prob_curr)

            # hazard rate for the previous period
            auc += (t[-1] - t[0]) * given_haz[i - 1]

        # 2. Set up calculations for payments with the unknown hazard rate value
        num_points = int((maturities[-1] - maturities[-2]) * prem_per_year + 1)
        t = np.linspace(maturities[-2], maturities[-1], num_points)
        r = interp(t)

        for i in range(1, len(t)):
            surv_prob_prev = np.exp(-h * (t[i - 1] - t[0]) - auc)
            surv_prob_curr = np.exp(-h * (t[i] - t[0]) - auc)
            pmnt += s * a * np.exp(-r[i] * t[i]) * 0.5 * (surv_prob_prev + surv_prob_curr)
            dflt += np.exp(-r[i] * t[i]) * (1 - R) * (surv_prob_prev - surv_prob_curr)

        return abs(pmnt - dflt)

    haz_rates = []
    surv_prob = []
    t = [0] + list(cds_tenor)

    for i in range(len(cds_spreads)):
        get_haz = lambda x: bootstrap(x, haz_rates, cds_spreads[i], cds_tenor[0:i + 1], yield_curve[0:i + 1],
                                      prem_per_year, R)
        haz = round(optim.minimize(get_haz, cds_spreads[i] / (1 - R), method='SLSQP', tol=1e-10).x[0], 8)
        cond_surv = (t[i + 1] - t[i]) * haz
        haz_rates.append(haz)
        surv_prob.append(cond_surv)

    return haz_rates, np.exp(-np.cumsum(surv_prob))
