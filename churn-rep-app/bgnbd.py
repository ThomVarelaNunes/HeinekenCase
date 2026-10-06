"""
BG/NBD ("Buy Till You Die") written out by hand, so you can see how it works.
Reference: Fader, Hardie & Lee (2005), "Counting your customers the easy way".

The story the model tells about every account:
  * While "alive", it orders at its own random rate (rates differ between accounts:
    a gamma distribution with parameters r, alpha).
  * After every order there is a chance p that it "dies" (stops for good). p also
    differs between accounts (a beta distribution with parameters a, b).

We never see who died. The model infers it from three numbers per account:
    x   = repeat orders            (n_orders - 1)
    t_x = when the last order was  (weeks since the first order)
    T   = how long we've known it  (weeks since the first order, up to the cutoff)

It uses ONLY the ordering rhythm -- no reviews, delivery, spend. That makes it a strong,
explainable baseline: anything the other models add on top is the value of the extra data.

It looks like any scikit-learn classifier (fit / predict_proba), so step 6 can treat it
like the other models. predict_proba gives P(no order in the next CHURN_DAYS).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.base import BaseEstimator, ClassifierMixin

import config


def _rfm(X):
    """Turn our feature table into the model's x, t_x, T (in weeks)."""
    x = (X["n_orders"] - 1).clip(lower=0).to_numpy(float)
    T = X["tenure_days"].to_numpy(float) / 7
    t_x = (X["tenure_days"] - X["days_since_last"]).to_numpy(float) / 7
    return x, t_x, T


class BGNBD(BaseEstimator, ClassifierMixin):
    def __init__(self, horizon_days=config.CHURN_DAYS, penalizer=0.001):
        self.horizon_days = horizon_days
        self.penalizer = penalizer

    def _neg_log_likelihood(self, log_params, x, t_x, T):
        r, alpha, a, b = np.exp(log_params)
        A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        A3 = -(r + x) * np.log(alpha + T)
        A4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-9)) - (r + x) * np.log(alpha + t_x), -np.inf)
        ll = A1 + A2 + np.logaddexp(A3, A4)
        return -ll.sum() + self.penalizer * np.sum(np.exp(log_params) ** 2)

    def fit(self, X, y=None):
        # y is not used: the model learns from the ordering pattern alone
        x, t_x, T = _rfm(X)
        res = minimize(self._neg_log_likelihood, x0=np.zeros(4), args=(x, t_x, T), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
        self.r_, self.alpha_, self.a_, self.b_ = np.exp(res.x)
        self.classes_ = np.array([0, 1])
        return self

    def p_alive(self, X):
        x, t_x, T = _rfm(X)
        r, alpha, a, b = self.r_, self.alpha_, self.a_, self.b_
        ratio = np.where(x > 0, a / np.maximum(b + x - 1, 1e-9) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0)
        return 1 / (1 + ratio)

    def predict_proba(self, X):
        x, t_x, T = _rfm(X)
        t = self.horizon_days / 7
        alive = self.p_alive(X)
        # P(no order in next t weeks) = P(dead) + P(alive) * P(alive but just doesn't order)
        quiet_if_alive = ((self.alpha_ + T) / (self.alpha_ + T + t)) ** (self.r_ + x)
        p_churn = (1 - alive) + alive * quiet_if_alive
        return np.column_stack([1 - p_churn, p_churn])
