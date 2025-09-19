# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pytensor.tensor as pt
import numpy as np
from pymc import Binomial, Deterministic, HalfNormal, Model, Multinomial, Normal, sample


def phi(x):
    """Cumulative normal distribution."""
    return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2))


def hmetad_groupLevel(
    data, sample_model=True, num_samples: int = 1000, num_chains: int = 4, **kwargs
):
    """Hierarchical Bayesian modeling of meta-d' (group level).

    This is an internal function. The group level model must be called using
    :py:func:`metadpy.bayesian.hmetad`.

    Parameters
    ----------
    data : dict
        Response data.
    sample_model : boolean
        If `False`, only the model is returned without sampling.
    num_samples : int
        The number of samples per chains to draw (defaults to `1000`).
    num_chains : int
        The number of chains (defaults to `4`).
    **kwargs : keyword arguments
        All keyword arguments are passed to `func::pymc.sampling.sample`.

    Returns
    -------
    model : :py:class:`pymc.Model` instance
        The pymc model. Encapsulates the variables and likelihood factors.
    trace : :py:class:`pymc.backends.base.MultiTrace` or
        :py:class:`arviz.InferenceData`
        A `MultiTrace` or `ArviZ InferenceData` object that contains the samples.

    References
    ----------
    .. [#] Fleming, S.M. (2017) HMeta-d: hierarchical Bayesian estimation of
    metacognitive efficiency from confidence ratings, Neuroscience of Consciousness,
    3(1) nix007, https://doi.org/10.1093/nc/nix007

    """
    nRatings = data["nRatings"]
    nSubj = data["nSubj"]
    
    with Model() as model:

        # Group-level hyperpriors
        mu_c1 = Normal("mu_c1", mu=0.0, sigma=1)
        sigma_c1 = HalfNormal("sigma_c1", sigma=1)
        
        mu_d1 = Normal("mu_d1", mu=0.0, sigma=1)
        sigma_d1 = HalfNormal("sigma_d1", sigma=1)

        mu_logMratio = Normal("mu_logMratio", mu=0.0, sigma=1)
        sigma_logMratio = HalfNormal("sigma_logMratio", sigma=1)

        # Group-level hyperpriors for criteria
        mu_cS1_hn = HalfNormal("mu_cS1_hn", sigma=1, shape=nRatings - 1)
        sigma_cS1_hn = HalfNormal("sigma_cS1_hn", sigma=1)
        
        mu_cS2_hn = HalfNormal("mu_cS2_hn", sigma=1, shape=nRatings - 1)
        sigma_cS2_hn = HalfNormal("sigma_cS2_hn", sigma=1)

        # Subject-level parameters - each drawn from group distribution
        c1 = Normal("c1", mu=mu_c1, sigma=sigma_c1, shape=nSubj)
        d1 = Normal("d1", mu=mu_d1, sigma=sigma_d1, shape=nSubj)
        logMratio = Normal("logMratio", mu=mu_logMratio, sigma=sigma_logMratio, shape=nSubj)
        
        # Transform to meta_d
        meta_d = Deterministic("meta_d", d1 * pt.exp(logMratio))

        # Subject-level criteria
        cS1_hn = HalfNormal("cS1_hn", sigma=sigma_cS1_hn, shape=(nSubj, nRatings - 1))
        cS2_hn = HalfNormal("cS2_hn", sigma=sigma_cS2_hn, shape=(nSubj, nRatings - 1))

        # For each subject, fit the exact same model as the original subject-level model
        for s in range(nSubj):
            # Get subject-specific parameters
            c1_s = c1[s]
            d1_s = d1[s]
            meta_d_s = meta_d[s]
            
            # Subject-specific criteria using the exact same pattern as original
            cS1_s = Deterministic(f"cS1_{s}", pt.sort(-cS1_hn[s, :]) + (c1_s - data["Tol"]))
            cS2_s = Deterministic(f"cS2_{s}", pt.sort(cS2_hn[s, :]) + (c1_s - data["Tol"]))

            # TYPE 1 SDT BINOMIAL MODEL (exact same as original)
            h_s = phi(d1_s / 2 - c1_s)
            f_s = phi(-d1_s / 2 - c1_s)
            H_s = Binomial(f"H_{s}", data["s"][s], h_s, observed=data["hits"][s])
            FA_s = Binomial(f"FA_{s}", data["n"][s], f_s, observed=data["falsealarms"][s])

            # TYPE 2 SDT MODEL - exact same calculations as original
            # Means of SDT distributions
            S2mu = pt.flatten(meta_d_s / 2, 1)
            S1mu = pt.flatten(-meta_d_s / 2, 1)

            # Calculate normalisation constants
            C_area_rS1 = phi(c1_s - S1mu)
            I_area_rS1 = phi(c1_s - S2mu)
            C_area_rS2 = 1 - phi(c1_s - S2mu)
            I_area_rS2 = 1 - phi(c1_s - S1mu)

            # Get nC_rS1 probs - exact same pattern as original
            nC_rS1 = phi(cS1_s - S1mu) / C_area_rS1
            nC_rS1 = Deterministic(
                f"nC_rS1_{s}",
                pt.concatenate(
                    (
                        [
                            phi(cS1_s[0] - S1mu) / C_area_rS1,
                            nC_rS1[1:] - nC_rS1[:-1],
                            (
                                (phi(c1_s - S1mu) - phi(cS1_s[(nRatings - 2)] - S1mu))
                                / C_area_rS1
                            ),
                        ]
                    ),
                    axis=0,
                ),
            )

            # Get nI_rS2 probs - exact same pattern as original
            nI_rS2 = (1 - phi(cS2_s - S1mu)) / I_area_rS2
            nI_rS2 = Deterministic(
                f"nI_rS2_{s}",
                pt.concatenate(
                    (
                        [
                            ((1 - phi(c1_s - S1mu)) - (1 - phi(cS2_s[0] - S1mu))) / I_area_rS2,
                            nI_rS2[:-1] - (1 - phi(cS2_s[1:] - S1mu)) / I_area_rS2,
                            (1 - phi(cS2_s[nRatings - 2] - S1mu)) / I_area_rS2,
                        ]
                    ),
                    axis=0,
                ),
            )

            # Get nI_rS1 probs - exact same pattern as original  
            nI_rS1 = (-phi(cS1_s - S2mu)) / I_area_rS1
            nI_rS1 = Deterministic(
                f"nI_rS1_{s}",
                pt.concatenate(
                    (
                        [
                            phi(cS1_s[0] - S2mu) / I_area_rS1,
                            nI_rS1[:-1] + (phi(cS1_s[1:] - S2mu)) / I_area_rS1,
                            (phi(c1_s - S2mu) - phi(cS1_s[(nRatings - 2)] - S2mu)) / I_area_rS1,
                        ]
                    ),
                    axis=0,
                ),
            )

            # Get nC_rS2 probs - exact same pattern as original
            nC_rS2 = (1 - phi(cS2_s - S2mu)) / C_area_rS2
            nC_rS2 = Deterministic(
                f"nC_rS2_{s}",
                pt.concatenate(
                    (
                        [
                            ((1 - phi(c1_s - S2mu)) - (1 - phi(cS2_s[0] - S2mu))) / C_area_rS2,
                            nC_rS2[:-1] - ((1 - phi(cS2_s[1:] - S2mu)) / C_area_rS2),
                            (1 - phi(cS2_s[nRatings - 2] - S2mu)) / C_area_rS2,
                        ]
                    ),
                    axis=0,
                ),
            )

            # Avoid underflow of probabilities - exact same as original
            nC_rS1 = pt.switch(nC_rS1 < data["Tol"], data["Tol"], nC_rS1)
            nI_rS2 = pt.switch(nI_rS2 < data["Tol"], data["Tol"], nI_rS2)
            nI_rS1 = pt.switch(nI_rS1 < data["Tol"], data["Tol"], nI_rS1)
            nC_rS2 = pt.switch(nC_rS2 < data["Tol"], data["Tol"], nC_rS2)

            # TYPE 2 SDT MODEL Multinomial likelihood - exact same pattern as original
            subject_counts = data["counts"][s, :]
            
            Multinomial(
                f"CR_counts_{s}",
                n=data["cr"][s],
                p=nC_rS1,
                shape=nRatings,
                observed=subject_counts[:nRatings],
            )
            Multinomial(
                f"FA_counts_{s}",
                n=data["falsealarms"][s],
                p=nI_rS2,
                shape=nRatings,
                observed=subject_counts[nRatings : nRatings * 2],
            )
            Multinomial(
                f"M_counts_{s}",
                n=data["m"][s],
                p=nI_rS1,
                shape=nRatings,
                observed=subject_counts[nRatings * 2 : nRatings * 3],
            )
            Multinomial(
                f"H_counts_{s}",
                n=data["hits"][s],
                p=nC_rS2,
                shape=nRatings,
                observed=subject_counts[nRatings * 3 : nRatings * 4],
            )

        if sample_model is True:
            trace = sample(
                return_inferencedata=True,
                chains=num_chains,
                draws=num_samples,
                **kwargs
            )

            return model, trace

        else:
            return model