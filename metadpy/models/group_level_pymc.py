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
        # Type 1 hyperpriors
        mu_c1 = Normal("mu_c1", mu=0.0, sigma=1)
        sigma_c1 = HalfNormal("sigma_c1", sigma=1)
        
        mu_d1 = Normal("mu_d1", mu=0.0, sigma=1)
        sigma_d1 = HalfNormal("sigma_d1", sigma=1)

        # Subject-level Type 1 parameters
        c1 = Normal("c1", mu=mu_c1, sigma=sigma_c1, shape=nSubj)
        d1 = Normal("d1", mu=mu_d1, sigma=sigma_d1, shape=nSubj)

        # TYPE 1 SDT BINOMIAL MODEL
        h = phi(d1 / 2 - c1)
        f = phi(-d1 / 2 - c1)
        
        # Observed data for each subject
        H = Binomial("H", data["s"], h, observed=data["hits"])
        FA = Binomial("FA", data["n"], f, observed=data["falsealarms"])

        # Type 2 hyperpriors  
        mu_logMratio = Normal("mu_logMratio", mu=0.0, sigma=1)
        sigma_logMratio = HalfNormal("sigma_logMratio", sigma=1)
        
        # Subject-level Type 2 parameters
        logMratio = Normal("logMratio", mu=mu_logMratio, sigma=sigma_logMratio, shape=nSubj)
        meta_d = Deterministic("meta_d", d1 * pt.exp(logMratio))

        # Specify ordered prior on criteria bounded above and below by Type 1 c1
        # Group-level hyperpriors for criteria
        mu_cS1_hn = HalfNormal("mu_cS1_hn", sigma=1, shape=nRatings - 1)
        sigma_cS1_hn = HalfNormal("sigma_cS1_hn", sigma=1)
        
        mu_cS2_hn = HalfNormal("mu_cS2_hn", sigma=1, shape=nRatings - 1)
        sigma_cS2_hn = HalfNormal("sigma_cS2_hn", sigma=1)

        # Subject-level criteria
        cS1_hn = HalfNormal(
            "cS1_hn",
            sigma=sigma_cS1_hn,
            shape=(nSubj, nRatings - 1),
        )
        cS2_hn = HalfNormal(
            "cS2_hn", 
            sigma=sigma_cS2_hn,
            shape=(nSubj, nRatings - 1),
        )

        cS1 = Deterministic("cS1", pt.sort(-cS1_hn, axis=1) + pt.expand_dims(c1 - data["Tol"], 1))
        cS2 = Deterministic("cS2", pt.sort(cS2_hn, axis=1) + pt.expand_dims(c1 - data["Tol"], 1))

        # TYPE 2 SDT MODEL (META-D) for each subject
        for s in range(nSubj):
            # Means of SDT distributions for this subject
            S2mu = meta_d[s] / 2
            S1mu = -meta_d[s] / 2

            # Calculate normalisation constants for this subject
            C_area_rS1 = phi(c1[s] - S1mu)
            I_area_rS1 = phi(c1[s] - S2mu)
            C_area_rS2 = 1 - phi(c1[s] - S2mu)
            I_area_rS2 = 1 - phi(c1[s] - S1mu)

            # Get nC_rS1 probs for subject s (following original implementation)
            nC_rS1_temp = phi(cS1[s, :] - S1mu) / C_area_rS1
            nC_rS1_probs = pt.concatenate([
                [phi(cS1[s, 0] - S1mu) / C_area_rS1],
                nC_rS1_temp[1:] - nC_rS1_temp[:-1],
                [(phi(c1[s] - S1mu) - phi(cS1[s, nRatings - 2] - S1mu)) / C_area_rS1]
            ])

            # Get nI_rS2 probs for subject s
            nI_rS2_temp = (1 - phi(cS2[s, :] - S1mu)) / I_area_rS2
            nI_rS2_probs = pt.concatenate([
                [((1 - phi(c1[s] - S1mu)) - (1 - phi(cS2[s, 0] - S1mu))) / I_area_rS2],
                nI_rS2_temp[:-1] - (1 - phi(cS2[s, 1:] - S1mu)) / I_area_rS2,
                [(1 - phi(cS2[s, nRatings - 2] - S1mu)) / I_area_rS2]
            ])

            # Get nI_rS1 probs for subject s  
            nI_rS1_temp = phi(cS1[s, :] - S2mu) / I_area_rS1
            nI_rS1_probs = pt.concatenate([
                [phi(cS1[s, 0] - S2mu) / I_area_rS1],
                nI_rS1_temp[:-1] + phi(cS1[s, 1:] - S2mu) / I_area_rS1,
                [(phi(c1[s] - S2mu) - phi(cS1[s, nRatings - 2] - S2mu)) / I_area_rS1]
            ])

            # Get nC_rS2 probs for subject s
            nC_rS2_temp = (1 - phi(cS2[s, :] - S2mu)) / C_area_rS2
            nC_rS2_probs = pt.concatenate([
                [((1 - phi(c1[s] - S2mu)) - (1 - phi(cS2[s, 0] - S2mu))) / C_area_rS2],
                nC_rS2_temp[:-1] - (1 - phi(cS2[s, 1:] - S2mu)) / C_area_rS2,
                [(1 - phi(cS2[s, nRatings - 2] - S2mu)) / C_area_rS2]
            ])

            # Avoid underflow of probabilities
            nC_rS1_probs = pt.switch(nC_rS1_probs < data["Tol"], data["Tol"], nC_rS1_probs)
            nI_rS2_probs = pt.switch(nI_rS2_probs < data["Tol"], data["Tol"], nI_rS2_probs)
            nI_rS1_probs = pt.switch(nI_rS1_probs < data["Tol"], data["Tol"], nI_rS1_probs)
            nC_rS2_probs = pt.switch(nC_rS2_probs < data["Tol"], data["Tol"], nC_rS2_probs)

            # Extract counts for this subject
            subject_counts = data["counts"][s, :]
            
            Multinomial(
                f"CR_counts_{s}",
                n=data["cr"][s],
                p=nC_rS1_probs,
                observed=subject_counts[:nRatings],
            )
            Multinomial(
                f"FA_counts_{s}",
                n=data["falsealarms"][s],
                p=nI_rS2_probs,
                observed=subject_counts[nRatings : nRatings * 2],
            )
            Multinomial(
                f"M_counts_{s}",
                n=data["m"][s],
                p=nI_rS1_probs,
                observed=subject_counts[nRatings * 2 : nRatings * 3],
            )
            Multinomial(
                f"H_counts_{s}",
                n=data["hits"][s],
                p=nC_rS2_probs,
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