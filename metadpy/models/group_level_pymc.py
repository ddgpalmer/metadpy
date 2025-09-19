# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pytensor.tensor as pt
from pymc import Binomial, Deterministic, HalfNormal, Model, Multinomial, Normal, sample


def phi(x):
    """Cumulative normal distribution."""
    return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2))


def hmetad_groupLevel(
    data, sample_model=True, num_samples: int = 1000, num_chains: int = 4, **kwargs
):
    """Hierarchical Bayesian modeling of meta-d' (group level) with vectorized parameters.

    This is an internal function. The group level model must be called using
    :py:func:`metadpy.bayesian.hmetad`.

    Parameters
    ----------
    data : dict
        Response data containing data for multiple subjects.
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
    nRatings = data["nratings"]
    nSubj = data["nSubj"]
    
    with Model() as model:

        # Group-level hyperpriors
        mu_logMratio = Normal("mu_logMratio", mu=0.0, sigma=1.0)
        sigma_logMratio = HalfNormal("sigma_logMratio", sigma=1.0)
        mu_c2 = Normal("mu_c2", mu=0.0, sigma=2.0)
        sigma_c2 = HalfNormal("sigma_c2", sigma=2.0)

        # Subject-level parameters (vectorized)
        c1_vals = pt.as_tensor(data["c1"])
        d1_vals = pt.as_tensor(data["d1"])
        
        logMratio = Normal("logMratio", mu=mu_logMratio, sigma=sigma_logMratio, shape=nSubj)
        Mratio = Deterministic("Mratio", pt.exp(logMratio))
        meta_d = Deterministic("meta_d", Mratio * d1_vals)

        # Group-level parameters (means and standard deviations)
        # Group-level d1 statistics
        d1_group_mean = Deterministic("d1_group_mean", pt.mean(d1_vals))
        d1_group_std = Deterministic("d1_group_std", pt.std(d1_vals))
        
        # Group-level c1 statistics  
        c1_group_mean = Deterministic("c1_group_mean", pt.mean(c1_vals))
        c1_group_std = Deterministic("c1_group_std", pt.std(c1_vals))
        
        # Group-level meta_d statistics
        meta_d_group_mean = Deterministic("meta_d_group_mean", pt.mean(meta_d))
        meta_d_group_std = Deterministic("meta_d_group_std", pt.std(meta_d))

        # TYPE 1 SDT BINOMIAL MODEL (vectorized)
        h = phi(d1_vals / 2 - c1_vals) 
        f = phi(-d1_vals / 2 - c1_vals)
        
        H = Binomial("H", n=data["s"], p=h, observed=data["hits"], shape=nSubj)
        FA = Binomial("FA", n=data["n"], p=f, observed=data["falsealarms"], shape=nSubj)

        # Vectorized ordered priors on criteria bounded above and below by Type 1 c1
        cS1_hn = HalfNormal("cS1_hn", sigma=sigma_c2, shape=(nSubj, nRatings - 1))
        cS1 = Deterministic("cS1", pt.sort(-cS1_hn, axis=-1) + (c1_vals[:, None] - data["Tol"]))

        cS2_hn = HalfNormal("cS2_hn", sigma=sigma_c2, shape=(nSubj, nRatings - 1))
        cS2 = Deterministic("cS2", pt.sort(cS2_hn, axis=-1) + (c1_vals[:, None] + data["Tol"]))

        # Group-level c2 statistics (computed from all subjects' c2 criteria)
        # c2 represents the Type 2 confidence criteria (combination of cS1 and cS2)
        c2_combined = pt.concatenate([cS1, cS2], axis=1)  # Shape: (nSubj, 2*(nRatings-1))
        c2_group_mean = Deterministic("c2_group_mean", pt.mean(c2_combined, axis=0))
        c2_group_std = Deterministic("c2_group_std", pt.std(c2_combined, axis=0))

        # Means of SDT distributions (vectorized)
        S2mu = meta_d / 2
        S1mu = -meta_d / 2

        # Calculate normalisation constants (vectorized)
        C_area_rS1 = phi(c1_vals - S1mu)
        I_area_rS1 = phi(c1_vals - S2mu)
        C_area_rS2 = 1 - phi(c1_vals - S2mu)
        I_area_rS2 = 1 - phi(c1_vals - S1mu)
        
        last_idx = nRatings - 2

        # Get nC_rS1 probs (vectorized)
        nC_rS1_temp = phi(cS1 - S1mu[:, None]) / C_area_rS1[:, None]
        
        nC_rS1_first = phi(cS1[:, 0] - S1mu) / C_area_rS1
        nC_rS1_mid = nC_rS1_temp[:, 1:] - nC_rS1_temp[:, :-1]
        nC_rS1_last = (phi(c1_vals - S1mu) - phi(cS1[:, last_idx] - S1mu)) / C_area_rS1
        
        nC_rS1 = Deterministic(
            "nC_rS1",
            pt.concatenate([
                nC_rS1_first[:, None],
                nC_rS1_mid,
                nC_rS1_last[:, None]
            ], axis=1)
        )

        # Get nI_rS2 probs (vectorized)
        nI_rS2_temp = (1 - phi(cS2 - S1mu[:, None])) / I_area_rS2[:, None]
        
        nI_rS2_first = ((1 - phi(c1_vals - S1mu)) - (1 - phi(cS2[:, 0] - S1mu))) / I_area_rS2
        nI_rS2_mid = nI_rS2_temp[:, :-1] - (1 - phi(cS2[:, 1:] - S1mu[:, None])) / I_area_rS2[:, None]
        nI_rS2_last = (1 - phi(cS2[:, last_idx] - S1mu)) / I_area_rS2
        
        nI_rS2 = Deterministic(
            "nI_rS2",
            pt.concatenate([
                nI_rS2_first[:, None],
                nI_rS2_mid,
                nI_rS2_last[:, None]
            ], axis=1)
        )

        # Get nI_rS1 probs (vectorized)
        nI_rS1_temp = phi(cS1 - S2mu[:, None]) / I_area_rS1[:, None]
        
        nI_rS1_first = phi(cS1[:, 0] - S2mu) / I_area_rS1
        nI_rS1_mid = phi(cS1[:, 1:] - S2mu[:, None]) / I_area_rS1[:, None] - nI_rS1_temp[:, :-1]
        nI_rS1_last = (phi(c1_vals - S2mu) - phi(cS1[:, last_idx] - S2mu)) / I_area_rS1
        
        nI_rS1 = Deterministic(
            "nI_rS1",
            pt.concatenate([
                nI_rS1_first[:, None],
                nI_rS1_mid,
                nI_rS1_last[:, None]
            ], axis=1)
        )

        # Get nC_rS2 probs (vectorized)
        nC_rS2_temp = (1 - phi(cS2 - S2mu[:, None])) / C_area_rS2[:, None]
        
        nC_rS2_first = ((1 - phi(c1_vals - S2mu)) - (1 - phi(cS2[:, 0] - S2mu))) / C_area_rS2
        nC_rS2_mid = nC_rS2_temp[:, :-1] - ((1 - phi(cS2[:, 1:] - S2mu[:, None])) / C_area_rS2[:, None])
        nC_rS2_last = (1 - phi(cS2[:, last_idx] - S2mu)) / C_area_rS2
        
        nC_rS2 = Deterministic(
            "nC_rS2",
            pt.concatenate([
                nC_rS2_first[:, None],
                nC_rS2_mid,
                nC_rS2_last[:, None]
            ], axis=1)
        )

        # Avoid underflow of probabilities (vectorized)
        nC_rS1 = pt.switch(nC_rS1 < data["Tol"], data["Tol"], nC_rS1)
        nI_rS2 = pt.switch(nI_rS2 < data["Tol"], data["Tol"], nI_rS2)
        nI_rS1 = pt.switch(nI_rS1 < data["Tol"], data["Tol"], nI_rS1)
        nC_rS2 = pt.switch(nC_rS2 < data["Tol"], data["Tol"], nC_rS2)

        # TYPE 2 SDT MODEL (META-D) - Vectorized multinomial likelihood
        # Reshape data counts to match vectorized structure
        counts_reshaped = pt.as_tensor(data["counts"])  # Shape: (nSubj, 16)
        
        # Extract counts for each response type
        cr_counts = counts_reshaped[:, :nRatings]  # Shape: (nSubj, nRatings)
        fa_counts = counts_reshaped[:, nRatings:nRatings*2]  # Shape: (nSubj, nRatings)  
        m_counts = counts_reshaped[:, nRatings*2:nRatings*3]  # Shape: (nSubj, nRatings)
        h_counts = counts_reshaped[:, nRatings*3:nRatings*4]  # Shape: (nSubj, nRatings)
        
        # Create vectorized multinomial distributions
        CR_counts = Multinomial(
            "CR_counts",
            n=data["cr"],  # Shape: (nSubj,)
            p=nC_rS1,  # Shape: (nSubj, nRatings)
            observed=cr_counts,  # Shape: (nSubj, nRatings)
            shape=(nSubj, nRatings)
        )
        
        FA_counts = Multinomial(
            "FA_counts",
            n=data["falsealarms"],  # Shape: (nSubj,)
            p=nI_rS2,  # Shape: (nSubj, nRatings)
            observed=fa_counts,  # Shape: (nSubj, nRatings)
            shape=(nSubj, nRatings)
        )
        
        M_counts = Multinomial(
            "M_counts", 
            n=data["m"],  # Shape: (nSubj,)
            p=nI_rS1,  # Shape: (nSubj, nRatings)
            observed=m_counts,  # Shape: (nSubj, nRatings)
            shape=(nSubj, nRatings)
        )
        
        H_counts = Multinomial(
            "H_counts",
            n=data["hits"],  # Shape: (nSubj,)
            p=nC_rS2,  # Shape: (nSubj, nRatings)
            observed=h_counts,  # Shape: (nSubj, nRatings)
            shape=(nSubj, nRatings)
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