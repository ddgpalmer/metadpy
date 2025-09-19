# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import pytensor.tensor as pt
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

        # Subject-level parameters
        # Use empirical values for Type 1 parameters to reduce complexity initially
        c1_vals = data["c1"]  # Use empirical Type 1 criterion
        d1_vals = data["d1"]  # Use empirical Type 1 d'
        
        # Store empirical d1 and c1 values as Deterministic variables for availability in idata
        d1 = Deterministic("d1", pt.as_tensor_variable(d1_vals))
        c1 = Deterministic("c1", pt.as_tensor_variable(c1_vals))
        
        # Hierarchical log M-ratio
        logMratio = Normal("logMratio", mu=mu_logMratio, sigma=sigma_logMratio, shape=nSubj)
        Mratio = Deterministic("Mratio", pt.exp(logMratio))
        
        # Compute meta_d for each subject and store as Deterministic variable
        meta_d = Deterministic("meta_d", Mratio * d1)
        
        # Build the model for each subject using the same structure as the single-subject model
        for s in range(nSubj):
            c1_s = c1[s]
            d1_s = d1[s]
            meta_d_s = meta_d[s]

            # TYPE 1 SDT BINOMIAL MODEL
            h_s = phi(d1_s / 2 - c1_s)
            f_s = phi(-d1_s / 2 - c1_s)
            H_s = Binomial(f"H_{s}", data["s"][s], h_s, observed=data["hits"][s])
            FA_s = Binomial(f"FA_{s}", data["n"][s], f_s, observed=data["falsealarms"][s])

            # Specify ordered prior on criteria bounded above and below by Type 1 c1
            cS1_hn_s = HalfNormal(f"cS1_hn_{s}", sigma=sigma_c2, shape=nRatings - 1)
            cS1_s = Deterministic(f"cS1_{s}", pt.sort(-cS1_hn_s) + (c1_s - data["Tol"]))

            cS2_hn_s = HalfNormal(f"cS2_hn_{s}", sigma=sigma_c2, shape=nRatings - 1)
            cS2_s = Deterministic(f"cS2_{s}", pt.sort(cS2_hn_s) + (c1_s + data["Tol"]))
            
            # Store c2 criteria as Deterministic variables for availability in idata
            # c2 represents the Type 2 confidence criteria
            c2_s = Deterministic(f"c2_{s}", pt.concatenate([cS1_s, cS2_s], axis=0))

            # Means of SDT distributions
            S2mu_s = pt.flatten(meta_d_s / 2, 1)
            S1mu_s = pt.flatten(-meta_d_s / 2, 1)

            # Calculate normalisation constants
            C_area_rS1_s = phi(c1_s - S1mu_s)
            I_area_rS1_s = phi(c1_s - S2mu_s)
            C_area_rS2_s = 1 - phi(c1_s - S2mu_s)
            I_area_rS2_s = 1 - phi(c1_s - S1mu_s)

            # Get nC_rS1 probs
            nC_rS1_temp = phi(cS1_s - S1mu_s) / C_area_rS1_s
            nC_rS1_s = Deterministic(
                f"nC_rS1_{s}",
                pt.concatenate(
                    [
                        pt.reshape(phi(cS1_s[0] - S1mu_s) / C_area_rS1_s, (1,)),
                        nC_rS1_temp[1:] - nC_rS1_temp[:-1],
                        pt.reshape((phi(c1_s - S1mu_s) - phi(cS1_s[(nRatings - 2)] - S1mu_s)) / C_area_rS1_s, (1,)),
                    ],
                    axis=0,
                ),
            )

            # Get nI_rS2 probs
            nI_rS2_temp = (1 - phi(cS2_s - S1mu_s)) / I_area_rS2_s
            nI_rS2_s = Deterministic(
                f"nI_rS2_{s}",
                pt.concatenate(
                    [
                        pt.reshape(((1 - phi(c1_s - S1mu_s)) - (1 - phi(cS2_s[0] - S1mu_s))) / I_area_rS2_s, (1,)),
                        nI_rS2_temp[:-1] - (1 - phi(cS2_s[1:] - S1mu_s)) / I_area_rS2_s,
                        pt.reshape((1 - phi(cS2_s[nRatings - 2] - S1mu_s)) / I_area_rS2_s, (1,)),
                    ],
                    axis=0,
                ),
            )

            # Get nI_rS1 probs
            nI_rS1_temp = phi(cS1_s - S2mu_s) / I_area_rS1_s
            nI_rS1_s = Deterministic(
                f"nI_rS1_{s}",
                pt.concatenate(
                    [
                        pt.reshape(phi(cS1_s[0] - S2mu_s) / I_area_rS1_s, (1,)),
                        phi(cS1_s[1:] - S2mu_s) / I_area_rS1_s - nI_rS1_temp[:-1],
                        pt.reshape((phi(c1_s - S2mu_s) - phi(cS1_s[(nRatings - 2)] - S2mu_s)) / I_area_rS1_s, (1,)),
                    ],
                    axis=0,
                ),
            )

            # Get nC_rS2 probs
            nC_rS2_temp = (1 - phi(cS2_s - S2mu_s)) / C_area_rS2_s
            nC_rS2_s = Deterministic(
                f"nC_rS2_{s}",
                pt.concatenate(
                    [
                        pt.reshape(((1 - phi(c1_s - S2mu_s)) - (1 - phi(cS2_s[0] - S2mu_s))) / C_area_rS2_s, (1,)),
                        nC_rS2_temp[:-1] - ((1 - phi(cS2_s[1:] - S2mu_s)) / C_area_rS2_s),
                        pt.reshape((1 - phi(cS2_s[nRatings - 2] - S2mu_s)) / C_area_rS2_s, (1,)),
                    ],
                    axis=0,
                ),
            )

            # Avoid underflow of probabilities
            nC_rS1_s = pt.switch(nC_rS1_s < data["Tol"], data["Tol"], nC_rS1_s)
            nI_rS2_s = pt.switch(nI_rS2_s < data["Tol"], data["Tol"], nI_rS2_s)
            nI_rS1_s = pt.switch(nI_rS1_s < data["Tol"], data["Tol"], nI_rS1_s)
            nC_rS2_s = pt.switch(nC_rS2_s < data["Tol"], data["Tol"], nC_rS2_s)

            # TYPE 2 SDT MODEL (META-D)
            # Multinomial likelihood for response counts ordered as c(nR_S1,nR_S2)
            Multinomial(
                f"CR_counts_{s}",
                n=data["cr"][s],
                p=nC_rS1_s,
                shape=nRatings,
                observed=data["counts"][s, :nRatings],
            )
            Multinomial(
                f"FA_counts_{s}",
                n=data["falsealarms"][s],
                p=nI_rS2_s,
                shape=nRatings,
                observed=data["counts"][s, nRatings : nRatings * 2],
            )
            Multinomial(
                f"M_counts_{s}",
                n=data["m"][s],
                p=nI_rS1_s,
                shape=nRatings,
                observed=data["counts"][s, nRatings * 2 : nRatings * 3],
            )
            Multinomial(
                f"H_counts_{s}",
                n=data["hits"][s],
                p=nC_rS2_s,
                shape=nRatings,
                observed=data["counts"][s, nRatings * 3 : nRatings * 4],
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