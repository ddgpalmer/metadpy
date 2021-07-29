# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np
import pymc3 as pm

from metadPy import load_dataset
from metadPy.bayesian import extractParameters, hmetad


class Testsdt(TestCase):
    def test_preprocess(self):
        """Test preprocess function"""
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])
        data = extractParameters(nR_S1, nR_S2)
        assert round(data["d1"], 3) == 1.535
        assert round(data["c1"]) == 0
        assert np.all(
            data["counts"]
            == np.array([52, 32, 35, 37, 26, 12, 4, 2, 2, 5, 15, 22, 33, 38, 40, 45])
        )
        assert data["nratings"] == 4
        assert data["Tol"] == 1e-05
        assert data["FA"] == 44
        assert data["CR"] == 156
        assert data["M"] == 44
        assert data["H"] == 156
        assert data["N"] == 200
        assert data["S"] == 200

    def test_hmetad(self):
        """Test hmetad function"""
        group_df = load_dataset("rm")

        ####################
        # Test subject level
        ####################
        this_df = group_df[(group_df.Subject == 0) & (group_df.Condition == 0)]
        model = hmetad(
            data=this_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            sample_model=False,
        )
        assert isinstance(model, pm.Model)

        ####################
        # Test group level
        ####################
        this_df = group_df[group_df.Condition == 0]
        model = hmetad(
            data=this_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            subject="Subject",
            sample_model=False,
        )
        assert isinstance(model, pm.Model)

        #######################
        # Test repeated measure
        #######################
        model = hmetad(
            data=group_df,
            nRatings=4,
            stimuli="Stimuli",
            accuracy="Accuracy",
            confidence="Confidence",
            subject="Subject",
            within="Condition",
            sample_model=False,
        )
        assert isinstance(model, pm.Model)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
