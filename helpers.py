#!/usr/bin/env python

import ROOT

import pyhf
import tempfile
import os
import shutil
import subprocess
from scipy.stats import norm
import math
import numpy as np

def write_data(data, output_root):
    source_data = data
    root_file = output_root

    binning = source_data['binning']
    bindata = source_data['bindata']

    f = ROOT.TFile(root_file, 'RECREATE')
    data = ROOT.TH1F('data', 'data', *binning)
    for i, v in enumerate(bindata['data']):
        data.SetBinContent(i + 1, v)
    data.Sumw2()

    bkg = ROOT.TH1F('bkg', 'bkg', *binning)
    for i, v in enumerate(bindata['bkg']):
        bkg.SetBinContent(i + 1, v)
    bkg.Sumw2()

    if 'bkgerr' in bindata:
        bkgerr = ROOT.TH1F('bkgerr', 'bkgerr', *binning)

        # shapesys must be as multiplicative factor
        for i, v in enumerate(bindata['bkgerr']):
            bkgerr.SetBinContent(i + 1, v / bkg.GetBinContent(i + 1))
        bkgerr.Sumw2()

    sig = ROOT.TH1F('sig', 'sig', *binning)
    for i, v in enumerate(bindata['sig']):
        sig.SetBinContent(i + 1, v)
    sig.Sumw2()
    f.Write()
    f.Close()

def get_p0(infile):

    infile = ROOT.TFile.Open(infile)
    workspace = infile.Get("combined")
    data = workspace.data("obsData")

    sbModel = workspace.obj("ModelConfig")
    poi = sbModel.GetParametersOfInterest().first()
    poi.setVal(1)
    sbModel.SetSnapshot(ROOT.RooArgSet(poi))

    bModel = sbModel.Clone()
    bModel.SetName("bonly")
    poi.setVal(0)
    bModel.SetSnapshot(ROOT.RooArgSet(poi))

    np = workspace.var("gamma_uncorrshape_bin_0")
    np.setRange(1e-10, 10.)

    ac = ROOT.RooStats.AsymptoticCalculator(data, sbModel, bModel)
    #ac.SetPrintLevel(10)
    #ac.SetPrintLevel(0)
    ac.SetOneSidedDiscovery(True)

    result = ac.GetHypoTest()
    pnull_obs = result.NullPValue()
    palt_obs = result.AlternatePValue()
    pnull_exp = []
    for sigma in [-2, -1, 0, 1, 2]:
        usecls = 0
        pnull_exp.append(ac.GetExpectedPValues(pnull_obs, palt_obs, sigma, usecls))

    infile.Close()

    return {'p0_obs': pnull_obs, 'p0_exp': pnull_exp}


def run_hf(data, template_dir="multibin_histfactory_p0"):
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(template_dir, os.path.join(tmpdir, "hf"))
        write_data(
            data,
            os.path.join(tmpdir, "hf/data/data.root")
        )
        subprocess.run(["hist2workspace", "config/example.xml"], cwd=os.path.join(tmpdir, "hf"))
        return get_p0(os.path.join(tmpdir, "hf/results/example_combined_GaussExample_model.root"))


def run_pyhf(data, use_minuit=False):
    bindata = data["bindata"]
    if use_minuit:
        pyhf.set_backend("numpy", custom_optimizer=pyhf.optimize.minuit_optimizer())
    else:
        pyhf.set_backend("numpy", custom_optimizer=pyhf.optimize.scipy_optimizer())
    model = pyhf.simplemodels.hepdata_like(
        signal_data=bindata["sig"],
        bkg_data=bindata["bkg"],
        bkg_uncerts=bindata["bkgerr"],
    )
    observations = bindata["data"]
    data = pyhf.tensorlib.astensor(observations + model.config.auxdata)
    test_poi = 0.
    obs, exp = pyhf.infer.hypotest(
        test_poi, data, model, return_expected_set=True, use_q0=True,
    )
    return {"p0_obs" : obs[0], "p0_exp" : list(exp.reshape(-1))}


def get_p0_obs_asymptotic_exact(data):
    # http://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf
    bd = data["bindata"]
    b = bd["bkg"][0]
    tau = b / (bd["bkgerr"][0] ** 2)
    m = tau * b
    n = bd["data"][0]
    if n <= b:
        return 0.5
    return norm.sf(
        np.sqrt(-2 * (n * np.log((n + m) / ((1 + tau) * n)) + m * np.log((tau * (n + m)) / ((1 + tau) * m))))
    )


def get_p0_exp_asymptotic_exact(data):
    """
    Calculate the expected p0 value (given an observation also,
    using the mles from a constrained fit (with mu=1) to data for the asimov data)
    based on the asymptotic approximation.
    See Eur.Phys.J.C71, `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_ ("CCGV paper") for the formulas.
    The derivation follows the same procedure as described for p0 in  <http://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf>
    """

    def ll(n, m, mu, s, b, tau):
        "Log likelihood without factorials (cancel in ratio)"
        return n*math.log(mu*s+b)-(mu*s+b)+m*math.log(tau*b)-tau*b

    def get_mles(mu, n, m, s, tau):
        # MLEs from CCGV paper
        muhat = (n-m/tau)/s
        bhat = m/tau
        bhathat = (n+m-(1+tau)*mu*s+math.sqrt((n+m-(1+tau)*mu*s)**2+4*(1+tau)*m*mu*s))/(2*(1+tau))
        return muhat, bhat, bhathat

    mu = 0

    bd = data["bindata"]

    n = bd["data"][0]
    b = bd["bkg"][0]
    s = bd["sig"][0]
    tau = b / (bd["bkgerr"][0] ** 2)
    m = tau * b

    ## Asimov dataset for mu=1 (expected discovery)
    # constrained fit to get parameter values for asimov dataset
    muhat_a, bhat_a, bhathat_a = get_mles(1, n, m, s, tau)
    n_a = bhathat_a + s
    m_a = tau*bhathat_a

    # fit the asimov datatset
    muhat, bhat, bhathat = get_mles(mu, n_a, m_a, s, tau)

    condll = ll(n_a, m_a, mu, s, bhathat, tau)
    uncondll = ll(n_a, m_a, muhat, s, bhat, tau)

    z = math.sqrt(-2.*(condll-uncondll))

    return [norm.sf(z + i) for i in [2, 1, 0, -1, -2]]


if __name__ == "__main__":

    data = {
        "binning": [1, -0.5, 1.5],
        "bindata": {"data": [80.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [25.0]},
    }
    bindata = data["bindata"]

    res_hf = run_hf(dict(data, binning=[1, 0.5, 1.5]))
    res_pyhf = run_pyhf(data)

    print(res_hf)
    print(res_pyhf)
