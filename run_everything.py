#!/usr/bin/env python

import ROOT

import json
import sys
import pyhf
import tempfile
import os
import shutil
import subprocess

def write_data(data, output_root):
    # source_data = json.load(open(sys.argv[1]))
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

    ac = ROOT.RooStats.AsymptoticCalculator(data, sbModel, bModel)
    #ac.SetPrintLevel(10)
    ac.SetPrintLevel(0)
    ac.SetOneSidedDiscovery(True)

    result = ac.GetHypoTest()
    pnull_obs = result.NullPValue()
    palt_obs = result.AlternatePValue()
    pnull_exp = []
    for sigma in [-2, -1, 0, 1, 2]:
        usecls = 0
        pnull_exp.append(ac.GetExpectedPValues(pnull_obs, palt_obs, sigma, usecls))

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


def run_pyhf(data):
    bindata = data["bindata"]
    pyhf.set_backend("numpy")
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
