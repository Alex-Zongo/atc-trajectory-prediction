"""The models the demo can run, first training run (seed 0) of each.

Each entry names a checkpoint in the models directory (runs/v2 locally, /app/models on the
Space) and says where the paper reports it. Every semantic model can be paired with every
trajectory model that reads a structured representation; the ADS-B-only and the
transcript-conditioned predictors read no representation and run as references.
"""

SBM = {
    "final": {"file": "sbm_jacktol_ho_s0.pt", "label": "Final SBM (contrast + handoff)",
                  "note": "The SBM of the main results: trained on the contrast data, fine-tuned on the handoff data.",
                  "where": "Sections VI.A to VI.F"},
    "contrast": {"file": "sbm_jacktol_cs_s0.pt", "label": "Contrast SBM",
                     "note": "Trained on the contrast data, before the handoff fine-tuning.",
                     "where": "Section VI.C"},
    "initial": {"file": "sbm_jacktol_anyorder_s0.pt", "label": "Initial SBM",
                    "note": "Trained on the initial data only, without callsign contrast. Used by the ablations.",
                    "where": "Section VI.E, Appendices I and J"},
    "independent": {"file": "sbm_jacktol_independent_s0.pt", "label": "Independent fields",
                        "note": "Ablation: the four fields are predicted independently.",
                        "where": "Table 6"},
    "mixture": {"file": "sbm_jacktol_mixture_s0.pt", "label": "Mixture over fields",
                    "note": "Ablation: a mixture over joint field values. No beam search; the most probable "
                         "representation is the most frequent of 48 draws.",
                    "where": "Table 6"},
    "no_history": {"file": "sbm_jacktol_nohist_s0.pt", "label": "Without aircraft history",
                       "note": "Ablation: the SBM reads the transcript and callsign only.",
                       "where": "Table 6"},
}

# kind: "ctp" (flow, also the category and primitive flows), "mdn" (mixture density)
CTP = {
    "flow_pca": {"file": "ctp_basis_atoms_zfix_s0.pt", "kind": "ctp", "label": "Flow on 18 PCA coefficients",
                     "note": "The CTP of the main results.", "where": "Sections VI.A to VI.F"},
    "flow_pca_earlier": {"file": "ctp_basis_atoms_s0.pt", "kind": "ctp",
                             "label": "Flow on PCA, earlier training labels",
                             "note": "Reference for the per-second velocity variant, which uses the same labels.",
                             "where": "Table 6 (2,000-segment group)"},
    "flow_velocities": {"file": "ctp_steps_atoms_s0.pt", "kind": "ctp", "label": "Flow on 180 per-second velocities",
                            "note": "Representation ablation, earlier training labels.", "where": "Table 6"},
    "flow_sine": {"file": "ctp_sine_atoms_s0.pt", "kind": "ctp", "label": "Flow on a sine basis",
                      "note": "Basis ablation: quarter-wave sines with the PCA's number of functions.",
                      "where": "Table 6, Appendix J"},
    "mixture_pca": {"file": "anchors_atoms_zfix_s0.pt", "kind": "mdn", "label": "Mixture density on PCA",
                        "note": "Eight Gaussian components offset from k-means anchors (MultiPath).",
                        "where": "Section VI.D, Appendix C.2"},
    "mixture_sine": {"file": "anchors_sine_atoms_s0.pt", "kind": "mdn", "label": "Mixture density on a sine basis",
                         "note": "Mixture-density head on the sine basis.", "where": "Appendix J"},
    "clusters_cond": {"file": "kindctp_joint_atoms_s0.pt", "kind": "ctp", "label": "Trajectory clusters, conditioned flow",
                          "note": "Eight trajectory clusters predicted first, then condition the PCA flow.",
                          "where": "Table 12"},
    "categories_cond": {"file": "kindctp_axis_atoms_s0.pt", "kind": "ctp", "label": "Separate motion categories, conditioned flow",
                            "note": "Track, altitude and speed categories predicted in sequence, then condition the flow.",
                            "where": "Table 12"},
    "clusters_aux": {"file": "kindctp_ro_atoms_s0.pt", "kind": "ctp", "label": "Trajectory clusters, auxiliary loss only",
                         "note": "Cluster classification as a training loss; the flow never sees the category.",
                         "where": "Table 12"},
    "prim_turn": {"file": "prim_atoms_s0.pt", "kind": "ctp", "label": "Primitives (turn rate), clusters",
                      "note": "Five-parameter rate profiles per axis plus six residual coefficients.",
                      "where": "Table 12"},
    "prim_turn_categories": {"file": "prim_axis_atoms_s0.pt", "kind": "ctp", "label": "Primitives (turn rate), separate categories",
                                 "note": "Primitive profiles conditioned on per-axis categories.", "where": "Table 12"},
    "prim_bank": {"file": "prim_v3_atoms_s0.pt", "kind": "ctp", "label": "Primitives (bank angle), clusters",
                      "note": "Bank-angle profile converted to turn rate with g tan(phi) / V.", "where": "Table 12"},
    "prim_bank_sampled": {"file": "prim_v3ss_atoms_s0.pt", "kind": "ctp", "label": "Primitives (bank angle), sampled categories in training",
                              "note": "Some training categories replaced by the classifier's own draws.", "where": "Table 12 (dagger)"},
    "prim_bank_aux": {"file": "prim_v3ro_atoms_s0.pt", "kind": "ctp", "label": "Primitives (bank angle), auxiliary loss only",
                          "note": "Category loss only; the flow never sees the category.", "where": "Table 12"},
}

REFERENCE = {
    "adsb_only": {"file": "ctp_basis_none_s0.pt", "label": "ADS-B only",
                      "note": "Same encoder and flow, no instruction tokens.", "where": "Table 3"},
    "transcript": {"file": "ctp_basis_text_s0.pt", "label": "Transcript-conditioned flow",
                       "note": "Reads the transcript tokens directly, no SBM. Initial-data vocabulary.",
                       "where": "Table 3"},
    "transcript_sbm_init": {"file": "ctp_basis_textinit_s0.pt", "label": "Transcript-conditioned, SBM-initialized",
                                "note": "Text encoder initialized from a trained SBM.", "where": "Section VI.A"},
}

SAMPLERS = {
    "euler12": {"method": "euler", "nfe": 12, "label": "12 Euler steps (default)"},
    "heun32": {"method": "heun", "nfe": 32, "label": "32 Heun steps (63 evaluations)"},
    "euler4": {"method": "euler", "nfe": 4, "label": "4 Euler steps (fast, coarse)"},
}

DEFAULTS = {"sbm": "final", "ctp": "flow_pca", "sampler": "euler12", "K": 12, "S": 6}


def files():
    """Every checkpoint the demo needs."""
    return sorted({v["file"] for t in (SBM, CTP, REFERENCE) for v in t.values()})


# The curated selection shown in the demo: the main models and the comparisons the main
# text discusses. The other entries stay available through the API.
SHOWN = {
    "sbm": ["final", "initial", "no_history"],
    "ctp": ["flow_pca", "mixture_pca", "flow_sine", "clusters_cond", "prim_bank"],
    "samplers": ["euler12", "heun32"],
}


def public(everything=False, available=None):
    """The selectable variants. available: the checkpoint files installed; variants whose
    checkpoint is missing are left out, so a clone with only some models still runs."""
    ok = (lambda e: True) if available is None else (lambda e: e["file"] in available)
    strip = lambda t, keys: {k: {kk: vv for kk, vv in t[k].items() if kk != "file"} for k in keys if ok(t[k])}
    pick = lambda name, t: list(t) if everything else SHOWN[name]
    sbm, ctp = strip(SBM, pick("sbm", SBM)), strip(CTP, pick("ctp", CTP))
    defaults = dict(DEFAULTS)
    defaults["sbm"] = defaults["sbm"] if defaults["sbm"] in sbm else next(iter(sbm), None)
    defaults["ctp"] = defaults["ctp"] if defaults["ctp"] in ctp else next(iter(ctp), None)
    return {"sbm": sbm, "ctp": ctp, "reference": strip(REFERENCE, list(REFERENCE)),
            "samplers": {k: SAMPLERS[k]["label"] for k in pick("samplers", SAMPLERS)},
            "defaults": defaults}
