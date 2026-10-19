"""What the hadron-correlation analyses share (zjet/, dijet/): nothing here knows the trigger.

    runs       the analyses' data and figures: runs/<run>/{data,figures} (git-ignored)
    keys       angles and the portable random choices (event_key, key_index, key_uniform)
    weights    event weights sigma_k / N_k (x N_coll), effective events
    jetmap     jets with FastJet: anti-k_T, rho A with StripRho (FastJet imported by the caller)
    figdata    every figure's data as .npz and .root next to its PNG
    replot     a figure redrawn from its .npz

The C++ mirror is example/analysis_root/common (AnaCommon.h, JetMap.h).

Scripts put example/analysis on sys.path and import ``from jetcorr import ...``.
"""
