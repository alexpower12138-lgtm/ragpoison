"""Detection signals (WP3, WP4).

Planned modules, one signal per file, each a pure function of
``(query, candidates, context)``:
    base.py                 SignalResult and the signal registry
    s1_sim_overfit.py       S1 abnormally high similarity to the query
    s2_cross_consistency.py S2 disagreement with the rest of the candidate pool
    s3_freq_density.py      S3 answer centrality (frequency density)
    s3b_majority.py         S3b naive majority-vote conflict (negative control)
    s4_instruction.py       S4 instruction / template traces
    s5_multiview.py         S5 four-view judgement
    s6_pseudo_ppl.py        S6 character 3-gram pseudo-perplexity
    s7_prior.py             S7 retrieval prior (fusion weight only)
"""

from __future__ import annotations
