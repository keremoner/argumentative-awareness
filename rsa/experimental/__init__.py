"""
Experimental variants of Speaker1.

These implement alternative informativeness and persuasiveness formulations
beyond the paper's default (obs informativeness + E[theta] persuasiveness).

The switching listeners that used to live here (``Listener1Switch``,
``SuspicionSwitchListener``, ...) were superseded by
``rsa.detection.DetectionListener`` and have been removed.
"""

from .speaker1_variants import (
    Speaker1_state_inf_def_pers,
    Speaker1_obs_inf_new_pers1,
    Speaker1_state_inf_new_pers1,
    Speaker1_state_inf_new_pers2,
    Speaker1_state_inf_new_pers4,
)
