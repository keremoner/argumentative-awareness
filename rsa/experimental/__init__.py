"""
Experimental variants of Speaker1 and Listener.

These implement alternative informativeness and persuasiveness formulations
beyond the paper's default (obs informativeness + E[theta] persuasiveness).
"""

from .speaker1_variants import (
    Speaker1_state_inf_def_pers,
    Speaker1_obs_inf_new_pers1,
    Speaker1_state_inf_new_pers1,
    Speaker1_state_inf_new_pers2,
    Speaker1_state_inf_new_pers4,
)
from .listener_switch import Listener1Switch, Listener2Switch
