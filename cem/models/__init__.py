# -*- coding: utf-8 -*-
# @Author: Mateo Espinosa Zarlenga
# @Date:   2022-09-19 18:28:44
# @Last Modified by:   Mateo Espinosa Zarlenga
# @Last Modified time: 2022-09-19 18:28:44
from gymnasium.envs.registration import register

import pdb
pdb.set_trace()
register(id="cem/AFAEnv-v0",entry_point="cem.models.:AFAEnv")        