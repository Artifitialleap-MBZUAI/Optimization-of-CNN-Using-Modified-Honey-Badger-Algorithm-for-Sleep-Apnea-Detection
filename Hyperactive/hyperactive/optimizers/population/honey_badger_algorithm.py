

import os
import random
import numpy as np
import pandas as pd 
from .particle_swarm_optimization import ParticleSwarmOptimizer
from ...base_positioner import BasePositioner
import math
from numpy import linalg as LA
import math
rng = np.random.default_rng()

class HoneyBadgerAlgorithm(ParticleSwarmOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_particles(self, _cand_):
        
        _p_list_ = [Particle() for _ in range(self._arg_.n_part)]
        for i, _p_ in enumerate(_p_list_):
            _p_.nr = i
            _p_.pos_current = _cand_._space_.get_random_pos()
            _p_.pos_best = _p_.pos_current
            _p_.velo = np.zeros(len(_cand_._space_.para_space))
            _p_.I=0
        return _p_list_
    def _intensity(self, _cand_, _p_list_, iter_count):
      epsilon = 0.00000000000000022204
      for _p_ in _p_list_:
      
          _p_.di=LA.norm([[ _p_.pos_current-_p_.pos_current+epsilon]])
          _p_.S= LA.norm([_p_.pos_current-[_p_.pos_current+1]+epsilon])
          _p_.di = np.power(_p_.di, 2)
          _p_.S= np.power(_p_.S, 2)
        
      
      for _p_ in _p_list_:
          _p_.n = random.random()
          _p_.I = _p_.n*_p_.S/[4*math.pi*_p_.di]
      return _p_.I    


    def _move_particles(self, _cand_, _p_list_, iter_count):
        
       
        alpha=self._arg_.h_c*math.exp(-iter_count/self._config_.n_iter)             # density factor in Eq. (3)
        self._intensity(_cand_, _p_list_, iter_count)           # intensity in Eq. (2)  
        Vs=random.random()


        #w = 0.7

        for _p_ in _p_list_:
            ph=0
            xpos_new = np.zeros([2, 20])
            Vs=random.random()
            F=self._arg_.h_vec_flag[math.floor((2*random.random()))]
            _p_.di=_p_.pos_best-_p_.pos_current
            if (Vs <0.5):  
              r3=np.random.random()
              r4=np.random.randn()
              r5=np.random.randn()
              xpos_new =_p_.pos_best+F*self._arg_.h_beta*_p_.pos_best+F*r3*alpha*_p_.di*np.abs(math.cos(2*math.pi*r4)*(1-math.cos(2*math.pi*r5)))
              _p_.velo =xpos_new
              ph=1

            else:
              r7=random.random()
        
              xpos_new =_p_.pos_best+F*r7*alpha*_p_.di
          
              _p_.velo =xpos_new
       

            _p_.pos_new = _p_.move_part(_cand_, _p_.pos_current,ph)
        

    def _eval_particles(self, _cand_, _p_list_, X, y,i,run):
      
         result = []
         for _p_ in _p_list_:
             _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)
             
             result.append(_p_.score_new)            
             if _p_.score_new > _cand_.score_best:
                 _cand_, _p_ = self._update_pos(_cand_, _p_)
             print(result)
             df =pd.DataFrame()
             
             #df = pd.DataFrame(result)
         df=pd.read_csv("output/""all_result_run"+str(run)+".csv")
         df["itr"+str(i+1)] = result
        # df["itr"+str(i+1)] = result
         #df = pd.DataFrame({"itr"+str(i+1):result})   
         #df.insert(len(df.columns), "itr"+str(i+1), result)
        
         
         #df = pd.DataFrame({"itr"+str(i+1):result})
         df.to_csv("output/""all_result_run"+str(run)+".csv", index= False)
    def _iterate(self, i, _cand_, _p_list_, X, y,run):
           
      
        self._move_particles(_cand_, _p_list_, i)
        self._eval_particles(_cand_, _p_list_, X, y,i,run) 
   

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        _p_list_ = self._init_particles(_cand_)

        return _p_list_


class Particle(BasePositioner):
    def __init__(self):
        self.nr = None
        self.velo = None
        
    def move_part(self, _cand_, pos,ph):
        if ph==1:
            pos_new = (pos + self.velo).astype(int)
        else:
            pos_new = (self.velo).astype(int)
        # limit movement
        n_zeros = [0] * len(_cand_._space_.dim)
        return np.clip(pos_new, n_zeros, _cand_._space_.dim)
