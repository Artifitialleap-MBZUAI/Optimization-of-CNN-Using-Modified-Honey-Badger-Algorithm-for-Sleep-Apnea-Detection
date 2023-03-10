# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:05:31 2022

@author: Ammar.Abasi
"""

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


class MHoneyBadgerAlgorithm(ParticleSwarmOptimizer):
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
  
    def _rank_individuals(self, _p_list_):
        scores_list = []
        for _p_ in _p_list_:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        return idx_sorted_ind
   
    def _move_particles(self, _cand_, _p_list_, iter_count, X, y):
        
        #lower=_cand_.getlower()
                #pos_ = int(pos[i])
                #values_dict[key] = list(self.para_space[key])[pos_]

            #return values_dict


        
        alpha=self._arg_.h_c*math.exp(-iter_count/self._config_.n_iter)             # density factor in Eq. (3)
        self._intensity(_cand_, _p_list_, iter_count)           # intensity in Eq. (2)  
        Vs=random.random()

        idx_sorted_ind = self._rank_individuals(_p_list_)
        sbest=_p_list_[idx_sorted_ind[1]].pos_current   
        tbest=_p_list_[idx_sorted_ind[1]].pos_current    

       
        for i, _p_ in enumerate(_p_list_):
           if i<self._arg_.n_part/2:
             
            _p_.lower=[0,0,0,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            _p_.upper=[0,1,0,101,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0]
            _p_.C_J=np.add(_p_.lower, _p_.upper)        
            _p_.C_J=np.multiply(_p_.C_J,0.5)
            r_c=np.random.random()           
                
            ph=0
            xpos_new = np.zeros([1, 25])
            Vs=random.random()
            F=self._arg_.h_vec_flag[math.floor((2*random.random()))]
            _p_.di=_p_.pos_best-_p_.pos_current
            _p_.score_current = _cand_.eval_pos(_p_.pos_current, X, y)
            if(r_c<0.5):
             ph=1
             xpos_new =  _p_.C_J+(_p_.C_J+_p_.pos_current)*r_c
            else:
             xpos_new =  _p_.C_J-(_p_.C_J+_p_.pos_current)*r_c
                
                    
            _p_.velo =xpos_new
       
            
            _p_.pos_new = _p_.move_part(_cand_, _p_.pos_current,ph)
           else:
                        
            ph=0
            xpos_new = np.zeros([2, 20])
            Vs=random.random()
            F=self._arg_.h_vec_flag[math.floor((2*random.random()))]
            _p_.di=_p_.pos_best-_p_.pos_current
            _p_.score_current = _cand_.eval_pos(_p_.pos_current, X, y)
            r_e=np.random.dirichlet(np.ones(3),size=1)
            for r in r_e:
                r_e1=r[0]
                r_e2=r[1]
                r_e3=r[2]                  
           
            if (Vs <0.5):  
              r3=np.random.random()
              r4=np.random.randn()
              r5=np.random.randn()
             
              
              xpos_new =(r_e1*_p_.pos_best+r_e2*sbest+r_e3*tbest)+F*self._arg_.h_beta* _p_.pos_current+F*r3*alpha*_p_.di*np.abs(math.cos(2*math.pi*r4)*(1-math.cos(2*math.pi*r5)))
              _p_.velo =xpos_new
              ph=1

            else:
              r7=np.random.random()
        
              xpos_new =(r_e1*_p_.pos_best+r_e2*sbest+r_e3*tbest)+F*r7*alpha*_p_.di
          
              _p_.velo =xpos_new
       
            
            _p_.pos_new = _p_.move_part(_cand_, _p_.pos_current,ph)
        for i, _p_ in enumerate(_p_list_):
           rm=np.random.random()
           rm1=np.random.random()
           rm2=np.random.random()
           mk=iter_count/self._config_.n_iter
           coe=(rm1+rm2)*mk
           m=.01
           if rm<m:
              print("mutation")
              ph=0
              xpos_new =(_p_.pos_best+coe)*(_p_.pos_best-_p_.pos_current)
          
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
             #df =pd.DataFrame()
             
             #df = pd.DataFrame(result)
         #df=pd.read_csv("output/""all_result_run"+str(run)+".csv")
         #df["itr"+str(i+1)] = result
        # df["itr"+str(i+1)] = result
         #df = pd.DataFrame({"itr"+str(i+1):result})   
         #df.insert(len(df.columns), "itr"+str(i+1), result)
        
         
         #df = pd.DataFrame({"itr"+str(i+1):result})
        # df.to_csv("output/""all_result_run"+str(run)+".csv", index= False)
    def _iterate(self, i, _cand_, _p_list_, X, y,run):
           
      
        self._move_particles(_cand_, _p_list_, i, X, y)
        self._eval_particles(_cand_, _p_list_, X, y,i,run) 
   

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        _p_list_ = self._init_particles(_cand_)
        
        for _p_ in _p_list_:
            _p_.score_current = _cand_.eval_pos(_p_.pos_current, X, y)
            _p_.score_best = _p_.score_current
            
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
