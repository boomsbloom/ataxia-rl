from __future__ import division
import glob, os, ast, warnings, random, sys, pickle, json, itertools
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def code_correct_choice(row):
  if row.image == 1:
    if row.pFReward1 >= 0.5:
      if row.f_chosen == 1:
        return 1
      else:
        return 0
    else:
      if row.f_chosen == 0:
        return 1
      else:
        return 0
  else: # this is image 2
    if row.pFReward1 >= 0.5:
      # better to choose J
      if row.f_chosen == 1:
        return 0
      else:
        return 1
    else:
      if row.f_chosen == 0:
        return 0
      else:
        return 1

def vkf_bin(outcomes,lambd,v0,omega):

  n_trials = len(outcomes)
  outcomes = np.array(outcomes)

  w0 = omega
  m = 0
  w = w0
  v = v0

  lr = np.zeros(n_trials)
  vol = np.zeros(n_trials)
  um = np.zeros(n_trials)

  for i in range(n_trials):

    lr[i] = np.sqrt(w+v)
    vol[i] = v
    um[i] = sigmoid(m)

    wpre = w

    y = outcomes[i]

    # Update
    delta = np.sqrt(w+v) * (y - sigmoid(m))
    m = m + delta
    k = (w + v) / (w + v + omega)
    w = (1 - k) * (w + v)
    v = v +lambd * (delta**2 + k*wpre - k*v)

  return lr, vol, um

def runRMIdealObserver(d,data_dir):

  def get_model_choice(row):
    if row.expected_f < 0.5:
      return 0
    elif row.expected_f > 0.5:
      return 1
    else: return np.random.choice([0,1])

  def recode_outcomes(row):
      if row.f_chosen == 1:
          return row.outcome
      elif row.f_chosen == 0:
          return 1 - row.outcome

  d["outcome_coded"] = d.apply(lambda row: recode_outcomes(row),axis=1) # outcomes coded like a one-armed bandit (needed for bayes models)

  sim_out = {"sub_factor":[],"lambda":[],
             "v0":[],"omega":[],"mse":[]}
  lambds = [0]#[0,0.1,0.2,0.3]#[0.01,0.1,0.5,1]
  omegas = [np.std(d.outcome)]#[0.01,0.1,0.5,1]
  v0s = [1.5]#np.linspace(1,2,20)#[]#[0.5,1,2]#[0.01,0.1,0.5,1]
  param_combos = list(itertools.product(*[lambds,omegas,v0s]))
  predictions = {"sub_factor":[],"expected_f":[],"image":[],"trial_number":[]}
  for p in param_combos:
    lambd = p[0]
    omega = p[1]
    v0 = p[2]
    for s in d.sub_factor.unique():
      d_in = d[(d.sub_factor == s)]
      lr1, vol1, um1 = vkf_bin(d_in[d_in.image==1].outcome_coded,lambd,v0,omega)
      lr2, vol2, um2 = vkf_bin(d_in[d_in.image==2].outcome_coded,lambd,v0,omega)

      i1se = np.sum((d_in[d_in.image==1].pFReward1-um1)**2)
      i2se = np.sum(((1-d_in[d_in.image==2].pFReward1)-um2)**2)
      mse = (i1se + i2se) / (len(um1)+len(um2))

      sim_out["sub_factor"].append(s)
      sim_out["lambda"].append(lambd)
      sim_out["v0"].append(v0)
      sim_out["omega"].append(omega)
      sim_out["mse"].append(mse)

      predictions["trial_number"].extend(np.concatenate([np.array(d_in[d_in.image==1].trial_number),np.array(d_in[d_in.image==2].trial_number)]))
      predictions["expected_f"].extend(np.concatenate([um1,um2]))
      predictions["image"].extend([i for j in [[1]*len(um1),[2]*len(um2)] for i in j])
      predictions["sub_factor"].extend([s] * len(np.concatenate([um1,um2])))

  sim_out = pd.DataFrame.from_dict(sim_out)
  predictions = pd.DataFrame.from_dict(predictions)
  predictions = predictions.groupby('sub_factor').apply(lambda x: x.sort_values(by='trial_number')).reset_index(drop=True)
  predictions["f_chosen"] = predictions.apply(lambda row: get_model_choice(row), axis=1)
  predictions["outcome"] = d.outcome
  predictions["pFReward1"] = d.pFReward1
  predictions["accuracy"] = predictions.apply(lambda row: code_correct_choice(row), axis=1)
  predictions.to_csv(os.path.join(data_dir,"idealMapData4Stan.csv"))

  return predictions


def load_mturk_data(data_dir=None):
    '''
    Load the raw task data from Mturk
    '''
    pid_files = glob.glob(os.path.join(data_dir,"*_experiment_data.csv"))
    full_data = []
    for file in pid_files:
        f = pd.read_csv(file)
        f["subject_id"] = file.split('/')[-1].split('_')[0]
        full_data.append(f)
    full_data = pd.concat(full_data)
    n_subjects = len(full_data.subject_id.unique())

    return n_subjects, full_data

def clean_hybrid_data(data=None,phase="choice"):

    global t_counter
    global switch_counter
    switch_counter = 0
    def get_t_since_reversal(row):
        global t_counter
        if row.switch_trial or row.trial_number == 1: t_counter = 0
        else: t_counter+=1
        return t_counter

    def get_switch_number(row):
        global switch_counter
        if row.trial_number == 1: switch_counter = 0
        if row.switch_trial: switch_counter+=1
        return switch_counter

    data = data[data.phase == phase]
    data['t_since_reversal'] = data.apply(lambda row: get_t_since_reversal(row),axis=1)
    data['switch_number'] = data.apply(lambda row: get_switch_number(row),axis=1)
    data["old_value"] = data["old_value"].astype("float")
    data = data[data.choice != "no_response"]
    data["value"] = data["value"].astype("float")
    data["lucky_chosen"] = data["lucky_chosen"].astype("float")

    return data

def clean_hybrid_pat_data(data=None):

    global t_counter
    global switch_counter
    switch_counter = 0
    def get_t_since_reversal(row):
        global t_counter
        if row.switch_trial or row.trial_number == 1: t_counter = 0
        else: t_counter+=1
        return t_counter

    def get_switch_number(row):
        global switch_counter
        if row.trial_number == 1: switch_counter = 0
        if row.switch_trial: switch_counter+=1
        return switch_counter

    data['t_since_reversal'] = data.apply(lambda row: get_t_since_reversal(row),axis=1)
    data['switch_number'] = data.apply(lambda row: get_switch_number(row),axis=1)
    data["old_value"] = data["old_value"].astype("float")
    data = data[~data.red_chosen.isna()]
    data["value"] = data["value"].astype("float")
    data["lucky_chosen"] = data["lucky_chosen"].astype("float")

    return data

def get_old_data(data=None):

    old_data = data[data.old_trial == True]
    old_data["old_chosen"] = old_data["old_chosen"].astype("float")
    old_data["trial_number"] = old_data["trial_number"].astype("float")

    encoded_t_since_reversals, encoded_switch_numbers = [], []

    for pid in old_data.subject_id.unique():
        pid_data = data[data.subject_id == pid]
        pid_old_data = old_data[old_data.subject_id == pid]
        for i_row, row in pid_old_data.iterrows():
                old_trial_n = row.old_trial_number
                encoded_t_since_reversal = pid_data[pid_data.trial_number == old_trial_n].t_since_reversal.iloc[0]
                encoded_t_since_reversals.append(encoded_t_since_reversal)
                encoded_switch_number = pid_data[pid_data.trial_number == old_trial_n].switch_number.iloc[0]
                encoded_switch_numbers.append(encoded_switch_number)

    old_data['encoded_t_since_reversal'] = encoded_t_since_reversals
    old_data['encoded_switch_number'] = encoded_switch_numbers
    old_data['within_switch'] = np.where(old_data['encoded_switch_number'] == old_data['switch_number'], 1, 0)

    return old_data

def recode_old_deck(row):
    #effect of familiarity for previously seen objects
    #.5=red old;0=both new;-.5=blue old
    if row.old_trial:
        if row.old_value == row.value:
            # Old was chosen
            if row.red_chosen:
                return 0.5
            else:
                return -0.5
        else:
            # Old was not chosen
            if row.red_chosen:
                return -0.5
            else:
                return 0.5
    else:
        return 0

def recode_old_value(row):
    if row.old_trial:
        if row.old_deck == 0.5:
            return (row.old_value-0.5)
        else:
            return (0.5-row.old_value)
    else:
        return 0

def recode_outcomes(row):
    # code outcomes as coming from a one-armed bandit (needed for some models)
    if row.red_chosen == 0: return 0.5 - row.outcome
    elif row.red_chosen == 1: return row.outcome - 0.5

def set_enc_vals(row,meanenct):
    # for mean centering trials since reversal at encoding
    if row.old_deck == 0: return meanenct
    else: return row.encoded_t_since_reversal

def get_reversal_df(n_around,data,ys):

    reversal_data = {"sub_factor":[],"subject_id":[],"time_point":[],"trial_number":[],"ctrl_patient_id":[]}
    for y in ys: reversal_data[y]=[]
    for subj in data.sub_factor.unique():
        s = data[data.sub_factor == subj].reset_index()
        change_indices = s[s["t_since_reversal"]==0].index
        for idx in change_indices:
            if len(s) - idx < n_around:
              surr_data = s.iloc[np.arange(idx-(n_around-1),idx+(len(s) - idx))]
              trial_labels = np.arange(-n_around+1,len(s) - idx)
            elif idx != 0:
                surr_data = s.iloc[np.arange(idx-(n_around-1),idx+n_around)]
                trial_labels = np.arange(-n_around+1,n_around)
            else:
                surr_data = s.iloc[np.arange(0,idx+n_around)]
                trial_labels = np.arange(0,n_around)
            reversal_data["time_point"].extend(trial_labels)
            reversal_data["trial_number"].extend(surr_data.trial_number)
            reversal_data["ctrl_patient_id"].extend(surr_data.ctrl_patient_id)
            reversal_data["subject_id"].extend(surr_data.subject_id)
            reversal_data["sub_factor"].extend([subj] * len(trial_labels))
            for y in ys: reversal_data[y].extend(surr_data[y])
    reversal_data = pd.DataFrame.from_dict(reversal_data)

    return reversal_data

def code_as_stay(row,imgData):
   t=row.img_trial_number
   if t == 1:
       return 0
   else:
       prev_row = imgData[imgData.img_trial_number == t-1]
       if prev_row.choice.iloc[0] == "f":
           if prev_row.outcome.iloc[0] == 1:
               return 0.5
           else:
               return -0.5
       elif prev_row.choice.iloc[0] == "j":
           if prev_row.outcome.iloc[0] == 1:
               return -0.5
           else:
               return 0.5
       else:
           return np.nan
