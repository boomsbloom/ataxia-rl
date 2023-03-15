from __future__ import division
import glob, sys, os, ast, warnings, json
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import importlib
from support_functions import *

def process(task,data_dir,group):
    if task == 1:
        grp_data_dir = os.path.join(data_dir,'rm',group,'*_experiment_data.csv')

        # Load and process the response mapping data

        # Mapping between control and patient IDs
        ctrl_pat_map = pd.read_csv(os.path.join(data_dir,"ctrl_pat_map.csv"))
        ctrl_pat_map["patient"] = ctrl_pat_map["patient"].astype("str").str.zfill(3)

        data = []
        for i, f in enumerate(sorted(glob.glob(grp_data_dir))):
           d = pd.read_csv(f)
           d['subject_id'] = f.split('/')[-1].split('_')[0]
           d = cleanRMData(d)
           data.append(d)
        data = pd.concat(data).reset_index(drop=True)

        def recode_outcomes(row):
            if row.f_chosen == 1:
                return row.outcome - 0.5
            elif row.f_chosen == 0:
                return 0.5 - row.outcome

        data = data[["subject_id","outcome","f1_chosen","trial_number","rt","image","f_chosen","pFReward1","stay"]].dropna()
        data["sub_factor"] = pd.factorize(data.subject_id)[0].astype("int")+1
        data["outcome_coded"] = data.apply(lambda row: recode_outcomes(row),axis=1) # outcomes coded like a one-armed bandit (needed for bayes models)

        if group == "control":

            ctrl_patient_ids = []
            for i, row in data.iterrows():
              for j, row2 in ctrl_pat_map.iterrows():
                if row.subject_id == row2.control:
                  ctrl_patient_ids.append(row2.patient)
            data["ctrl_patient_id"] = ctrl_patient_ids
            data["group"] = 2

            # Load and process the CCAS data for these controls
            #ccas_data = processCCAS(data_dir,data)

        elif group == "patient":

            data["group"] = 1
            data["ctrl_patient_id"] = data["subject_id"]

        # for getting the proportion of no responses
        #data.groupby(["subject_id"])["f_chosen"].count()/200#["f_chosen"].isna()

        dfile = os.path.join(data_dir,'%sMapData4Stan.csv'%group)
        data.to_csv(dfile,index=False)

    elif task == 2:

        def recode_outcomes(row):
            if row.red_chosen == 1:
                return row.outcome - 0.5
            elif row.red_chosen == 0:
                return 0.5 - row.outcome

        if group == "control":
            # Load the raw behavioral data
            n_subs, orig_data = load_mturk_data(data_dir=os.path.join(data_dir,"hybrid/control"))
            orig_data = orig_data[orig_data.phase == "choice"]
            data = clean_hybrid_data(data=orig_data)
            old_data = get_old_data(data=data)

            '''
            Consolidate the data into a single csv and save
            '''
            # get factorized subject ID
            data["sub_factor"] = pd.factorize(data.subject_id)[0]+1
            # get the # of trials since reversal at encoding time
            encoded_t_since = []
            for i, row in data.iterrows():
                curr_sub = row.subject_id
                curr_trial = row.trial_number
                curr_old = old_data[(old_data.subject_id == curr_sub) & (old_data.trial_number == curr_trial)]
                if curr_old.empty:
                    encoded_t_since.append(0)
                else:
                    encoded_t_since.append(int(curr_old.encoded_t_since_reversal))
            data["encoded_t_since_reversal"] = encoded_t_since
            data["outcome"] = data["value"] # rename value to outcome for clarity
            data["red_chosen"] = data["choice"].replace({"blue":0,"red":1}) # whether red was chosen or not
            data["old_deck"] = data.apply(lambda row: recode_old_deck(row),axis=1) # which deck is old
            data["old_value_4_HBI"] = data.apply(lambda row: recode_old_value(row),axis=1) # old value coded in terms of the red deck
            data["old_value"] = data["old_value"].astype('float')
            data["outcome_coded"] = data.apply(lambda row: recode_outcomes(row),axis=1) # outcomes coded like a one-armed bandit (needed for bayes models)
            data["encoding_trial"] = data["old_trial_number"].replace({np.nan:0}) # replace nan and rename to be more intuitive
            data["old_chosen"] = data["old_chosen"].replace({False:0,True:1}) # convert to numerical
            data["red_chosen"] = data["red_chosen"]+1 # replace 0 and 1 with 1 and 2. This is mostly leftover from some old code..should eventaully change.
            # subset to variables of interest (or possible interest) for stan
            v = ["outcome","red_chosen","old_deck","subject_id","old_value_4_HBI","old_value","outcome_coded","sub_factor","red_luck","red_object","blue_object",
                 "encoding_trial","encoded_t_since_reversal","t_since_reversal","trial_number","rt","lucky_chosen"]
            sd = data[v][data[v].outcome.notna()].reset_index(drop=True)
            # mean center trials since reversal variables
            mean_enc_t = np.mean(sd[sd["old_deck"] != 0]["encoded_t_since_reversal"])
            sd['enc_t_since_centered'] = sd.apply(lambda row: set_enc_vals(row,mean_enc_t),axis=1)
            sd['enc_t_since_centered'] = sd['enc_t_since_centered'] - mean_enc_t
            sd["t_since_centered"] = sd["t_since_reversal"] - np.mean(sd["t_since_reversal"])
            # get the observation on which each object was encoded across the entire dataset (needed in stan)
            curr_sub = 1000
            encoding_trials = []
            for i, row in sd.iterrows():
                if curr_sub != row.sub_factor:
                    curr_sub = row.sub_factor
                enc_trial = row.encoding_trial
                obs_n = sd.index[(sd.sub_factor == curr_sub) & (sd.trial_number == enc_trial)]
                if len(obs_n) == 0:
                    encoding_trials.append(0)
                else:
                    encoding_trials.append(obs_n[0]+1)
            sd["encoding_obs"] = encoding_trials
            # Make a variable which says whether previous trial was old or not
            sd["prev_old"] = sd.groupby(["sub_factor"]).old_deck.shift().replace({0.5:1,-0.5:1})
            # Make a variable which says whether old value is above orbelow 50 cents
            sd["should_choose_old"] = np.where(sd.old_value > 0.5, 1, 0) + np.where(sd.old_value.notnull() , 0, np.nan)
            # Make a variable saying whether they chose appropriately (chose old if above 50, did not choose old is below 50)
            sd["old_chosen"] = ((sd["old_deck"] == 0.5) & (sd["red_chosen"] == 2) | (sd["old_deck"] == -0.5) & (sd["red_chosen"] == 1)).replace({False:0.0,True:1.0})
            sd["correct_old_choice"] = (sd["old_chosen"] == sd["should_choose_old"]).replace({False:0.0,True:1.0})
            sd["lucky_deck"] = sd["red_luck"].replace({"lucky":0.5,"unlucky":-0.5})
            sd["old_on_lucky"] = sd["old_deck"] == sd["lucky_deck"]
            # incongruent: old on lucky = 1 and should choose old = 0
            # incongruent: old on lucky = 0 and should choose old = 1
            sd["incongruent"] = np.where((sd.should_choose_old == 1) & (sd.old_on_lucky != sd.should_choose_old), 1, 0) + np.where((sd.should_choose_old == 0) & (sd.old_on_lucky != sd.should_choose_old), 1, 0)

            # Mapping between control and patient IDs
            ctrl_pat_map = pd.read_csv(os.path.join(data_dir,"ctrl_pat_map.csv"))
            ctrl_pat_map["patient"] = ctrl_pat_map["patient"].astype("str").str.zfill(3)
            ctrl_patient_ids = []
            for i, row in data.iterrows():
              for j, row2 in ctrl_pat_map.iterrows():
                if row.subject_id == row2.control:
                  ctrl_patient_ids.append(row2.patient)
            sd["ctrl_patient_id"] = ctrl_patient_ids

            sd.to_csv(os.path.join(data_dir,"controlHybridData.csv"),index=False)
            data = sd

        elif group == "patient":

            data = pd.read_csv(os.path.join(data_dir,"patientHybridData.csv"))
            data = data.rename({"SubNum":"subject_id","ChosenValue":"value",
                                "RedChosen":"red_chosen","Trial":"trial_number",
                                "RedLuck":"red_luck","OldTrial":"old_trial","RT":"rt",
                                "OldChosen":"old_chosen","OldValue":"old_value"},axis=1)
            data["lucky_chosen"] = data["red_chosen"] == data["red_luck"]
            switch_trials = []
            for s in data.subject_id.unique():
              sd = data[data.subject_id == s]
              switch_trial = sd.red_luck.diff()
              switch_trial = switch_trial.replace({np.nan:1,-1:1})
              switch_trials.append(switch_trial)
            data["switch_trial"] = pd.concat(switch_trials)

            data = clean_hybrid_pat_data(data)
            data["sub_factor"] = pd.factorize(data.subject_id)[0]+1
            data["outcome"] = data["value"] # rename value to outcome for clarity
            data["old_deck"] = data.apply(lambda row: recode_old_deck(row),axis=1) # which deck is old
            data["old_value_4_HBI"] = data.apply(lambda row: recode_old_value(row),axis=1) # old value coded in terms of the red deck
            data["old_value"] = data["old_value"].astype('float')
            data["outcome_coded"] = data.apply(lambda row: recode_outcomes(row),axis=1) # outcomes coded like a one-armed bandit (needed for bayes models)
            data["old_chosen"] = data["old_chosen"].replace({False:0,True:1}) # convert to numerical
            data["red_chosen"] = data["red_chosen"]+1 # replace 0 and 1 with 1 and 2. This is mostly leftover from some old code..should eventaully change.
            # subset to variables of interest (or possible interest) for stan
            v = ["outcome","red_chosen","old_deck","subject_id","old_value_4_HBI","old_value","outcome_coded",
                 "sub_factor","red_luck","t_since_reversal","trial_number","rt","lucky_chosen"]
            sd = data[v][data[v].outcome.notna()].reset_index(drop=True)
            # mean center trials since reversal variables
            sd["t_since_centered"] = sd["t_since_reversal"] - np.mean(sd["t_since_reversal"])
            # Make a variable which says whether previous trial was old or not
            sd["prev_old"] = sd.groupby(["sub_factor"]).old_deck.shift().replace({0.5:1,-0.5:1})
            # Make a variable which says whether old value is above orbelow 50 cents
            sd["should_choose_old"] = np.where(sd.old_value > 0.5, 1, 0) + np.where(sd.old_value.notnull() , 0, np.nan)
            # Make a variable saying whether they chose appropriately (chose old if above 50, did not choose old is below 50)
            sd["old_chosen"] = ((sd["old_deck"] == 0.5) & (sd["red_chosen"] == 2) | (sd["old_deck"] == -0.5) & (sd["red_chosen"] == 1)).replace({False:0.0,True:1.0})
            sd["correct_old_choice"] = (sd["old_chosen"] == sd["should_choose_old"]).replace({False:0.0,True:1.0})
            sd["lucky_deck"] = sd["red_luck"].replace({"lucky":0.5,"unlucky":-0.5})
            sd["old_on_lucky"] = sd["old_deck"] == sd["lucky_deck"]
            # incongruent: old on lucky = 1 and should choose old = 0
            # incongruent: old on lucky = 0 and should choose old = 1
            sd["incongruent"] = np.where((sd.should_choose_old == 1) & (sd.old_on_lucky != sd.should_choose_old), 1, 0) + np.where((sd.should_choose_old == 0) & (sd.old_on_lucky != sd.should_choose_old), 1, 0)
            sd["ctrl_patient_id"] = sd.subject_id.astype("str").str.zfill(3)

            data = sd

    return data

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

def recode_choice(row):
   if row.choice == "f":
       if row.image == 1:
           return 1
       else:
           return 0
   elif row.choice == "j":
       if row.image == 1:
           return 0
       else:
           return 1
   else:
       return np.nan

def get_correct_choice(row):
   if row.image == 1:
       if row.pFReward1 >= 0.5:
           return 1
       else:
           return 0
   elif row.image == 2:
       if row.pFReward1 < 0.5:
           return 1
       else:
           return 0

def cleanRMData(data):
   data = data[data.phase == "choice"]
   try: data["outcome"] = data["outcome"].replace({"too_slow":np.nan,"0":0,"1":1})
   except: pass

   data["f1_chosen"] = data.apply(lambda row: recode_choice(row), axis=1)
   data["f_chosen"] = data["choice"].replace({"f":1,"j":0,"none":np.nan})
   data["correct_choice"] = data.apply(lambda row: get_correct_choice(row), axis=1)

   img1data = data[data.image == 1]
   img1data["img_trial_number"] = range(1,len(img1data)+1)
   img1data["stay"] = img1data.apply(lambda row: code_as_stay(row,img1data), axis=1)
   img2data = data[data.image == 2]
   img2data["img_trial_number"] = range(1,len(img2data)+1)
   img2data["stay"] = img2data.apply(lambda row: code_as_stay(row,img2data), axis=1)
   data = pd.concat([img1data,img2data]).sort_values(by="trial_number")

   data["task_time"] = pd.cut(data["trial_number"],3,labels=["early","mid","late"])
   data = data[["subject_id","outcome","f1_chosen","trial_number","rt","image","f_chosen","pFReward1","stay","task_time"]]#.dropna()
   data["pFReward2"] = 1-data["pFReward1"]

   return data

def pat_signal_detect(row):
    if row.ObjRec > 3: # think new
      if row.OldNew == 0:
        return "cr"
      else:
        return "miss"
    elif row.ObjRec < 3:
      if row.OldNew == 1:
        return "hit"
      else:
        return "fa"
    else:
      return "idk"

def ctrl_signal_detect(row):
    if 'old' in row.response:
        if row.object_type == "old":
            return "hit"
        elif row.object_type == "new":
            return "fa"
    elif 'new' in row.response:
        if row.object_type == "old":
            return "miss"
        elif row.object_type == "new":
            return "cr"
    elif row.response == "dont_know":
        return "idk"

def compute_dprime(n_Hit=None,n_Miss=None,n_FA=None,n_CR=None):
    import scipy

    # Ratios
    hit_rate = n_Hit/(n_Hit + n_Miss)
    fa_rate = n_FA/(n_FA + n_CR)

    # Adjusted ratios
    hit_rate_adjusted = (n_Hit+ 0.5)/((n_Hit+ 0.5) + n_Miss + 1)
    fa_rate_adjusted = (n_FA+ 0.5)/((n_FA+ 0.5) + n_CR + 1)

    # dprime
    dprime = scipy.stats.norm.ppf(hit_rate_adjusted) - scipy.stats.norm.ppf(fa_rate_adjusted)

    return dprime


def mem_process(data_dir,ud):
    n_subs, orig_data = load_mturk_data(data_dir=data_dir)
    obj_values = orig_data[orig_data.phase == "choice"].groupby(["subject_id","old_object"]).old_value.mean().reset_index(name="old_value")
    mem_data = orig_data[orig_data.phase.isin(["recognition_mem","choice_mem","value_mem"])]#.phase.unique()
    task2_data = orig_data[orig_data.phase == "choice"]
    data = clean_hybrid_data(data=task2_data)

    # Create recognition memory dataframe
    d4mem = ud.merge(data,on=["subject_id","trial_number","old_chosen","blue_object","red_object","old_value"])
    recog_data = mem_data[mem_data.phase == "recognition_mem"]
    recog_data['response_type'] = recog_data.apply(lambda row: signal_detect(row),axis=1)
    objMemData = []
    for i, row, in recog_data.iterrows():
        sid = row.subject_id
        obj = row.object
        sd = d4mem[d4mem.subject_id == sid]
        obj_data = sd[(sd.blue_object == obj) | (sd.red_object == obj)][["old_trial","old_chosen","old_value"]]
        obj_data["view_number"] = range(1,len(obj_data)+1)
        obj_data["subject_id"] = sid
        obj_data["object"] = obj
        obj_data["response_type"] = row.response_type
        objMemData.append(obj_data)
    objMemData = pd.concat(objMemData).reset_index(drop=True)
    objMemData["accuracy"] = objMemData["response_type"].replace({"hit":1,"miss":0,"idk":np.nan})

    # Get value memory dataframe
    value_mem = mem_data[(mem_data.phase == "value_mem") & (mem_data.button_pressed.notnull())]
    true_value = []
    for i, row in value_mem.iterrows():
        val = obj_values[(obj_values.subject_id == row.subject_id) & (obj_values.old_object == row.object)]["old_value"]
        try: v = float(val)
        except: v = np.nan
        true_value.append(v)
    value_mem["true_value"] = true_value
    value_mem["correct"] = value_mem["response"].astype("float") == value_mem["true_value"].astype("float")
    value_mem["value_diff"] = value_mem["true_value"].astype("float") - value_mem["response"].astype("float")

    value_mem["abs_value"] = np.abs(value_mem.true_value.astype('float')-0.5).round(2)
    objMemData["abs_value"] = np.abs(objMemData.old_value.astype('float')-0.5).round(2)

    # Get larger, merged value memory dataframe
    value_mem2 = value_mem.merge(objMemData[["subject_id","object","view_number"]],on=["subject_id","object"])
    value_mem2["accuracy"] = value_mem2.correct.replace({True:1,False:0})

    # Subset dataframes for those objs that were retrieved
    encObjMemData = objMemData[objMemData.view_number == 2]
    value_enc_mem = value_mem2[value_mem2.view_number == 2]

    # Create dprime dataframe
    sdt_data = recog_data.groupby(["subject_id","response_type"]).size().unstack(fill_value=0).stack().reset_index(name="num_responses")
    dprime_df = {'subject_id':[],'dprime':[]}
    for subj in sdt_data.subject_id.unique():
        subj_data = sdt_data[sdt_data.subject_id == subj]
        dprime = compute_dprime(n_Hit=subj_data[subj_data.response_type == "hit"].num_responses.iloc[0],
                                n_Miss=subj_data[subj_data.response_type == "miss"].num_responses.iloc[0],
                                n_FA=subj_data[subj_data.response_type == "fa"].num_responses.iloc[0],
                                n_CR=subj_data[subj_data.response_type == "cr"].num_responses.iloc[0])
        dprime_df['subject_id'].append(subj)
        dprime_df['dprime'].append(dprime)
    dprime_df = pd.DataFrame.from_dict(dprime_df)

    #dprime_df.to_csv(os.path.join(data_dir,"subsequentMemoryData/Exp1Task2DprimeData.csv"))
    #objMemData[objMemData.old_trial == True].to_csv(os.path.join(data_dir,"subsequentMemoryData/Exp1Task2RecMemData.csv"))
    #value_mem2[["prior_ru","enc_ru","delta","enc_delta","abs_value","subject_id","accuracy","object","response","true_value","view_number"]].to_csv(os.path.join(data_dir,"subsequentMemoryData/Exp1Task2ValMemData.csv"))

    return objMemData, dprime_df, value_mem, value_mem2, encObjMemData, value_enc_mem

def processCCAS(data_dir,ctrl_data):
    # Save the Control's CCAS data
    import json

    dpath = os.path.join(data_dir,"ccas/*_experiment_data.csv")

    # and then also have to get the last row!
    ccas_states = ['1_digitspan_fwd','2_semantic_fluency','3_phonemic_fluency','4_category_switching','5_similarity','6_gonogo','8_digitspan_bwd','9_verbal_recall']

    ctrl_ccas_data = []
    for i, f in enumerate(sorted(glob.glob(dpath))):
      subj_id = f.split('/')[-1].split('_')[0]
      d = pd.read_csv(f)
      d['subject_id'] = subj_id
      last_row = d.iloc[-1] # memory test result
      d = d[(d.state.isin(ccas_states)) | (d.stimulus == '<p>Starting in <b><span id="clock8">3</span></b> seconds...')]
      d = d.append(last_row)
      ctrl_ccas_data.append(d)
    ctrl_ccas_data = pd.concat(ctrl_ccas_data).reset_index(drop=True)

    fwd_d = ctrl_ccas_data[(ctrl_ccas_data.state == "1_digitspan_fwd") | (ctrl_ccas_data.stimulus == '<p>Starting in <b><span id="clock8">3</span></b> seconds...')]
    bwd_d = ctrl_ccas_data[(ctrl_ccas_data.state == "8_digitspan_bwd") | (ctrl_ccas_data.stimulus == '<p>Starting in <b><span id="clock8">3</span></b> seconds...')]
    ds_fwd_stims = [[1,7],[6,3],[5,8,2],[6,9,4],[6,4,3,9],[7,2,8,6],[4,2,7,3,1],[7,5,8,3,6],
                    [6,1,9,4,7,3],[3,9,2,4,8,7],[5,9,1,7,4,2,8],[4,1,7,9,3,8,6],[5,8,1,9,2,6,4,7],
                    [3,8,2,9,5,1,7,4],[2,7,5,8,6,2,5,8,4],[7,1,3,9,4,2,5,6,8]]
    ds_bwd_stims = [[2,4],[5,7],[6,2,9],[4,1,5],[3,2,7,9],[4,9,6,8],[1,5,2,8,6],
                    [6,1,8,4,3],[5,3,9,4,1,8],[7,2,4,8,5,6],[8,1,2,9,3,6,5],
                    [4,7,3,9,1,2,8],[9,4,3,7,6,2,5,8],[7,2,8,1,9,6,5,3]]
    [d.reverse() for d in ds_bwd_stims]
    def get_ds_score(sd,stims):
      answers = ','.join(sd.response.astype('str').tolist()).split('nan')
      responses = []
      for f in answers:
        try:
          resps = [int(i[0][0]) for i in f.split()[0].split(',')[1:-1]]
          if len(resps) > 0:
            responses.append(resps)
        except:
          pass

      scorer = 0
      for i, fr  in enumerate(responses):
        if fr == stims[i]:
          scorer+=1

      return scorer

    ds_fwd_answers, ds_bwd_answers = [], []
    for s in fwd_d.subject_id.unique():
      ds_fwd_answers.append(get_ds_score(fwd_d[fwd_d.subject_id == s],ds_fwd_stims))
      ds_bwd_answers.append(get_ds_score(bwd_d[bwd_d.subject_id == s],ds_bwd_stims))

    subject_ids = list(ctrl_ccas_data[ctrl_ccas_data.state == "2_semantic_fluency"]["subject_id"])

    semantic_fluency_answers = [json.loads(i)['Q0'] for i in ctrl_ccas_data[ctrl_ccas_data.state == "2_semantic_fluency"]["responses"]]
    phonemic_fluency_answers = [json.loads(i)['Q0'] for i in ctrl_ccas_data[ctrl_ccas_data.state == "3_phonemic_fluency"]["responses"]]
    category_switching_answers = [json.loads(i)['Q0'] for i in ctrl_ccas_data[ctrl_ccas_data.state == "4_category_switching"]["responses"]]
    retrieval_answers = ctrl_ccas_data[(ctrl_ccas_data.trial_type == "survey-text") & (ctrl_ccas_data.state != ctrl_ccas_data.state) & (ctrl_ccas_data.trial_index != 1)]["responses"]
    retrieval_answers = [json.loads(i)['Q0'] for i in retrieval_answers]
    similarity_q1_answers = [json.loads(i)['Q0'] for i in ctrl_ccas_data[ctrl_ccas_data.state == "5_similarity"]["responses"]]
    similarity_q2_answers = [json.loads(i)['Q1'] for i in ctrl_ccas_data[ctrl_ccas_data.state == "5_similarity"]["responses"]]
    similarity_q3_answers = [json.loads(i)['Q2'] for i in ctrl_ccas_data[ctrl_ccas_data.state == "5_similarity"]["responses"]]
    similarity_q4_answers = [json.loads(i)['Q3'] for i in ctrl_ccas_data[ctrl_ccas_data.state == "5_similarity"]["responses"]]
    similarity_answers = ["%s,%s,%s,%s"%(similarity_q1_answers[i],similarity_q2_answers[i],similarity_q3_answers[i],similarity_q4_answers[i]) for i in range(len(subject_ids))]
    gng_data = ctrl_ccas_data[ctrl_ccas_data.state == "6_gonogo"]
    gng_data["responses"] = gng_data["responses"].astype("int")
    gng_answers = list(gng_data.groupby(["subject_id"])["responses"].sum())

    # Save answers in a separate data file
    ctrl_ccas_data = {"subject_id":[],"test_name":[],"answers":[]}

    ctrl_ccas_data["subject_id"].extend(subject_ids)
    ctrl_ccas_data["test_name"].extend(["1_digitspan_fwd"] * len(subject_ids))
    ctrl_ccas_data["answers"].extend(ds_fwd_answers)

    ctrl_ccas_data["subject_id"].extend(subject_ids)
    ctrl_ccas_data["test_name"].extend(["2_semantic_fluency"] * len(subject_ids))
    ctrl_ccas_data["answers"].extend(semantic_fluency_answers)

    ctrl_ccas_data["subject_id"].extend(subject_ids)
    ctrl_ccas_data["test_name"].extend(["3_phonemic_fluency"] * len(subject_ids))
    ctrl_ccas_data["answers"].extend(phonemic_fluency_answers)

    ctrl_ccas_data["subject_id"].extend(subject_ids)
    ctrl_ccas_data["test_name"].extend(["4_category_switching"] * len(subject_ids))
    ctrl_ccas_data["answers"].extend(category_switching_answers)

    ctrl_ccas_data["subject_id"].extend(subject_ids)
    ctrl_ccas_data["test_name"].extend(["9_verbal_recall"] * len(subject_ids))
    ctrl_ccas_data["answers"].extend(retrieval_answers)

    ctrl_ccas_data["subject_id"].extend(subject_ids)
    ctrl_ccas_data["test_name"].extend(["5_similarity"] * len(subject_ids))
    ctrl_ccas_data["answers"].extend(similarity_answers)

    ctrl_ccas_data["subject_id"].extend(subject_ids)
    ctrl_ccas_data["test_name"].extend(["6_gonogo"] * len(subject_ids))
    ctrl_ccas_data["answers"].extend(gng_answers)

    ctrl_ccas_data["subject_id"].extend(subject_ids)
    ctrl_ccas_data["test_name"].extend(["8_digitspan_bwd"] * len(subject_ids))
    ctrl_ccas_data["answers"].extend(ds_bwd_answers)

    ctrl_ccas_data = pd.DataFrame.from_dict(ctrl_ccas_data)

    ctrl_ccas_data.to_csv("/Users/jonathannicholas/gradschool/manuscripts/ataxia-rl/data/ccasCtrlData.csv")

    return ctrl_ccas_data

def ccas4Stats(ctrl_data,ccas,data_dir):
    ccas['Total Score (No Bwd)'] = ccas[['1_semantic_fluency','2_phonemic_fluency','3_category_switching','4_verbal_recall','5_digitspan_fwd','7_similarity','8_gonogo']].sum(axis=1)
    ccas["subject_id"] = ccas["subject_id"].astype("int")

    a = ctrl_data[["subject_id",'ctrl_patient_id']].groupby(["subject_id","ctrl_patient_id"]).size().reset_index()
    a["subject_id"] = a["subject_id"].astype("int")
    b = a.merge(ccas[["Total Score","Total Score (No Bwd)","group","subject_id"]],on=["subject_id"])
    b["patient_id"] = b["ctrl_patient_id"].astype("int")
    b = b.drop("ctrl_patient_id",axis=1)
    c = ccas[ccas.group == "PAT"][["Total Score","Total Score (No Bwd)","group","subject_id"]]
    c["patient_id"] = c["subject_id"]
    d = pd.concat([b,c]).drop(0,axis=1)
    d["group_coded"] = d.group.replace({"PAT":0.5,"CTRL":-0.5})
    d = d.rename({"Total Score":"total_score","Total Score (No Bwd)":"total_score_no_bwd"},axis=1)
    d.to_csv(os.path.join(data_dir,"ccasRmForStats.csv"))
    return d

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
