#!/usr/bin/python3
#Importing modules
import numpy as np, pandas as pd
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE,ADASYN)
from statistics import mean
import re,sys,os
import nltk.corpus
from nltk.corpus import stopwords
from scipy.stats import wilcoxon

#Evaluation Metrics
from sklearn import metrics
from sklearn.metrics import (classification_report,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,precision_recall_fscore_support,roc_auc_score,roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.metrics import (geometric_mean_score,sensitivity_score,specificity_score,sensitivity_specificity_support)

def generate_rslt_tables():
  df_lst=[]
  baseline_ky=""
  weighted_ky=""
  high_prec_smpl_ky=""
  high_rcl_smpl_ky=""
  high_f1_smpl_ky=""
  high_auc_smpl_ky=""
  mean_prec_val=0
  mean_rcl_val=0
  mean_f1_val=0
  mean_auc_val=0
  
  rslt_summary_dict={}
  prec_summary_table_dict=[]#{'prec':[]}
  rcl_summary_table_dict=[]#{'rcl':[]}
  f1_summary_table_dict=[]#{'f1':[]}
  auc_summary_table_dict=[]#{'auc':[]}
  mean_rslts_dic={'prec-mean':[],'rcl-mean':[],'f1-mean':[],'roc-auc-mean':[],'spec-mean':[],'sens-mean':[],'gm-mean':[]}
  for smplng_strtgy_key in tst_rslt_mstr_dic:
    for proj_ky in tst_rslt_mstr_dic[smplng_strtgy_key]:
      if proj_ky == "overall_test":
        continue
      logging.info("proj_ky: %s",proj_ky)
      mean_rslts_dic['prec-mean'].append(tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["prec"])
      mean_rslts_dic['rcl-mean'].append(tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["rcl"])
      mean_rslts_dic['f1-mean'].append(tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["f1"])
      mean_rslts_dic['roc-auc-mean'].append(tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["roc_auc"])
      mean_rslts_dic['spec-mean'].append(tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["spec"])
      mean_rslts_dic['sens-mean'].append(tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["sens"])
      mean_rslts_dic['gm-mean'].append(tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["gm"])
      logging.info("prec-mean: %s",tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["prec"])
      logging.info("rcl-mean: %s",tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["rcl"])
      logging.info("f1-mean: %s",tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["f1"])
      logging.info("roc_auc: %s",tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["roc_auc"])		
      df_rec={'Project':proj_ky,'Precision': tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["prec"], 'Recall':  tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["rcl"],'F1':tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["f1"], 'ROC-AUC':tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["roc_auc"],'Specificity':tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["spec"],'Sensitivity':tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["sens"],'Geometric-Mean':tst_rslt_mstr_dic[smplng_strtgy_key][proj_ky]["gm"]}
      df_lst.append(df_rec)
    logging.info("mean(mean_rslts_dic[prec-mean] : %s",mean(mean_rslts_dic['prec-mean']))
    logging.info("mean(mean_rslts_dic[rcl-mean] : %s",mean(mean_rslts_dic['rcl-mean']))
    logging.info("mean(mean_rslts_dic[f1-mean] : %s",mean(mean_rslts_dic['f1-mean']))
    logging.info("mean(mean_rslts_dic[roc-auc-mean] : %s",mean(mean_rslts_dic['roc-auc-mean']))
    df_rec={'Project':"Mean",'Precision': mean(mean_rslts_dic['prec-mean']), 'Recall':  mean(mean_rslts_dic['rcl-mean']),'F1':mean(mean_rslts_dic['f1-mean']), 'ROC-AUC':mean(mean_rslts_dic['roc-auc-mean']),'Specificity':mean(mean_rslts_dic['spec-mean']),'Sensitivity':mean(mean_rslts_dic['sens-mean']),'Geometric-Mean':mean(mean_rslts_dic['gm-mean'])}
    rslt_summary_dict[smplng_strtgy_key]=mean_rslts_dic
    prec_summary_table_dict.append(round(mean(mean_rslts_dic['prec-mean']),3))
    rcl_summary_table_dict.append(round(mean(mean_rslts_dic['rcl-mean']),3))
    f1_summary_table_dict.append(round(mean(mean_rslts_dic['f1-mean']),3))
    auc_summary_table_dict.append(round(mean(mean_rslts_dic['roc-auc-mean']),3))
    if "weighted" in smplng_strtgy_key:
        weighted_ky=smplng_strtgy_key
    if "baseline" in smplng_strtgy_key:
        baseline_ky=smplng_strtgy_key
    else:
        if mean(mean_rslts_dic['prec-mean']) > mean_prec_val:
            mean_prec_val=mean(mean_rslts_dic['prec-mean'])
            high_prec_smpl_ky=smplng_strtgy_key
        if mean(mean_rslts_dic['rcl-mean']) > mean_rcl_val:
            mean_rcl_val=mean(mean_rslts_dic['rcl-mean'])
            high_rcl_smpl_ky=smplng_strtgy_key
        if mean(mean_rslts_dic['f1-mean']) > mean_f1_val:
            mean_f1_val=mean(mean_rslts_dic['f1-mean'])
            high_f1_smpl_ky=smplng_strtgy_key
        if mean(mean_rslts_dic['roc-auc-mean']) > mean_auc_val:
            mean_auc_val=mean(mean_rslts_dic['roc-auc-mean'])
            high_auc_smpl_ky=smplng_strtgy_key        
    mean_tst_rslts_mstr_dic[smplng_strtgy_key[:smplng_strtgy_key.index('_')]]["prec"].append(round(mean(mean_rslts_dic['prec-mean']),3))
    mean_tst_rslts_mstr_dic[smplng_strtgy_key[:smplng_strtgy_key.index('_')]]["rcl"].append(round(mean(mean_rslts_dic['rcl-mean']),3))
    mean_tst_rslts_mstr_dic[smplng_strtgy_key[:smplng_strtgy_key.index('_')]]["roc-auc"].append(round(mean(mean_rslts_dic['roc-auc-mean']),3))
    mean_tst_rslts_mstr_dic[smplng_strtgy_key[:smplng_strtgy_key.index('_')]]["f1"].append(round(mean(mean_rslts_dic['f1-mean']),3))
    df_lst.append(df_rec)  
    tst_rslt_df = pd.DataFrame(df_lst)
    with  open(output_path+model_name+"_"+smplng_strtgy_key+".tex", 'w') as fh:
        fh.write(tst_rslt_df.to_latex())
    df_lst=[]
    mean_rslts_dic={'prec-mean':[],'rcl-mean':[],'f1-mean':[],'roc-auc-mean':[],'spec-mean':[],'sens-mean':[],'gm-mean':[]}
  
  stat_test(rslt_summary_dict,weighted_ky,baseline_ky,'prec-mean',"wb")
  stat_test(rslt_summary_dict,high_prec_smpl_ky,baseline_ky,'prec-mean',"sb")
  stat_test(rslt_summary_dict,high_prec_smpl_ky,weighted_ky,'prec-mean',"sw")
  
  stat_test(rslt_summary_dict,weighted_ky,baseline_ky,'rcl-mean',"wb")
  stat_test(rslt_summary_dict,high_rcl_smpl_ky,baseline_ky,'rcl-mean',"sb")
  stat_test(rslt_summary_dict,high_rcl_smpl_ky,weighted_ky,'rcl-mean',"sw")

  stat_test(rslt_summary_dict,weighted_ky,baseline_ky,'f1-mean',"wb")
  stat_test(rslt_summary_dict,high_f1_smpl_ky,baseline_ky,'f1-mean',"sb")
  stat_test(rslt_summary_dict,high_f1_smpl_ky,weighted_ky,'f1-mean',"sw")  

  stat_test(rslt_summary_dict,weighted_ky,baseline_ky,'roc-auc-mean',"wb")
  stat_test(rslt_summary_dict,high_auc_smpl_ky,baseline_ky,'roc-auc-mean',"sb")
  stat_test(rslt_summary_dict,high_auc_smpl_ky,weighted_ky,'roc-auc-mean',"sw")
  
  logging.info("prec_summary_table & avg: %s,%s",prec_summary_table_dict,round(mean(prec_summary_table_dict),3))
  logging.info("rcl_summary_table & avg: %s,%s",rcl_summary_table_dict,round(mean(rcl_summary_table_dict),3))
  logging.info("f1_summary_table & avg: %s,%s",f1_summary_table_dict,round(mean(f1_summary_table_dict),3))
  logging.info("auc_summary_table & avg: %s,%s",auc_summary_table_dict,round(mean(auc_summary_table_dict),3))

def chart_values(rslt_dict):
  for smplng_algo in rslt_dict:
    logging.info("%s",smplng_algo)
    for val in rslt_dict[smplng_algo]:
      logging.info("%s : (%s)",val,rslt_dict[smplng_algo][val])	
    logging.info("----------------------------------------------------------------")

def stat_test(rslt_summary_dict,key1,key2,metric_key,tst_for):

  if mean(rslt_summary_dict[key1][metric_key]) >mean(rslt_summary_dict[key2][metric_key]):
      if tst_for == "wb":
          logging.info("Weighted mean > Baseline for %s : %s,%s",metric_key,mean(rslt_summary_dict[key1][metric_key]),mean(rslt_summary_dict[key2][metric_key]) )
      elif tst_for == "sb":
          logging.info("Sampling mean > Baseline for %s : %s,%s",metric_key,mean(rslt_summary_dict[key1][metric_key]),mean(rslt_summary_dict[key2][metric_key]) )
      elif tst_for == "sw":
          logging.info("Sampling mean > Weighted for %s : %s,%s",metric_key,mean(rslt_summary_dict[key1][metric_key]),mean(rslt_summary_dict[key2][metric_key]) )

      perc_growth = [(((y - x) * 100) / x) for x, y in zip(rslt_summary_dict[key2][metric_key],rslt_summary_dict[key1][metric_key]) ]
      logging.info(" perc_growth: %s",perc_growth )
      logging.info(" avg perc_growth: %s",mean(perc_growth) )
      stat, p_value = wilcoxon(rslt_summary_dict[key2][metric_key], rslt_summary_dict[key1][metric_key])
      logging.info("stat,p_value %s,%s",stat,p_value)
      if p_value > 0.05: #95% confidence level
          logging.info("No statistically significant change inferred")
      else:
          logging.info("Statistically significant change inferred")
  else:
      if tst_for == "wb":
          logging.info("Need not calc statistical test as Weighted mean < Baseline for %s : %s,%s",metric_key,mean(rslt_summary_dict[key1][metric_key]),mean(rslt_summary_dict[key2][metric_key]) )
      if tst_for == "sb":
          logging.info("Need not calc statistical test as Sampling mean < Baseline for %s : %s,%s",metric_key,mean(rslt_summary_dict[key1][metric_key]),mean(rslt_summary_dict[key2][metric_key]) )        
      if tst_for == "sw":
          logging.info("Need not calc statistical test as Sampling mean < Weighted for %s : %s,%s",metric_key,mean(rslt_summary_dict[key1][metric_key]),mean(rslt_summary_dict[key2][metric_key]) )              

          
def proc_train_test_cross_proj(td_dataset_dict):
    td_vctrzd_dataset_dict={}
    train_tst_lst=[]
    tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
    for  ky in td_dataset_dict["train_data"]:
      data_clean = clean_text(td_dataset_dict["train_data"][ky], 'commenttext', 'commenttext')
      data_clean['commenttext'] = data_clean['commenttext'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	
      train_tst_lst=[]
      data_clean_tst_tmp = clean_text(td_dataset_dict["test_data"][ky], 'commenttext', 'commenttext')
      data_clean_tst_tmp['commenttext'] = data_clean_tst_tmp['commenttext'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
      X_temp=data_clean_tst_tmp['commenttext']
      vctrzr=tfidf_vectorizer.fit(data_clean['commenttext'])
      
      X_temp=vctrzr.transform(X_temp)
      Y_temp=data_clean_tst_tmp['tag']
      train_tst_lst.append(X_temp)
      train_tst_lst.append(Y_temp)
      train_tst_lst.append(td_dataset_dict["test_data"][ky]['commenttext'])
      td_vctrzd_dataset_test[ky]=train_tst_lst		  		
	  
      X=data_clean['commenttext']
      X=vctrzr.transform(X)
      Y=data_clean['tag']
      train_tst_lst=[]
      train_tst_lst.append(X)
      train_tst_lst.append(Y)
      td_vctrzd_dataset_train[ky]=train_tst_lst
	
    td_vctrzd_dataset_dict["test_data"]=td_vctrzd_dataset_test
    td_vctrzd_dataset_dict["train_data"]=td_vctrzd_dataset_train
    return td_vctrzd_dataset_dict




def ml_model(iteration):
    
    logging.info("%s Modeling begins",model_name)
    n_folds=10
    kfold=StratifiedKFold(n_splits=n_folds, shuffle=True,random_state =iteration+5 )
    global bal_tech


    global val_conf_mat_tp,cross_val_prec_list,cross_val_rcl_list,cross_val_f1_list,cross_val_rocauc_list,tst_rslt_dic,tst_rslt_mstr_dic
    global cross_val_sens_list,cross_val_spec_list,cross_val_gm_list,model_history,best_model_ind,mstr_tst_df,scale_positive_weight
    for ovrsmplng_algo in sampling_algo_lst:
      for i in sampling_strategy_lst:
        for tst_proj_ky in vctrzd_td_dataset["train_data"]:
          logging.info("Model Fitting for %s data %s sampling strategy %s begins for %s iteration",tst_proj_ky,ovrsmplng_algo,i,iteration)
          pckl_model=str(i).replace(".","_")+"_"+ovrsmplng_algo+"_"+model_name+"_"+tst_proj_ky+"_"+"model.pkl"
          model_file_path=model_path + pckl_model 
          logging.info("Model path: %s",model_file_path)
          X_train_data=mstr_sampling_dict_x[tst_proj_ky][ovrsmplng_algo][i]
          Y_train_data=mstr_sampling_dict_y[tst_proj_ky][ovrsmplng_algo][i]
          for train_index, test_index in kfold.split(X_train_data,Y_train_data):
            # train-test split
            X_train = X_train_data[train_index]
            Y_train = Y_train_data[train_index]
            X_test = X_train_data[test_index]
            Y_test = Y_train_data[test_index]

            logging.info("X_train.shape: %s ",X_train.shape) 
            logging.info("X_test.shape: %s ",X_test.shape) 
            logging.info("Y_train.shape: %s ",Y_train.shape) 
            logging.info("Y_test.shape: %s ",Y_test.shape) 
            if "weighted" in ovrsmplng_algo:
                bal_tech="weighted"
                logging.info("1 bal_tech: %s ",bal_tech)
                scale_pos_cntr=Counter(Y_train)
                scale_positive_weight=int(scale_pos_cntr[0]/scale_pos_cntr[1])                
                logging.info("scale_positive_weight: %s ",scale_positive_weight)
                logging.info("scale_pos_cntr 0,1,wgt: %s,%s,%s ",scale_pos_cntr[0],scale_pos_cntr[1],scale_positive_weight)                
            else:
                bal_tech="other"

            model_history.append(model_fitting(model_name,X_train, X_test, Y_train, Y_test))#, epochs, batch_size))
          logging.info("Model Fitting for %s data %s sampling strategy %s ends for %s iteration",tst_proj_ky,ovrsmplng_algo,i,iteration)
          logging.info("=====================================================================")
          logging.info("%s - Key Observations for %s sampling strategy %s ",iteration,ovrsmplng_algo,i)
          logging.info("----------------")
          for v_cnf_mt_tp in [val_conf_mat_tp]:logging.info("TP from each cv: %s",v_cnf_mt_tp)      
          best_model_ind=np.argmax(val_conf_mat_tp)

          logging.info("Best Performing Model index: %s",best_model_ind)
          logging.info("Precision: %s",cross_val_prec_list[best_model_ind])
          logging.info("Recall: %s",cross_val_rcl_list[best_model_ind])
          logging.info("F1: %s",cross_val_f1_list[best_model_ind])
          logging.info("ROC-AUC: %s",cross_val_rocauc_list[best_model_ind])
          logging.info("Geometric Mean: %s",cross_val_gm_list[best_model_ind])
          logging.info("Sensitivity: %s",cross_val_sens_list[best_model_ind])
          logging.info("Specificity: %s",cross_val_spec_list[best_model_ind])
          logging.info("*********")
          logging.info("Mean Precision: %s",mean(cross_val_prec_list))
          logging.info("Mean Recall: %s",mean(cross_val_rcl_list))
          logging.info("Mean F1: %s",mean(cross_val_f1_list))
          logging.info("Mean ROC-AUC: %s",mean(cross_val_rocauc_list))
          logging.info("Mean Geometric Mean: %s",mean(cross_val_gm_list))
          logging.info("Mean Sensitivity: %s",mean(cross_val_sens_list))
          logging.info("Mean Specificity: %s",mean(cross_val_spec_list))

          logging.info("*********overall test data********* %s",tst_proj_ky)
          test_stats(model_history[best_model_ind],vctrzd_td_dataset["test_data"][tst_proj_ky][0],vctrzd_td_dataset["test_data"][tst_proj_ky][1],ovrsmplng_algo+"_"+str(i).replace(".","_"),tst_proj_ky,"module_test",vctrzd_td_dataset["test_data"][tst_proj_ky][2])
          #logging.info("%s - Saving the model for %s data for sampling strategy %s... at  %s ",ovrsmplng_algo, tst_proj_ky,i, model_file_path)    
          #pickle.dump(model_history[best_model_ind],open(model_file_path, 'wb'))
          
          val_conf_mat_tp=[]
          cross_val_prec_list=[]
          cross_val_rcl_list=[]
          cross_val_f1_list=[]
          cross_val_rocauc_list=[]
          cross_val_sens_list=[]
          cross_val_spec_list=[]
          cross_val_gm_list=[]
          model_history = []
          best_model_ind=0
        mstr_tst_df.to_csv(output_path+model_name+'_'+ovrsmplng_algo+'_'+str(i)+'_'+str(iteration)+".csv")
        mstr_tst_df=pd.DataFrame()
        tst_rslt_mstr_dic[ovrsmplng_algo+"_"+str(i).replace(".","_")]=tst_rslt_dic
        tst_rslt_dic={}
             

    logging.info("%s Modeling ends",model_name)
		
def test_stats(model,test_inp,ground_truth_inp,smplng_strtgy_ky,tst_proj_ky,module_flag,df_cmnts):
    rslt={}
    
    global tst_rslt_dic,mstr_tst_df
    event_time=time.time()
    y_class = model.predict(test_inp)
    y_pred_prob =  model.predict_proba(test_inp)[:,1]
    logging.info("test data for %s",tst_proj_ky)
    logging.info("-inference time- %s seconds ---" % (time.time() - event_time))
    logging.info(classification_report(ground_truth_inp, y_class)) 
    logging.info(precision_recall_fscore_support(ground_truth_inp,y_class,average='binary'))
    logging.info(confusion_matrix(ground_truth_inp, y_class))

    rslt["proj"]=tst_proj_ky
    rslt["prec"]=precision_score(ground_truth_inp, y_class, average='binary')
    rslt["rcl"]=recall_score(ground_truth_inp, y_class, average='binary')
    rslt["f1"]=f1_score(ground_truth_inp, y_class,average='binary')
    rslt["roc_auc"]=roc_auc_score(ground_truth_inp,y_class)
    rslt["spec"]=specificity_score(ground_truth_inp,y_class,average='binary')
    rslt["sens"]=sensitivity_score(ground_truth_inp,y_class,average='binary')
    rslt["gm"]=geometric_mean_score(ground_truth_inp,y_class,average='binary')

    if module_flag == "module_test":
      tst_rslt_dic[tst_proj_ky]=rslt
      data={'Project':ground_truth_inp,'Comments':df_cmnts,'Ground_Truth':ground_truth_inp}
      tst_df=pd.DataFrame(data)
      tst_df['Project'] = tst_proj_ky
      tst_df['Predicted_Class'] = y_class
      tst_df['Predicted_Probability'] = y_pred_prob
      mstr_tst_df=mstr_tst_df.append(tst_df,ignore_index=True)    
      logging.info("mstr_tst_df shape: %s",mstr_tst_df.shape)

    #tst_rslt_mstr_dic[smplng_strtgy_ky]=tst_rslt_dic
    
    logging.info("%s : %s : Precision: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["prec"])
    logging.info("%s : %s : Recall: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["rcl"])
    logging.info("%s : %s : F1: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["f1"])
    logging.info("%s : %s : ROC-AUC: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["roc_auc"])
    logging.info("%s : %s : Specificity: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["spec"])
    logging.info("%s : %s : Sensitivity: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["sens"])
    logging.info("%s : %s : Geometric Mean: %s",smplng_strtgy_ky,tst_proj_ky,rslt["gm"])

	
def model_fitting(model_name,train_x, val_x, train_y, val_y):#, EPOCHS=20, BATCH_SIZE=32):
  global model_history,best_model_ind,val_conf_mat_tp,cross_val_prec_list,cross_val_rcl_list,cross_val_f1_list,cross_val_rocauc_list,cross_val_sens_list,cross_val_spec_list,cross_val_gm_list
  global scale_positive_weight
  if model_name.lower() == "lr" :
    logging.info("Fitting a Logistic Regression Model...")
    if bal_tech == "weighted":
        scikit_log_reg = LogisticRegression(verbose=1,solver='liblinear',class_weight='balanced',max_iter=1000)
    else:
        scikit_log_reg = LogisticRegression(verbose=1,solver='liblinear',max_iter=1000)
    event_time=time.time()
    model=scikit_log_reg.fit(train_x,train_y)
    logging.info("Training time for LR model: %s",(time.time()-event_time))
  elif model_name.lower() == 'gb' :
    logging.info("Fitting a Gradient Boost Model...")
    if bal_tech == "weighted":
        scikit_gradboost = XGBClassifier(scale_pos_weight=scale_positive_weight)
    else:
        scikit_gradboost = XGBClassifier()
    event_time=time.time()
    model=scikit_gradboost.fit(train_x,train_y)
    logging.info("Training time for GB model: %s",(time.time()-event_time))
  elif model_name.lower() == 'rf' :
    logging.info("Fitting a Random Forest Model...")
    if bal_tech == "weighted":
        scikit_randomforest = RandomForestClassifier(class_weight='balanced_subsample')
    else:
        scikit_randomforest = RandomForestClassifier()
    event_time=time.time()
    model=scikit_randomforest.fit(train_x,train_y)
    logging.info("Training time for RF model: %s",(time.time()-event_time))
  
  event_time=time.time()
  train_pred_y=model.predict(train_x)
  logging.info("Training-Prediction time: %s",(time.time()-event_time))
  conf_mat = confusion_matrix(train_y, train_pred_y)
  logging.info("Confusion Matrix")
  logging.info("%s",conf_mat)
  logging.info("%s",classification_report(train_y, train_pred_y))
  logging.info("Cross-Validating with validation data...")
  event_time=time.time()
  val_pred_y=model.predict(val_x)
  ##Sensitivity -- Recall of +ve class (in binary classification)
  ##Specificity -- Recall of -ve class (in binary classification)
  logging.info("Cross-Validation Prediction time: %s",(time.time()-event_time))
  val_conf_mat=confusion_matrix(val_y, val_pred_y)
  logging.info("%s",val_conf_mat)
  val_conf_mat_tp.append(val_conf_mat[1][1])
  logging.info("%s",classification_report(val_y, val_pred_y))
  cross_val_prec_list.append(precision_score(val_y, val_pred_y, average='binary'))
  cross_val_rcl_list.append(recall_score(val_y, val_pred_y, average='binary'))
  cross_val_f1_list.append(f1_score(val_y, val_pred_y, average='binary'))
  cross_val_rocauc_list.append(roc_auc_score(val_y, val_pred_y))
  cross_val_sens_list.append(specificity_score(val_y, val_pred_y,average='binary'))
  cross_val_spec_list.append(sensitivity_score(val_y, val_pred_y, average='binary'))
  cross_val_gm_list.append(geometric_mean_score(val_y, val_pred_y, average='binary'))
  logging.info("Precision: %s ",precision_score(val_y, val_pred_y, average='binary'))
  logging.info("Recall: %s ",recall_score(val_y, val_pred_y, average='binary'))
  logging.info("F1: %s ",f1_score(val_y, val_pred_y, average='binary'))
  logging.info("ROC-AUC: %s ",roc_auc_score(val_y, val_pred_y))
  logging.info("Specificity: %s ",specificity_score(val_y, val_pred_y,average='binary'))
  logging.info("Sensitivity: %s ",sensitivity_score(val_y, val_pred_y, average='binary'))
  logging.info("Geometric Mean: %s",geometric_mean_score(val_y, val_pred_y, average='binary'))
  return model

# Removes punctuation and special characters
def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

############################ Main ####################################
model_name=sys.argv[1]
output_path='/scratch/project_2002565/bal_td/cross_rslt_tables/'
log_path='/scratch/project_2002565/bal_td/cross_log/'
model_path="/scratch/project_2002565/bal_td/chk_pt/"
encoder_path="/scratch/project_2002565/bal_td/encoder/"
input_file='/scratch/project_2002565/bal_td/data/technical_debt_dataset.csv'
tech_debt_df = pd.read_csv(input_file,header=0,encoding='latin-1')
tech_debt_df.loc[tech_debt_df.classification == "DEFECT",'tag']=int(1)
tech_debt_df.loc[tech_debt_df.classification == "DESIGN",'tag']=int(1)
tech_debt_df.loc[tech_debt_df.classification == "DOCUMENTATION",'tag']=int(1)
tech_debt_df.loc[tech_debt_df.classification == "IMPLEMENTATION",'tag']=int(1)
tech_debt_df.loc[tech_debt_df.classification == "TEST",'tag']=int(1)
tech_debt_df.loc[tech_debt_df.classification == "WITHOUT_CLASSIFICATION",'tag']=int(0)

tech_debt_df['tag']=LabelEncoder().fit_transform(tech_debt_df.tag)

index = tech_debt_df.set_index('projectname')
jedit_df=index.loc[['jEdit-4.2']]
argouml_df=index.loc[['argouml']]
jmeter_df=index.loc[['apache-jmeter-2.10']]
sql_df=index.loc[['sql12']]
columba_df=index.loc[['columba-1.4-src']]
jruby_df=index.loc[['jruby-1.4.0']]
jfreechart_df=index.loc[['jfreechart-1.0.19']]
emf_df=index.loc[['emf-2.4.1']]
ant_df=index.loc[['apache-ant-1.7.0']]
hibernate_df=index.loc[['hibernate-distribution-3.3.2.GA']]
consolidated=index.loc[['jEdit-4.2','argouml','apache-jmeter-2.10','sql12','columba-1.4-src','jruby-1.4.0','jfreechart-1.0.19','emf-2.4.1','apache-ant-1.7.0','hibernate-distribution-3.3.2.GA'], ['classification', 'commenttext','tag']]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO,datefmt='%a, %d %b %Y %H:%M:%S', filename=log_path+model_name+'_cross_proj.log', filemode='w')    

model_path="/scratch/project_2002565/bal_td/chk_pt/"  #smote_rf_model.pkl
encoder_path="/scratch/project_2002565/bal_td/encoder/"  #smote_rf_enc.pkl
model_history = []
best_model_ind=0
val_conf_mat_tp=[]
cross_val_prec_list=[]
cross_val_rcl_list=[]
cross_val_f1_list=[]
cross_val_rocauc_list=[]
cross_val_sens_list=[]
cross_val_spec_list=[]
cross_val_gm_list=[]
mstr_sampling_dict_x={}
mstr_sampling_dict_y={}
sample_dict_x={}
sample_dict_y={}
sampling_dict_x={}
sampling_dict_y={}
mean_tst_rslts_mstr_dic={'baseline':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'weighted':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'smote':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'adasyn':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'bline':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'svmsmote':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]}}
sampling_algo_lst=["baseline","weighted","smote","adasyn","svmsmote","bline"]
sampling_strategy_lst=[1.0]
tst_rslt_mstr_dic={}
for ovrsmplng_algo in sampling_algo_lst:
    for i in sampling_strategy_lst:
        tst_rslt_mstr_dic[ovrsmplng_algo+"_"+str(i).replace(".","_")]=[]
td_dataset_train={}
td_dataset_test={}
td_vctrzd_dataset_train={}
td_vctrzd_dataset_test={}
td_dataset={}
vctrzd_td_dataset={}
mstr_y_imbalance_dict={}
tst_rslt_dic={}
scale_positive_weight=0
mstr_tst_df=pd.DataFrame()


dfs=[jedit_df,argouml_df,jmeter_df,sql_df,columba_df,jruby_df,jfreechart_df,emf_df,hibernate_df] #ant_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["Ant"]=cmbnd_dfs
td_dataset_test["Ant"]=ant_df
logging.info("%s,%s",cmbnd_dfs.shape,ant_df.shape)
dfs=[jedit_df,jmeter_df,sql_df,columba_df,jruby_df,jfreechart_df,emf_df,ant_df,hibernate_df] #argouml_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["Argouml"]=cmbnd_dfs
td_dataset_test["Argouml"]=argouml_df
logging.info("%s,%s",cmbnd_dfs.shape,argouml_df.shape)
dfs=[jedit_df,argouml_df,jmeter_df,sql_df,jruby_df,jfreechart_df,emf_df,ant_df,hibernate_df] #columba_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["Columba"]=cmbnd_dfs
td_dataset_test["Columba"]=columba_df
logging.info("%s,%s",cmbnd_dfs.shape,columba_df.shape)
dfs=[jedit_df,argouml_df,jmeter_df,sql_df,columba_df,jruby_df,jfreechart_df,ant_df,hibernate_df] #emf_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["Emf"]=cmbnd_dfs
td_dataset_test["Emf"]=emf_df
logging.info("%s,%s",cmbnd_dfs.shape,emf_df.shape)
dfs=[jedit_df,argouml_df,jmeter_df,sql_df,columba_df,jruby_df,jfreechart_df,emf_df,ant_df] #hibernate_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["Hibernate"]=cmbnd_dfs
td_dataset_test["Hibernate"]=hibernate_df
logging.info("%s,%s",cmbnd_dfs.shape,hibernate_df.shape)
dfs=[argouml_df,jmeter_df,sql_df,columba_df,jruby_df,jfreechart_df,emf_df,ant_df,hibernate_df] #jedit_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["JEdit"]=cmbnd_dfs
td_dataset_test["JEdit"]=jedit_df
logging.info("%s,%s",cmbnd_dfs.shape,jedit_df.shape)
dfs=[jedit_df,argouml_df,jmeter_df,sql_df,columba_df,jruby_df,emf_df,ant_df,hibernate_df] #jfreechart_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["JFreeChart"]=cmbnd_dfs
td_dataset_test["JFreeChart"]=jfreechart_df
logging.info("%s,%s",cmbnd_dfs.shape,jfreechart_df.shape)
dfs=[jedit_df,argouml_df,sql_df,columba_df,jruby_df,jfreechart_df,emf_df,ant_df,hibernate_df] #jmeter_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["JMeter"]=cmbnd_dfs
td_dataset_test["JMeter"]=jmeter_df
logging.info("%s,%s",cmbnd_dfs.shape,jmeter_df.shape)
dfs=[jedit_df,argouml_df,jmeter_df,sql_df,columba_df,jfreechart_df,emf_df,ant_df,hibernate_df] #jruby_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["JRuby"]=cmbnd_dfs
td_dataset_test["JRuby"]=jruby_df
logging.info("%s,%s",cmbnd_dfs.shape,jruby_df.shape)
dfs=[jedit_df,argouml_df,jmeter_df,columba_df,jruby_df,jfreechart_df,emf_df,ant_df,hibernate_df] #sql_df
cmbnd_dfs=pd.concat(dfs)
td_dataset_train["Squirrel"]=cmbnd_dfs
td_dataset_test["Squirrel"]=sql_df
logging.info("%s,%s",cmbnd_dfs.shape,sql_df.shape)
#for ky in td_dataset_train:
#  logging.info("%s,%s",ky,td_dataset_train[ky]['tag'].value_counts())

td_dataset["train_data"]=td_dataset_train
td_dataset["test_data"]=td_dataset_test

nltk.download('stopwords')
stop = stopwords.words('english')

data_clean = clean_text(consolidated, 'commenttext', 'commenttext')

data_clean['commenttext'] = data_clean['commenttext'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
vctrzr=tfidf_vectorizer.fit(data_clean['commenttext'])

vctrzd_td_dataset =proc_train_test_cross_proj(td_dataset)


logging.info("Oversampling sample creation begins")
for tst_proj_ky in vctrzd_td_dataset["train_data"]:
  X_train_data=vctrzd_td_dataset["train_data"][tst_proj_ky][0]
  Y_train_data=vctrzd_td_dataset["train_data"][tst_proj_ky][1]
  logging.info("Total #records after vectorization for test set %s X_train_data.shape: %s",tst_proj_ky,X_train_data.shape)
  logging.info("Testing Label split after vectorization for test set %s  Counter(Y_test_data): %s",tst_proj_ky,Counter(vctrzd_td_dataset["test_data"][tst_proj_ky][1]))
  logging.info("Total #records after after vectorization for test set %s X_test_data.shape: %s",tst_proj_ky,vctrzd_td_dataset["test_data"][tst_proj_ky][0].shape)
  for sampling_algo in (sampling_algo_lst):
    for x in sampling_strategy_lst:
      if(sampling_algo == "smote"):
        sampling_obj = SMOTE(sampling_strategy=x)
      elif(sampling_algo == "adasyn"):
        sampling_obj = ADASYN(sampling_strategy=x)
      elif(sampling_algo == "svmsmote"):
        sampling_obj = SVMSMOTE(sampling_strategy=x)
      elif(sampling_algo == "bline"):
        sampling_obj = BorderlineSMOTE(sampling_strategy=x)

      if(sampling_algo == "baseline" or sampling_algo == "weighted"): #if(x==0):
        X_train_sampling_data, Y_train_sampling_data = X_train_data, Y_train_data
      else:
        X_train_sampling_data, Y_train_sampling_data = sampling_obj.fit_resample(X_train_data, Y_train_data)
      sample_dict_x[x]=X_train_sampling_data
      sample_dict_y[x]=Y_train_sampling_data
      count_data=Counter(Y_train_sampling_data)
      mstr_y_imbalance_dict[tst_proj_ky]=Counter(Y_train_sampling_data)
      logging.info("******************************")
      logging.info("%s ~ Sampling Strategy ~ %s for %s",sampling_algo,x,tst_proj_ky)
      logging.info("X_train_sampling_data shape: %s",X_train_sampling_data.shape) 
      logging.info("Y_train_sampling_data shape: %s",Y_train_sampling_data.shape)
      logging.info("Sampling Split for train data: %s",Counter(Y_train_sampling_data))
      logging.info("Sampling Split for test data %s: %s",tst_proj_ky,Counter(vctrzd_td_dataset["test_data"][tst_proj_ky][1]))      
      logging.info("Sample created with %s Oversampling algorithm with sampling strategy %s",x,sampling_algo)
      logging.info("******************************")
    sampling_dict_x[sampling_algo]=sample_dict_x
    sampling_dict_y[sampling_algo]=sample_dict_y
    sample_dict_x={}
    sample_dict_y={}
  mstr_sampling_dict_x[tst_proj_ky]=sampling_dict_x
  mstr_sampling_dict_y[tst_proj_ky]=sampling_dict_y
  sampling_dict_x={}
  sampling_dict_y={}
  
try:
    for iteration in range(10):
        ml_model(iteration)
    generate_rslt_tables()
    logging.info("*********** CV ************")
    chart_values(cv_rslt_dic)
    logging.info("*********** TST ***************")	
    chart_values(mean_tst_rslts_mstr_dic)    
except Exception as details:
    logging.info('Unexpected error: %s',details)
    print('Unexpected error: {0}'.format(details))
    os._exit(0)
