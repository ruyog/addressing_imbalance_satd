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
import re
import nltk.corpus
from nltk.corpus import stopwords
import sys
from scipy.stats import wilcoxon

#Evaluation Metrics
from sklearn.metrics import (classification_report,precision_score,recall_score,f1_score,confusion_matrix,precision_recall_fscore_support,roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.metrics import (geometric_mean_score,sensitivity_score,specificity_score)

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
                  
  averaging_test_scores()
  
  logging.info("tst_rslt_mstr_dic keys: %s",tst_rslt_mstr_dic.keys())  
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

def averaging_test_scores():
    mean_rslts_dic={}
    mean_result={}
    proj_mean_result={}
    global tst_rslt_mstr_dic
    
    #initializing
    for smplng_strtgy_key in mstr_avg_tst_rslt_mstr_dic[0]:
        for proj_ky in mstr_avg_tst_rslt_mstr_dic[0][smplng_strtgy_key]:
            mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_prec-mean']=[]
            mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_rcl-mean']=[]
            mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_f1-mean']=[]
            mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_roc-auc-mean']=[]
            mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_spec-mean']=[]
            mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_sens-mean']=[]
            mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_gm-mean']=[]

    #populating
    for tst_rst_dict in mstr_avg_tst_rslt_mstr_dic:
        for smplng_strtgy_key in tst_rst_dict:
            for proj_ky in tst_rst_dict[smplng_strtgy_key]:
                if proj_ky == "overall_test":
                    continue
                mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_prec-mean'].append(tst_rst_dict[smplng_strtgy_key][proj_ky]["prec"])
                mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_rcl-mean'].append(tst_rst_dict[smplng_strtgy_key][proj_ky]["rcl"])
                mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_f1-mean'].append(tst_rst_dict[smplng_strtgy_key][proj_ky]["f1"])
                mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_roc-auc-mean'].append(tst_rst_dict[smplng_strtgy_key][proj_ky]["roc_auc"])
                mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_spec-mean'].append(tst_rst_dict[smplng_strtgy_key][proj_ky]["spec"])
                mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_sens-mean'].append(tst_rst_dict[smplng_strtgy_key][proj_ky]["sens"])
                mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_gm-mean'].append(tst_rst_dict[smplng_strtgy_key][proj_ky]["gm"])


    #storing in mstr dictionary for future use 
    for smplng_strtgy_key in tst_rst_dict:
        for proj_ky in tst_rst_dict[smplng_strtgy_key]:
            if proj_ky == "overall_test":
                continue
            mean_result['prec']=round(mean(mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_prec-mean']),3)
            logging.info("Listing all test results for Precision: %s",mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_prec-mean'])
            mean_result['rcl']=round(mean(mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_rcl-mean']),3)
            logging.info("Listing all test results for Recall: %s",mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_rcl-mean'])
            mean_result['f1']=round(mean(mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_f1-mean']),3)
            logging.info("Listing all test results for F1: %s",mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_f1-mean'])
            mean_result['roc_auc']=round(mean(mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_roc-auc-mean']),3)
            logging.info("Listing all test results for roc_auc: %s",mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_roc-auc-mean'])
            mean_result['spec']=round(mean(mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_spec-mean']),3)
            logging.info("Listing all test results for Specificity: %s",mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_spec-mean'])
            mean_result['sens']=round(mean(mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_sens-mean']),3)
            logging.info("Listing all test results for Sensitivity: %s",mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_sens-mean'])
            mean_result['gm']=round(mean(mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_gm-mean']),3)
            logging.info("Listing all test results for geometric mean: %s",mean_rslts_dic[smplng_strtgy_key+'_'+proj_ky+'_gm-mean'])
            proj_mean_result[proj_ky]=mean_result
            mean_result={}
        tst_rslt_mstr_dic[smplng_strtgy_key]=proj_mean_result
        proj_mean_result={}
        
        
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


def ml_model(iteration):
    
    logging.info("%s Modeling begins",model_name)

    cv_prec_lst=[]
    cv_rcl_lst=[]
    cv_f1_lst=[]
    cv_roc_auc_lst=[]
    cv_spec_lst=[]
    cv_sens_lst=[]
    cv_gm_lst=[]
    n_folds=10
    kfold=StratifiedKFold(n_splits=n_folds, shuffle=True,random_state =iteration+5 )
    epochs=20
    batch_size=32

    global val_conf_mat_tp,cross_val_prec_list,cross_val_rcl_list,cross_val_f1_list,cross_val_rocauc_list,cv_rslt_dic,mstr_tst_df
    global cross_val_sens_list,cross_val_spec_list,cross_val_gm_list,model_history,best_model_ind,tst_rslt_dic,tst_rslt_mstr_dic,scale_positive_weight
    global bal_tech

    for i in sampling_strategy_lst:
        for ovrsmplng_algo in (sampling_algo_lst):
            for proj_df in df_dct:  #0,0.5,0.6,0.7,0.8,0.9,1.0
                logging.info("Model Fitting for %s sampling strategy %s begins %s",ovrsmplng_algo,i,iteration)
                
                X_train_data=processed_dct_x[proj_df]['train'][ovrsmplng_algo][i]
                Y_train_data=processed_dct_y[proj_df]['train'][ovrsmplng_algo][i]
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
                    if "weighted" == ovrsmplng_algo:
                        bal_tech="weighted"
                        scale_pos_cntr=Counter(Y_train)

                        scale_positive_weight=int(scale_pos_cntr[0]/scale_pos_cntr[1])
                        logging.info("scale_pos_cntr 0,1,wgt: %s,%s,%s ",scale_pos_cntr[0],scale_pos_cntr[1],scale_positive_weight)
                        logging.info("scale_positive_weight for %s project : %s ",proj_df,scale_positive_weight)
                        logging.info("ovrsmplng_algo : %s ",ovrsmplng_algo)
                    else:
                        bal_tech="other"

                    model_history.append(model_fitting(model_name,X_train, X_test, Y_train, Y_test, epochs, batch_size))
                logging.info("Model Fitting for %s sampling strategy %s ends %s",ovrsmplng_algo,i,iteration)
                logging.info("=====================================================================")
                logging.info("%s - Key Observations for %s with %s sampling strategy %s  ",iteration,proj_df,ovrsmplng_algo,i)
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
        
                cv_prec_lst.append(round(mean(cross_val_prec_list),3))
                cv_rcl_lst.append(round(mean(cross_val_rcl_list),3))
                cv_roc_auc_lst.append(round(mean(cross_val_rocauc_list),3))
                cv_f1_lst.append(round(mean(cross_val_f1_list),3))
                cv_spec_lst.append(round(mean(cross_val_spec_list),3))
                cv_sens_lst.append(round(mean(cross_val_sens_list),3))
                cv_gm_lst.append(round(mean(cross_val_gm_list),3))
                logging.info("cv_prec_lst: %s,%s",cv_prec_lst,round(mean(cross_val_prec_list),3))

                logging.info("********* %s test data*********",proj_df)
                test_stats(model_history[best_model_ind],processed_dct_x[proj_df]['test'],processed_dct_y[proj_df]['test'],ovrsmplng_algo+"_"+str(i).replace(".","_"),proj_df,processed_dct_cmnt[proj_df]['test'])

                
                #logging.info("%s - Saving the model for sampling strategy %s... at  %s ",ovrsmplng_algo, i, model_file_path)    
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

            mstr_tst_df.to_csv(output_path+model_name+'_'+ovrsmplng_algo+'_'+str(i)+'_iter'+str(iteration)+".csv")
            mstr_tst_df=pd.DataFrame()   

            tst_rslt_mstr_dic[ovrsmplng_algo+"_"+str(i).replace(".","_")]=tst_rslt_dic
            mstr_avg_tst_rslt_mstr_dic.append(tst_rslt_mstr_dic)
            tst_rslt_dic={}
            cv_rslt_dic[ovrsmplng_algo]={'prec':cv_prec_lst,'rcl':cv_rcl_lst,'f1':cv_f1_lst,'roc_auc':cv_roc_auc_lst,'spec':cv_spec_lst,'sens':cv_sens_lst,'gm':cv_gm_lst}
            logging.info("oversampling algo,cv_prec_lst: %s,%s",ovrsmplng_algo,cv_prec_lst)
            cv_prec_lst=[]
            cv_rcl_lst=[]
            cv_f1_lst=[]
            cv_roc_auc_lst=[]
            cv_spec_lst=[]
            cv_sens_lst=[]
            cv_gm_lst=[]

		
    logging.info("%s Modeling ends",model_name)

def chart_values(rslt_dict):
  for smplng_algo in rslt_dict:
    logging.info("%s",smplng_algo)
    for val in rslt_dict[smplng_algo]:
      logging.info("%s : (%s)",val,rslt_dict[smplng_algo][val])	
    logging.info("----------------------------------------------------------------")

		
def test_stats(model,test_inp,ground_truth_inp,smplng_strtgy_ky,tst_proj_ky,tst_df_cmnts):
    rslt={}
    tst_df=pd.DataFrame()
    global tst_rslt_dic,mstr_tst_df
    event_time=time.time()
    y_class = model.predict(test_inp)
    y_pred_prob =  model.predict_proba(test_inp)[:,1]
    logging.info("-inference time- %s seconds ---" % (time.time() - event_time))
    logging.info(classification_report(ground_truth_inp, y_class)) 
    logging.info(precision_recall_fscore_support(ground_truth_inp,y_class,average='binary'))
    logging.info(confusion_matrix(ground_truth_inp, y_class))
	
    rslt["prec"]=round(precision_score(ground_truth_inp, y_class, average='binary'),3)
    rslt["rcl"]=round(recall_score(ground_truth_inp, y_class, average='binary'),3)
    rslt["f1"]=round(f1_score(ground_truth_inp, y_class,average='binary'),3)
    rslt["roc_auc"]=round(roc_auc_score(ground_truth_inp,y_class),3)
    rslt["spec"]=round(specificity_score(ground_truth_inp,y_class,average='binary'),3)
    rslt["sens"]=round(sensitivity_score(ground_truth_inp,y_class,average='binary'),3)
    rslt["gm"]=round(geometric_mean_score(ground_truth_inp,y_class,average='binary'),3)

     	
    tst_rslt_dic[tst_proj_ky]=rslt
    tst_df_cmnts=tst_df_cmnts
    logging.info("type ground_truth_inp: %s,%s",type(ground_truth_inp),ground_truth_inp.shape)
    logging.info("type y_class: %s,%s",type(y_class),y_class.shape)
    logging.info("type y_pred_prob: %s,%s",type(y_pred_prob),y_pred_prob.shape)    
    logging.info("type tst_df_cmnts: %s,%s",type(tst_df_cmnts),tst_df_cmnts.shape)

    if tst_proj_ky != 'overall_test':
        logging.info(classification_report(ground_truth_inp, y_class)) 
        data={'Project':ground_truth_inp,'Comments':tst_df_cmnts,'Ground_Truth':ground_truth_inp,'Predicted_Class':y_class,'Predicted_Probability':y_pred_prob}
        tst_df=pd.DataFrame(data)
        tst_df['Project'] = tst_proj_ky
        logging.info("tst_df shape: %s",tst_df.shape)
        logging.info("tst_df : %s",tst_df)
        mstr_tst_df=mstr_tst_df.append(tst_df,ignore_index=True)    
        logging.info("mstr_tst_df shape: %s",mstr_tst_df.shape)

    
    logging.info("%s : %s : Precision: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["prec"])
    logging.info("%s : %s : Recall: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["rcl"])
    logging.info("%s : %s : F1: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["f1"])
    logging.info("%s : %s : ROC-AUC: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["roc_auc"])
    logging.info("%s : %s : Specificity: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["spec"])
    logging.info("%s : %s : Sensitivity: %s ",smplng_strtgy_ky,tst_proj_ky,rslt["sens"])
    logging.info("%s : %s : Geometric Mean: %s",smplng_strtgy_ky,tst_proj_ky,rslt["gm"])

	
def model_fitting(model_name,train_x, val_x, train_y, val_y, EPOCHS=20, BATCH_SIZE=32):
  
  global model_history,best_model_ind,val_conf_mat_tp,cross_val_prec_list,cross_val_rcl_list,scale_positive_weight
  global cross_val_f1_list,cross_val_rocauc_list,cross_val_sens_list,cross_val_spec_list,cross_val_gm_list
  global bal_tech
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


def proc_train_tst_split(df):
    x_train_data_splt, x_test_data_splt, y_train_data_splt, y_test_data_splt = train_test_split(df['commenttext'], df['tag'], test_size=0.1,random_state=45)
    df_train = pd.concat([x_train_data_splt, y_train_data_splt],axis=1)
    df_test = pd.concat([x_test_data_splt, y_test_data_splt],axis=1)
    return df_train,df_test

########main##########
output_path='/scratch/project_2002565/bal_td/within_rslt_tables/'
log_path='/scratch/project_2002565/bal_td/within_log/'
model_path="/scratch/project_2002565/bal_td/chk_pt/"
encoder_path="/scratch/project_2002565/bal_td/encoder/"
input_file='/scratch/project_2002565/bal_td/data/technical_debt_dataset.csv'

model_name=sys.argv[1]
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

bal_tech="other"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO,datefmt='%a, %d %b %Y %H:%M:%S', filename=log_path+model_name+'_within.log', filemode='w')
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
sampling_algo_lst=["baseline","weighted","smote","adasyn","svmsmote","bline"]
tst_rslt_mstr_dic={}
tst_rslt_dic={}
cv_rslt_dic={}
mean_tst_rslts_mstr_dic={'baseline':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'weighted':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'smote':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'adasyn':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'bline':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]},'svmsmote':{'prec':[],'rcl':[],'f1':[],'roc-auc':[]}}
cv_rslt={}
df_dct={}
mstr_avg_tst_rslt_mstr_dic=[]
smplng_dict_x={}
smplng_dict_y={}
sampling_dict_x={}
sampling_dict_y={}
processed_dct_x={}
processed_dct_y={}
scale_positive_weight=0
mstr_y_imbalance_dict={}
smplng_dict_cmnt={}
processed_dct_cmnt={}
mstr_tst_df=pd.DataFrame()
sampling_strategy_lst=[1.0]
    

nltk.download('stopwords')
stop = stopwords.words('english')

tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)



df_dct['ant']=ant_df
df_dct['argouml']=argouml_df
df_dct['columba']=columba_df
df_dct['emf']=emf_df
df_dct['hibernate']=hibernate_df
df_dct['jedit']=jedit_df
df_dct['jfreechart']=jfreechart_df
df_dct['jmeter']=jmeter_df
df_dct['jruby']=jruby_df
df_dct['sql']=sql_df

for data_frame in df_dct:
    df_train,df_test=proc_train_tst_split(df_dct[data_frame])
    
    data_clean = clean_text(df_train, 'commenttext', 'commenttext')
    data_clean['commenttext'] = data_clean['commenttext'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    X_train_processed=data_clean['commenttext']
    Y_train_processed=data_clean['tag']

    data_clean = clean_text(df_test, 'commenttext', 'commenttext')
    data_clean['commenttext'] = data_clean['commenttext'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    X_test_processed=data_clean['commenttext']
    Y_test_processed=data_clean['tag']

    tfidf_vectorizer=tfidf_vectorizer.fit(X_train_processed)
    X_train_processed=tfidf_vectorizer.transform(X_train_processed)
    fnames_bs=tfidf_vectorizer.get_feature_names()

    X_test_processed=tfidf_vectorizer.transform(X_test_processed)
    fnames_bs=tfidf_vectorizer.get_feature_names()

    logging.info("Oversampling sample creation begins")
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

                
            if(sampling_algo == "baseline" or sampling_algo == "weighted"):
                X_train_sampling_data, Y_train_sampling_data = X_train_processed, Y_train_processed
            else:
                X_train_sampling_data, Y_train_sampling_data = sampling_obj.fit_resample(X_train_processed, Y_train_processed)
            sample_dict_x[x]=X_train_sampling_data
            sample_dict_y[x]=Y_train_sampling_data
            count_data=Counter(Y_train_sampling_data)
            mstr_y_imbalance_dict[data_frame]=Counter(Y_train_sampling_data)
            logging.info("******************************")
            logging.info("Before sampling Y_train_data shape for %s : %s",data_frame,Counter(Y_train_processed))
            logging.info("%s ~ Sampling Strategy ~ %s",sampling_algo,x)
            logging.info("X_train_sampling_data shape: %s",X_train_sampling_data.shape) 
            logging.info("Y_train_sampling_data shape: %s",Y_train_sampling_data.shape)
            logging.info("Sampling Split: %s",count_data)  
            logging.info("Sample created with %s Oversampling algorithm with sampling strategy %s",x,sampling_algo)
            logging.info("-------------------------------------------")  
            logging.info("Test size for %s: %s",data_frame,Counter(Y_test_processed))            
            logging.info("******************************")
        sampling_dict_x[sampling_algo]=sample_dict_x
        sampling_dict_y[sampling_algo]=sample_dict_y
        sample_dict_x={}
        sample_dict_y={}
    smplng_dict_x['train']=sampling_dict_x
    smplng_dict_y['train']=sampling_dict_y
    smplng_dict_x['test']=X_test_processed
    smplng_dict_y['test']=Y_test_processed
    smplng_dict_cmnt['test']=df_test['commenttext']
    processed_dct_x[data_frame]=smplng_dict_x
    processed_dct_y[data_frame]=smplng_dict_y
    processed_dct_cmnt[data_frame]=smplng_dict_cmnt
    sampling_dict_x={}
    sampling_dict_y={}
    smplng_dict_x={}
    smplng_dict_y={}
    smplng_dict_cmnt={}


try:
    for iteration in range(10):
        ml_model(iteration)
    generate_rslt_tables()
    logging.info("*********** TST ***************")
    chart_values(mean_tst_rslts_mstr_dic)    
except Exception as details:
    logging.info('Unexpected error: %s',details)
    print('Unexpected error: {0}'.format(details))
    os._exit(0)

