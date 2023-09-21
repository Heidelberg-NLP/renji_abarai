#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier


val_df_emb = pd.read_csv('../../data/train-val-splits/qrels_args_docs_val_emb_predictions.tsv', sep='\t')
val_df_fea = pd.read_csv('../../data/train-val-splits/qrels_args_docs_val_features_predictions.tsv', sep='\t')

merged_df = pd.merge(val_df_emb, val_df_fea, on=['q_id', 'doc_id'], how='left').drop_duplicates()

prob_cols = ['dnn_prob_x', 'lr_prob_x', 'svm_prob_x', 'rf_prob_x',
             'nb_prob_x', 'lgb_prob_x',
             'dnn_prob_y', 'lr_prob_y', 'svm_prob_y', 'rf_prob_y',
             'nb_prob_y', 'lgb_prob_y']

merged_df['dnn_prob_x'] = merged_df['dnn_prob_x'].apply(ast.literal_eval)
merged_df['dnn_prob_y'] = merged_df['dnn_prob_y'].apply(ast.literal_eval)
merged_df['lr_prob_x'] = merged_df['lr_prob_x'].apply(ast.literal_eval)
merged_df['lr_prob_y'] = merged_df['lr_prob_y'].apply(ast.literal_eval)
merged_df['svm_prob_x'] = merged_df['svm_prob_x'].apply(ast.literal_eval)
merged_df['svm_prob_y'] = merged_df['svm_prob_y'].apply(ast.literal_eval)
merged_df['rf_prob_x'] = merged_df['rf_prob_x'].apply(ast.literal_eval)
merged_df['rf_prob_y'] = merged_df['rf_prob_y'].apply(ast.literal_eval)
merged_df['nb_prob_x'] = merged_df['nb_prob_x'].apply(ast.literal_eval)
merged_df['nb_prob_y'] = merged_df['nb_prob_y'].apply(ast.literal_eval)
merged_df['lgb_prob_x'] = merged_df['lgb_prob_x'].apply(ast.literal_eval)
merged_df['lgb_prob_y'] = merged_df['lgb_prob_y'].apply(ast.literal_eval)

merged_df['feature_vector'] = merged_df[prob_cols].agg(list, axis=1)

X_train = list()
for item in merged_df.feature_vector.tolist():
    X_train.append([x for sublist in item for x in sublist])

test_df_fea = pd.read_csv("../../results-test/quality/chatnoir_10_custom_stopw_lemmas_features_predictions.tsv",
                          sep="\t")
test_df_emb = pd.read_csv("../../results-test/quality/chatnoir_10_custom_stopw_lemmas_emb_predictions.tsv", sep="\t")

merged_df_test = pd.merge(test_df_emb, test_df_fea, on=['qid', 'docno'], how='left').drop_duplicates()

prob_cols = ['dnn_prob_x', 'lr_prob_x', 'svm_prob_x', 'rf_prob_x',
             'nb_prob_x', 'lgb_prob_x',
             'dnn_prob_y', 'lr_prob_y', 'svm_prob_y', 'rf_prob_y',
             'nb_prob_y', 'lgb_prob_y']

merged_df_test['dnn_prob_x'] = merged_df_test['dnn_prob_x'].apply(ast.literal_eval)
merged_df_test['dnn_prob_y'] = merged_df_test['dnn_prob_y'].apply(ast.literal_eval)
merged_df_test['lr_prob_x'] = merged_df_test['lr_prob_x'].apply(ast.literal_eval)
merged_df_test['lr_prob_y'] = merged_df_test['lr_prob_y'].apply(ast.literal_eval)
merged_df_test['svm_prob_x'] = merged_df_test['svm_prob_x'].apply(ast.literal_eval)
merged_df_test['svm_prob_y'] = merged_df_test['svm_prob_y'].apply(ast.literal_eval)
merged_df_test['rf_prob_x'] = merged_df_test['rf_prob_x'].apply(ast.literal_eval)
merged_df_test['rf_prob_y'] = merged_df_test['rf_prob_y'].apply(ast.literal_eval)
merged_df_test['nb_prob_x'] = merged_df_test['nb_prob_x'].apply(ast.literal_eval)
merged_df_test['nb_prob_y'] = merged_df_test['nb_prob_y'].apply(ast.literal_eval)
merged_df_test['lgb_prob_x'] = merged_df_test['lgb_prob_x'].apply(ast.literal_eval)
merged_df_test['lgb_prob_y'] = merged_df_test['lgb_prob_y'].apply(ast.literal_eval)

merged_df_test['feature_vector'] = merged_df_test[prob_cols].agg(list, axis=1)

X_test = list()
for item in merged_df_test.feature_vector.tolist():
    X_test.append([x for sublist in item for x in sublist])

clf = RandomForestClassifier(criterion='entropy', max_depth=5, max_features='auto', n_estimators=200)

clf.fit(X_train, merged_df.qual_x.tolist())
probs = clf.predict_proba(X_test)
predictions = clf.predict(X_test)

merged_df_test["meta_pred"] = predictions.tolist()
merged_df_test["meta_prob"] = probs.tolist()

cols_to_keep = ['qid', 'query_x', 'docno', 'score_x', 'title_text_x', 'html_plain_x',
       'rank_x', 'meta_pred', 'meta_prob']

merged_df_test = merged_df_test[cols_to_keep]
merged_df_test = merged_df_test.rename(columns={"query_x": "query", "score_x": "score", "title_text_x": "title_text", "html_plain_x": "html_plain", "rank_x": "rank"}, errors="raise")

merged_df_test.to_csv("../../results-test/quality/chatnoir_10_custom_stopw_lemmas_meta_predictions.tsv", sep="\t", index=False)