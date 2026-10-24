"""Controlled aspect MNB removal of context features.
Uses the dev-selected contextual reference hyperparameters
alpha=0.1, empirical class priors, O score offset=4.0 for BOTH feature sets.
Reports dev and test strict character-offset metrics. No test-set tuning.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.data import load_jsonl
from .train_aspect_nb_baseline import vectors,decode_with_o_bias,to_predictions,metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed_v2')
    p.add_argument('--out',default='outputs/aspect_nb_baseline/controlled_context_ablation.json')
    args=p.parse_args()
    root=Path(args.data)
    rows={x:load_jsonl(root/f'{x}.jsonl') for x in ['train','dev','test']}
    y=np.asarray([label for row in rows['train'] for label in row['tags']])
    result={'fixed_hyperparameters':{'alpha':0.1,'fit_prior':True,'o_bias':4.0},'data':args.data,
            'purpose':'Remove context features with fixed selected model settings','results':{}}
    for context in (False,True):
        v=DictVectorizer(dtype=np.float32)
        xtrain=vectors(rows['train'],v,context,fit=True)
        model=MultinomialNB(alpha=0.1,fit_prior=True).fit(xtrain,y)
        feature='context' if context else 'local_only'
        result['results'][feature]={}
        for name in ('dev','test'):
            x=vectors(rows[name],v,context,fit=False)
            pred_labels=decode_with_o_bias(model.predict_log_proba(x),model.classes_,4.0)
            pred=to_predictions(rows[name],pred_labels)
            result['results'][feature][name]=metrics(rows[name],pred)
            print(feature,name,'pair_f1',result['results'][feature][name]['pair']['f1'])
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('Saved:',path)

if __name__=='__main__':
    main()
