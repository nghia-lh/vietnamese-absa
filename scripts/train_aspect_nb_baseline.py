"""Train reproducible classical Aspect BIO baselines on processed_v2.

No pretrained model. Each token has sparse lexical features; an optional version
uses neighboring word context. Dev selects NB alpha and whether class priors are
empirical. Test is evaluated once per dev-selected feature family, with exact
character-span and complete span/category/polarity metrics from src.evaluate_aspects.

Usage: python train_aspect_nb_baseline.py --data data/processed_v2 --out outputs/aspect_nb_baseline
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.data import load_jsonl
from src.decode import decode_bio
from src.evaluate_aspects import score


def token_features(words, i, contextual):
    w = words[i].lower()
    feats = {
        'word': w,
        'prefix2': w[:2], 'prefix3': w[:3],
        'suffix2': w[-2:], 'suffix3': w[-3:],
        'has_digit': str(any(ch.isdigit() for ch in w)),
        'punct_only': str(not any(ch.isalnum() for ch in w)),
        'length': str(min(len(w), 12)),
    }
    if contextual:
        for offset in [-2, -1, 1, 2]:
            neighbor = i + offset
            feats[f'neighbor{offset:+}'] = (words[neighbor].lower() if 0 <= neighbor < len(words) else '<BND>')
        feats['prev+word'] = feats['neighbor-1'] + '|' + w
        feats['word+next'] = w + '|' + feats['neighbor+1']
    return feats


def vectors(rows, vec, contextual, fit):
    feats = (token_features(row['words'], index, contextual)
             for row in rows for index in range(len(row['words'])))
    return vec.fit_transform(feats) if fit else vec.transform(feats)


def to_predictions(rows, flat_tags):
    predictions=[]
    cursor=0
    for row in rows:
        n=len(row['words'])
        assert n == len(row['tags']) == len(row['word_offsets'])
        pred_tags=flat_tags[cursor:cursor+n]
        assert len(pred_tags) == n
        cursor += n
        predictions.append({'text':row['text'], 'spans':decode_bio(row['words'],row['word_offsets'],pred_tags)})
    assert cursor==len(flat_tags)
    return predictions


def metrics(rows, predicted):
    return {'span':score(rows,predicted,pair=False),
            'pair':score(rows,predicted,pair=True),
            'reviews':len(rows)}


def dump_jsonl(path, rows):
    with path.open('w',encoding='utf8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def decode_with_o_bias(log_proba,classes,o_bias):
    """Tune an additive score offset for the non-span tag O only on development."""
    oi=list(classes).index('O')
    adjusted=np.array(log_proba,copy=True)
    adjusted[:,oi]+=o_bias
    return classes[np.argmax(adjusted,axis=1)]


def run(args):
    root=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows={split:load_jsonl(root/f'{split}.jsonl') for split in ('train','dev','test')}
    y_train=np.asarray([t for row in rows['train'] for t in row['tags']])
    results=[]
    biases = (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    for contextual in (False,True):
        family='local' if not contextual else 'context'
        vec=DictVectorizer(dtype=np.float32)
        Xtrain=vectors(rows['train'],vec,contextual,fit=True)
        Xdev=vectors(rows['dev'],vec,contextual,fit=False)
        candidates=[]
        best=None
        for alpha in (0.1,1.0):
            for fit_prior in (False,True):
                model=MultinomialNB(alpha=alpha,fit_prior=fit_prior)
                model.fit(Xtrain,y_train)
                logp=model.predict_log_proba(Xdev)
                for o_bias in biases:
                    guesses = decode_with_o_bias(logp,model.classes_,o_bias)
                    dev_pred=to_predictions(rows['dev'],guesses)
                    val=metrics(rows['dev'],dev_pred)
                    row={'alpha':alpha,'fit_prior':fit_prior,'o_bias':o_bias,'dev':val}
                    candidates.append(row)
                    if best is None or (val['pair']['f1'],val['span']['f1']) > (best['dev']['pair']['f1'],best['dev']['span']['f1']):
                        best=row
                print(f'{family}: alpha={alpha} fit_prior={fit_prior} current best DEV pair={best["dev"]["pair"]["f1"]:.6f}',flush=True)
        # Test evaluated once on the development-selected parameters for each pre-specified feature family.
        chosen=MultinomialNB(alpha=best['alpha'],fit_prior=best['fit_prior'])
        chosen.fit(Xtrain,y_train)
        Xtest=vectors(rows['test'],vec,contextual,fit=False)
        test_scores=chosen.predict_log_proba(Xtest)
        test_labels=decode_with_o_bias(test_scores,chosen.classes_,best['o_bias'])
        test_pred=to_predictions(rows['test'],test_labels)
        test=metrics(rows['test'],test_pred)
        dump_jsonl(out/f'{family}_test_predictions.jsonl',test_pred)
        results.append({'feature_family':family,'feature_count':Xtrain.shape[1],
            'selection_metric':'dev exact span+category+polarity pair F1',
            'candidates':candidates,
            'selected':{'alpha':best['alpha'],'fit_prior':best['fit_prior'],'o_bias':best['o_bias'],'dev':best['dev']},
            'test':test})
        print(f'{family}: selected alpha={best["alpha"]}, fit_prior={best["fit_prior"]}, o_bias={best["o_bias"]}; TEST pair={test["pair"]["f1"]:.6f} span={test["span"]["f1"]:.6f}',flush=True)
        del Xtrain, Xdev, Xtest,vec
    report={'experiment':'MultinomialNB BIO lexical baseline; local-vs-context feature ablation',
            'train_reviews':len(rows['train']),'dev_reviews':len(rows['dev']),
            'test_reviews':len(rows['test']),'training_split_only_for_vocabulary':True,
            'test_used_for_model_selection':False,
            'o_bias_tuned_on_development_only':True,'results':results}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('Saved metrics:',out/'metrics.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed_v2')
    p.add_argument('--out',default='outputs/aspect_nb_baseline')
    run(p.parse_args())
