"""Descriptive length strata for saved PhoBERT Aspect predictions (no retraining)."""
import argparse
import json
from pathlib import Path
from src.data import load_jsonl
from src.evaluate_aspects import score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gold',default='data/processed_v2/test.jsonl')
    p.add_argument('--predictions',default='outputs/evaluation/aspect/predictions.jsonl')
    p.add_argument('--threshold',type=int,default=40)
    p.add_argument('--out',default='outputs/aspect_nb_baseline/phobert_length_diagnostic.json')
    args=p.parse_args()
    g=load_jsonl(args.gold);pr=load_jsonl(args.predictions)
    assert len(g)==len(pr)
    assert all(x['text']==y['text'] for x,y in zip(g,pr))
    result={'threshold_segmented_words':args.threshold,'full_n':len(g),'strata':{}}
    for name,criterion in [('short',lambda n:n<=args.threshold),('long',lambda n:n>args.threshold)]:
        indices=[i for i,row in enumerate(g) if criterion(len(row['words']))]
        gr=[g[i] for i in indices]; pe=[pr[i] for i in indices]
        result['strata'][name]={'reviews':len(gr),'annotated_spans':sum(len(x['spans']) for x in gr),
                                'span':score(gr,pe,False),'pair':score(gr,pe,True)}
        print(name,'reviews',len(gr),'pair F1',result['strata'][name]['pair']['f1'])
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('Saved:',path)

if __name__=='__main__':
    main()
