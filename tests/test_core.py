import pytest
from src.data import parse_span
from src.overall import majority_label,build_overall
from src.decode import decode_bio
from src.evaluate_aspects import score
from src.audit import audit

def test_offsets_exclusive():
    # Kiểm tra vị trí kết thúc không tính ký tự cuối
    span = parse_span([0, 3, "BATTERY#NEGATIVE"], "pin yếu")

    assert span["text"] == "pin"
    assert span["start"] == 0
    assert span["end"] == 3

def test_majority_and_tie():
    spans=[{'polarity':'positive'},{'polarity':'negative'}]
    assert majority_label(spans) is None
    assert majority_label(spans+[{'polarity':'negative'}])=='negative'

def test_majority_excludes_ties():
    raw={'train':[{'text':'x','spans':[{'polarity':'positive'},{'polarity':'negative'}]}], 'dev':[], 'test':[]}
    out,stats=build_overall(raw,strategy='majority_only')
    assert not out['train'] and stats['train']['excluded_unlabeled_or_tied']==1

def test_decode_half_open():
    got=decode_bio(['pin','yếu','camera'],[(0,3),(4,7),(8,14)],['B-BATTERY#NEGATIVE','I-BATTERY#NEGATIVE','O'])
    assert got==[{'start':0,'end':7,'category':'BATTERY','polarity':'negative'}]

def test_exact_score():
    gold=[{'spans':[{'start':0,'end':3,'category':'BATTERY','polarity':'negative'}]}]
    assert score(gold,gold,pair=True)['f1']==1.0

def test_overlap_detection():
    d={'train':[{'text':'Repeat','spans':[]}],'dev':[{'text':'repeat','spans':[]}],'test':[]}
    assert audit(d)['cross_split_duplicate_text_count']==1
