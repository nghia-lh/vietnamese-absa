import pytest
from src.data import parse_span,attach_overall_by_exact_text
from src.segmentation import word_labels

def test_span():
    # Kiểm tra lấy đúng cụm khía cạnh từ review
    text = "pin yếu"

    span = parse_span([0, 3, "BATTERY#NEGATIVE"], text)

    assert span["text"] == "pin"
    assert span["category"] == "BATTERY"
    assert span["polarity"] == "negative"
    
def test_star_join_ambiguity():
    rows={'train':[{'text':'pin yếu','spans':[]}],'dev':[],'test':[]}
    matched,stats=attach_overall_by_exact_text(rows,[('pin yếu',1),('pin yếu',5)])
    assert matched['train']==[]
    assert stats['train_unmatched_or_conflicting']==1

def test_bio():
    text = "pin yếu nhưng"
    words = [('pin', 0, 3), ('yếu', 4, 7), ('nhưng', 8, 13)]
    spans = [{'start': 0, 'end': 7, 'category': 'BATTERY', 'polarity': 'negative'}]

    assert word_labels(words, spans, text) == [
        'B-BATTERY#NEGATIVE',
        'I-BATTERY#NEGATIVE',
        'O'
    ]