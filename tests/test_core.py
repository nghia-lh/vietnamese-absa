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
    
from src.segmentation import Segmenter


def test_underscore_alignment():
    # Giả lập kết quả tách từ để không cần khởi động Java
    class FakeEngine:
        def word_segment(self, text):
            return ["Pin còn 18 _ 19%"]

    seg = Segmenter.__new__(Segmenter)
    seg.engine = FakeEngine()

    text = "Pin còn 18_19%"
    words = seg.aligned_words(text)

    assert words == [
        ("Pin", 0, 3),
        ("còn", 4, 7),
        ("18", 8, 10),
        ("_", 10, 11),
        ("19%", 11, 14),
    ]

def test_vietnamese_tone_alignment():
    # Giả lập từ đã được VnCoreNLP tách
    class FakeEngine:
        def word_segment(self, text):
            return ["sức_khoẻ tốt"]

    seg = Segmenter.__new__(Segmenter)
    seg.engine = FakeEngine()

    text = "sức khỏe tốt"
    words = seg.aligned_words(text)

    assert words == [
        ("sức_khoẻ", 0, 8),
        ("tốt", 9, 12),
    ]
    
def test_emoji_alignment():
    # Giả lập kết quả VnCoreNLP trả về
    class FakeEngine:
        def word_segment(self, text):
            return ["Máy tốt \uf601"]

    seg = Segmenter.__new__(Segmenter)
    seg.engine = FakeEngine()

    text = "Máy tốt 😁"
    words = seg.aligned_words(text)

    assert words == [
        ("Máy", 0, 3),
        ("tốt", 4, 7),
        ("\uf601", 8, 9),
    ]
    
def test_emoji_inside_word_alignment():
    # Giả lập kết quả VnCoreNLP với emoji nằm giữa chữ
    class FakeEngine:
        def word_segment(self, text):
            return ["nữa\uf612máy"]

    seg = Segmenter.__new__(Segmenter)
    seg.engine = FakeEngine()

    text = "nữa😒máy"
    words = seg.aligned_words(text)

    assert words == [
        ("nữa\uf612máy", 0, 7),
    ]