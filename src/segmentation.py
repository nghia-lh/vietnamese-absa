"""VnCoreNLP word-segmentation with raw-text character alignment.
Rejects ambiguous alignments instead of manufacturing BIO annotations.
"""
import re

from pathlib import Path

class Segmenter:
    def __init__(self, model_dir):
        import py_vncorenlp

        # Dùng đường dẫn tuyệt đối để Java tìm đúng file JAR
        model_dir = Path(model_dir).resolve()

        self.engine = py_vncorenlp.VnCoreNLP(
            annotators=['wseg'],
            save_dir=str(model_dir)
        )
        
    def aligned_words(self, text):
        sentences = self.engine.word_segment(text)
        tokens = [w for sentence in sentences for w in sentence.split()]
        words = []
        cursor = 0
        for token in tokens:
            # VnCoreNLP joins syllables inside a word with underscores.
            pieces = token.split('_')
            regex = r'\s+'.join(re.escape(p) for p in pieces)
            found = re.search(regex, text[cursor:])
            if found is None:
                raise ValueError(f'Segmentation offset alignment failed: {token!r} in {text!r}')
            start, end = cursor + found.start(), cursor + found.end()
            while start < end and text[start].isspace(): start += 1
            words.append((token, start, end))
            cursor = end
        return words


def word_labels(words, spans):
    tags = []
    for _, start, end in words:
        overlaps = [s for s in spans if start < s['end'] and end > s['start']]
        if len(overlaps) > 1:
            raise ValueError('Segmenter word crosses multiple aspects')
        if not overlaps:
            tags.append('O'); continue
        span = overlaps[0]
        if start < span['start'] or end > span['end']:
            raise ValueError('Word segmentation crosses annotated aspect boundary')
        prefix = 'B' if start == span['start'] else 'I'
        tags.append(f"{prefix}-{span['category']}#{span['polarity'].upper()}")
    for span in spans:
        selected = [(start, end) for _, start, end in words if start >= span['start'] and end <= span['end']]
        if not selected or selected[0][0] != span['start'] or selected[-1][1] != span['end']:
            raise ValueError(f"Span not exactly covered by aligned words: {span}")
    return tags
