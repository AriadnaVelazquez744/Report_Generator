"""
Analizador gramatical basado en gramáticas libres de contexto (CFG)
usando chunking de NLTK sobre etiquetas POS.
"""
import logging
from typing import Dict, List
from collections import Counter

import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize, RegexpParser
from nltk.tree import Tree

logger = logging.getLogger(__name__)

# Asegurar recursos necesarios
for resource, downloader in [
    ("tokenizers/punkt", "punkt"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(downloader, quiet=True)


class GrammarAnalyzer:
    """
    Construye una aproximación de análisis gramatical estilo CFG mediante
    chunking de frases nominales, verbales y preposicionales.
    """

    DEFAULT_GRAMMAR = r"""
        CLAUSE: {<NP><VP>}
        NP: {<DT|PP\$|PRP\$>?<JJ.*>*<NN.*>+}
        VP: {<VB.*><NP|PP|CLAUSE|PRP|JJ.*|RB.*>*}
        PP: {<IN><NP>}
        ADJP: {<RB.*>*<JJ.*>}
    """

    def __init__(self, grammar: str | None = None):
        rule_set = grammar or self.DEFAULT_GRAMMAR
        self.chunker = RegexpParser(rule_set)

    def analyze(self, text: str, language: str = "spanish") -> Dict:
        """
        Analiza el texto y retorna estadísticas gramaticales basadas en CFG.

        Args:
            text: Texto a analizar.
            language: Idioma para tokenización (afecta word_tokenize/sent_tokenize).

        Returns:
            Diccionario con conteos de chunks, reglas más frecuentes y ejemplos.
        """
        if not text or not isinstance(text, str):
            return self._empty_result()

        try:
            sentences = sent_tokenize(text, language=language)
        except LookupError:
            nltk.download("punkt", quiet=True)
            sentences = sent_tokenize(text, language=language)

        chunk_counts: Counter[str] = Counter()
        rule_counts: Counter[str] = Counter()
        chunk_examples: List[Dict[str, str]] = []
        clause_structures: List[str] = []

        for sentence in sentences:
            tokens = word_tokenize(sentence, language=language)
            if not tokens:
                continue

            try:
                tagged = pos_tag(tokens)
            except LookupError:
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
                tagged = pos_tag(tokens)

            tree: Tree = self.chunker.parse(tagged)
            clause_structures.append(tree.pformat(margin=70))

            for production in tree.productions():
                lhs = production.lhs()
                rhs = " ".join(str(sym) for sym in production.rhs())
                rule_counts[f"{lhs} -> {rhs}"] += 1
                
            for subtree in tree.subtrees():
                label = subtree.label()
                if label in {"NP", "VP", "PP", "ADJP", "CLAUSE"}:
                    chunk_counts[label] += 1
                    if len(chunk_examples) < 15:
                        chunk_examples.append(
                            {
                                "label": label,
                                "text": " ".join(word for word, _ in subtree.leaves()),
                            }
                        )

        total_rules = sum(rule_counts.values()) or 1
        top_rules = [
            {"rule": rule, "frequency": count, "weight": round(count / total_rules, 4)}
            for rule, count in rule_counts.most_common(10)
        ]

        return {
            "sentence_count": len(sentences),
            "chunk_counts": dict(chunk_counts),
            "top_rules": top_rules,
            "sample_chunks": chunk_examples,
            "clause_structures": clause_structures[:5],
        }

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "sentence_count": 0,
            "chunk_counts": {},
            "top_rules": [],
            "sample_chunks": [],
            "clause_structures": [],
        }


