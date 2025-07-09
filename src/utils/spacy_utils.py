from spacy_model import nlp, stemmer


def force_noun_lemmatization(word: str) -> str:
    doc = nlp.make_doc(word.lower())
    for token in doc:
        token.pos_ = "NOUN"
    nlp.get_pipe("lemmatizer")(doc)
    return " ".join([token.lemma_.lower() for token in doc])

def lemmatize_text(text: str) -> set:
    return set(token.lemma_.lower() for token in nlp(text.lower()) if not token.is_punct and not token.is_space)

def stem_word(word: str) -> str:
    return stemmer.stem(word.lower())

def stem_sentence(sentence: list[str]) -> set:
    return {stem_word(word) for word in sentence.lower().split()}
