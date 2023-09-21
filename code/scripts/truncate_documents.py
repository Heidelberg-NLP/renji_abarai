import pandas as pd
from pathlib import Path
import random
from functools import partial

"""
This script truncates documents. This can be used if the documents (possibly with prompts) would exceed the maximum length of a model (for example ChatGPT).
If possible, documents are truncated at paragraph boundaries. If this is not possible, the document just is truncated at a fixed length.
"""

def load_data(fn:Path) -> pd.DataFrame:
    """
    :param fn: filename
    """
    df = pd.read_csv(fn, sep="\t")
    return df

def truncate_document(text:str, max_len:int=2500*4) -> str:
    """
    :param text: text to truncate
    :param max_len: maximum length of the text in chars
    :return: truncated text
    """
    assert isinstance(text, str), text
    if len(text) <= max_len:
        return text
    # split text by paragraphs
    paragraphs = text.split("\n\n")
    # select the first paragraphs that are less than max_len
    len_paragraphs = [len(p) + 2 for p in paragraphs]  # +2 for the \n\n
    len_paragraphs = [sum(len_paragraphs[:i+1]) for i in range(len(len_paragraphs))]  # cumulative sum
    valid_paragraphs = [p for p, l in zip(paragraphs, len_paragraphs) if l <= max_len]
    if len(valid_paragraphs) == 0:
        return text[:max_len]  # return the first max_len chars
    # join paragraphs
    text = "\n\n".join(valid_paragraphs)  # join paragraphs
    assert len(text) <= max_len, (text, len(text), max_len)
    assert len(text) >= 0, (text, len(text), max_len)
    return text

def save_data(df:pd.DataFrame, fn:Path):
    """
    :param df: dataframe to save
    :param fn: filename
    :return: None
    """
    df.to_csv(fn, sep="\t", index=False)

def main():
    # load data
    fn = Path("results-test/corpus/chatnoir_10_custom_stopw_lemmas.tsv")
    # fn = Path("results-test/corpus/chatnoir_10.tsv")
    df = load_data(fn)

    # truncate data
    df['truncated_html_plain'] = df['html_plain'].apply(truncate_document)

    # save data
    save_data(df, fn)

if __name__ == "__main__":
    main()
