import numpy as np
import pandas as pd
from googletrans import Translator
from dask import bag, diagnostics


def translate(words, dest):
    """Translates words to a language that is NOT english.
    If none is provided, a random language is chosen"""
    dest_choices = [
        "zh-cn",
        "ar",
        "fr",
        "sw",
        "ur",
        "vi",
        "ru",
        "hi",
        "el",
        "th",
        "es",
        "de",
        "tr",
        "bg",
    ]
    if not dest:
        dest = np.random.choice(dest_choices)

    translator = Translator()
    decoded = translator.translate(words, dest=dest).text
    return decoded


def trans_parallel(df, dest):
    """Gets a df as input. Translates premises and hypo using translate().
    Dask package optimizes the translation with bag objects, that make
    parallelization easy"""

    # create dask.bag objects
    premise_bag = bag.from_sequence(df.premise.tolist()).map(translate, dest)
    hypo_bag = bag.from_sequence(df.hypothesis.tolist()).map(translate, dest)

    with diagnostics.ProgressBar():
        # compute => perform the translations
        premises = premise_bag.compute()
        hypos = hypo_bag.compute()

    # replace "premise", "hypothesis" in the df with translated texts
    df[["premise", "hypothesis"]] = list(zip(premises, hypos))
    return df


if __name__ == "__main__":
    words_test_en = "This is a very English sentence"
    words_test_fr = "Cette phrase est une belle phrase en français"

    print("Test translate EN, dest EN ==========")
    print(translate(words_test_en, "en"))
    print(" ")
    print("Test translate FR, dest EN ==========")
    print(translate(words_test_fr, "en"))
    print(" ")
