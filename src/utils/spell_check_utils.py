import re
import string
import pkg_resources
from symspellpy import SymSpell
from nltk.metrics.distance import edit_distance
import string
import json
from collections import defaultdict
import os


def create_common_abbrev():
    return set(
        ["dr.","est.","i.e.","jr.","inc.","ltd.",
        "mr.","mrs.","ms.","oz.","sr.","vs.","e.g."
    ])


def create_worddict():
    sym_spell = SymSpell()
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    abbrevs = [depunctuate(a) for a in create_common_abbrev()]
    worddict = sym_spell.words
    for a in abbrevs:
        if a in worddict:
            del worddict[a]

    return worddict


def create_homoglyph_dict(sensitivity = 0.35, homoglyph_fp = './homoglyph_list.json'):
    with open(homoglyph_fp, 'r') as infile:
        homoglyph_list = json.load(infile)

    i = 0
    homoglyph_dict = defaultdict(list)
    while homoglyph_list[i]['sim_score'] >= sensitivity:
        a, b = homoglyph_list[i]['a'], homoglyph_list[i]['b']
        homoglyph_dict[a].append(b)
        homoglyph_dict[b].append(a)
        i += 1
    return homoglyph_dict


def create_distinct_lowercase():
    return list("aenr")


def create_nondistinct_lowercase():
    return list("wuosvcxz")


def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def string_cleaner(s):
    return (s
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("''", "\"")
        .replace("‘‘", "\"")
        .replace("’’", "\""))


def isnt_cap(s):
    return s.islower() or s in string.punctuation


def all_caps(s):
    return all(c.isupper() for c in s)


def safe_index_is_alpha(s, i):
    max_idx, min_idx = len(s) - 1, 0
    if i > max_idx: return True
    elif i < min_idx: return True
    else: return s[i].isalpha()


def safe_index_is_digit(s, i):
    max_idx, min_idx = len(s) - 1, 0
    if i > max_idx: return True
    elif i < min_idx: return True
    else: return s[i].isdigit()


def majority_normalize(s, simdict):

    num_digits = sum(1 for c in s if c.isdigit())
    num_alphas = sum(1 for c in s if c.isalpha())
    outs = ""

    if num_alphas > num_digits:
        for i in range(len(s)):
            if s[i].isdigit() and safe_index_is_alpha(s, i-1) and safe_index_is_alpha(s, i+1) and s[i] in simdict:
                outs += simdict[s[i]][0]
            else:
                outs += s[i]
    elif num_digits > num_alphas:
        for i in range(len(s)):
            if s[i].isalpha() and safe_index_is_digit(s, i-1) and safe_index_is_digit(s, i+1) and s[i] in simdict:
                try:
                    outs += [x for x in simdict[s[i]] if x.isdigit()][0]
                except IndexError:
                    outs += s[i]
            else:
                outs += s[i]
    else:
        outs = s

    return outs


def depunctuate(s):
    return s.translate(str.maketrans('', '', ',.?!$%&():;-"'))


def is_number(s):
    return depunctuate(s).isdigit()


def is_word(s, wordset):
    return depunctuate(s.lower()) in wordset


def is_initial(s):
    return len(s) == 2 and s[0].isupper() and s[0].isalpha() and s[1] == "."


def is_abbrev(s, abbrevset):
    return s.lower() in abbrevset


def visual_spell_checker(
        textline,
        worddict,
        vsim_dict,
        abbrevset,
        beam=1000,
        splitter_pattern=r"( |/|-|\"|')",
        majority_norm=True
    ):

    # final list to return
    splitters = splitter_pattern[1:-1].split("|")
    spell_checked_words = []

    # go through each word individually
    for w in re.split(splitter_pattern, textline):

        # dont do anytyhing if empty
        if len(w) > 0 and not w in splitters:

            # check if word or number
            if not is_word(w, worddict) and not is_number(w) and not all_caps(w):

                # if not, create list of candidate words to check iteratively
                candidate_words = [w]

                # also collect words found to be in dict
                words_in_dict = []
                numbers = []
                initials = []
                abbrevs = []

                # go character by character
                for idx, c in enumerate(w):

                    # check homoglyphs
                    if c in vsim_dict:
                        alts = vsim_dict[c]

                        # go thru homoglyphs and make subs in candidates
                        for alt in alts:
                            new_candidate_words = []
                            for cw in candidate_words:
                                altw = cw[:idx] + alt + cw[idx+1:]
                                # check if real word found
                                if is_word(altw, worddict):
                                    words_in_dict.append(altw)
                                    new_candidate_words.append(altw)
                                elif is_abbrev(altw, abbrevset):
                                    abbrevs.append(altw)
                                    new_candidate_words.append(altw)
                                elif is_number(altw):
                                    numbers.append(altw)
                                    new_candidate_words.append(altw)
                                elif is_initial(altw):
                                    initials.append(altw)
                                    new_candidate_words.append(altw)
                                else:
                                    new_candidate_words.append(altw)
                            # add new candidates for next homo sub
                            candidate_words += new_candidate_words
                            # beam it!
                            candidate_words = candidate_words[-beam:]

                # pick max freq word in dict, or pick number, or append uncorrected
                if len(words_in_dict) > 0:
                    freqs = [worddict[depunctuate(rw).lower()] for rw in words_in_dict]
                    max_freq = max(freqs)
                    max_freq_index = freqs.index(max_freq)
                    spell_checked_words.append(words_in_dict[max_freq_index])
                elif len(abbrevs) > 0:
                    spell_checked_words.append(abbrevs[0])
                elif len(initials) > 0:
                    spell_checked_words.append(initials[0])
                elif len(numbers) > 0:
                    spell_checked_words.append(numbers[0])
                else:
                    spell_checked_words.append(w)

            # if word found with no substitution needed, add it and move on
            else:
                spell_checked_words.append(w)

        # add in splitter
        else:
            spell_checked_words.append(w)

    if majority_norm:
        spell_checked_words = [majority_normalize(w, vsim_dict) \
            if not w in splitters and not is_number(w) else w for w in spell_checked_words]
    return "".join(spell_checked_words)


def textline_evaluation(
        pairs,
        print_incorrect=False,
        no_spaces_in_eval=False,
        norm_edit_distance=False,
        uncased=False
    ):

    n_correct = 0
    edit_count = 0
    length_of_data = len(pairs)
    n_chars = sum(len(gt) for gt, _ in pairs)

    for gt, pred in pairs:

        # eval w/o spaces
        pred, gt = string_cleaner(pred), string_cleaner(gt)
        gt = gt.strip() if not no_spaces_in_eval else gt.strip().replace(" ", "")
        pred = pred.strip() if not no_spaces_in_eval else pred.strip().replace(" ", "")
        if uncased:
            pred, gt = pred.lower(), gt.lower()

        # textline accuracy
        if pred == gt:
            n_correct += 1
        else:
            if print_incorrect:
                print(f"GT: {gt}\nPR: {pred}\n")

        # ICDAR2019 Normalized Edit Distance
        if norm_edit_distance:
            if len(gt) > len(pred):
                edit_count += edit_distance(pred, gt) / len(gt)
            else:
                edit_count += edit_distance(pred, gt) / len(pred)
        else:
            edit_count += edit_distance(pred, gt)

    accuracy = n_correct / float(length_of_data) * 100
    if norm_edit_distance:
        cer = edit_count / float(length_of_data)
    else:
        cer = edit_count / n_chars

    return accuracy, cer