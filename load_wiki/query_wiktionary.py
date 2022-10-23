import json
import pdb
from collections import defaultdict
import string
import os
from pathlib import Path
import spacy


wikidata_descr = {
    'walmart': 'U.S. discount retailer based in Arkansas.',
    'wyoming': 'least populous state of the United States of America',
    'safeway': 'American supermarket chain',
    'mcdonalds': 'American fast food restaurant chain',
    'washington d.c': 'capital city of the United States',
    'espn': 'American pay television sports network',
    'windows 95': 'operating system from Microsoft'
}
my_mapping = {
    'jumping rope': 'jump rope',
    'eden': 'Eden',
    'contemplating': 'contemplate',
    'rehabilitating': 'rehabilitate',
    'catalog': 'catalogue',
    'works': 'work',
    'hoping': 'hope',
    'wetlands': 'wetland',
    'waiting': 'wait',
    'sunglass': 'sunglasses',
    'centre': 'center',
    'bath room': 'bathroom',
    'phd': 'ph.d.',
    'sunglasses': 'sunglasses',
}
nlp=spacy.load('en_core_web_sm')

bad_form_of = []
def lemma_first(qc):   
    words = nlp(qc)
    qc_words = [w.text for w in words]
    lemma_word = words[0].lemma_ if words[0].lemma_ != '-PRON-' else words[0].text
    if qc_words[0] == lemma_word:
        return qc, qc_words
    else:
        qc_words[0] = lemma_word
        qc_new = ' '.join(qc_words)
        return qc_new, qc_words

def resolve_meaning(qc, wik_dict, return_mean=True):
    qc = qc.lower()
    if isinstance(qc, list):
        qc = qc[0]
    if qc in wikidata_descr:
        if return_mean:
            return wikidata_descr[qc]
        else:
            return qc, wikidata_descr[qc]
    if qc == '':
        return None
    if qc in my_mapping:
        # print('replacing {} with {}'.format(qc, my_mapping[qc]))
        qc = my_mapping[qc]
    else:
        qc_new, _ = lemma_first(qc)
        #print(qc_new, qc)
        qc = qc if (qc in wik_dict and qc_new not in wik_dict) else qc_new

    #     qc_new = qc.translate(str.maketrans('', '', string.punctuation))
        qc_new = qc.strip(string.punctuation+' ')
        if qc_new != qc:
            qc = qc_new
    if qc in wik_dict:
        for meaning in wik_dict[qc]:
            if 'senses' in meaning:
                for sense in meaning['senses']:
                    if 'form_of' in sense or 'alt_of' in sense:
                        form_str = 'form_of' if 'form_of' in sense else 'alt_of'
                        qc_new = sense[form_str][0]['word']
                        if qc.lower() == qc_new.lower():
                            continue
                        if len(qc_new.split(' ')) > 4:
                            # print('bad form: {}, {}', qc, qc_new)
                            bad_form_of.append(qc_new)
                        else:
                            # print('{}: replacing {} with {}'.format(form_str, qc, qc_new))
                            return resolve_meaning(qc_new, wik_dict, return_mean=return_mean)
                    elif 'heads' in meaning and meaning['heads'][0].get('2', '') == 'verb form':
                        try_str = sense['glosses'][0].split(' of ')
                        if len(try_str) != 2:
                            pdb.set_trace()
                        qc_new = try_str[-1]
                        return resolve_meaning(qc_new, wik_dict, return_mean=return_mean)
                
                    if 'glosses' in sense:
                        if return_mean:
                            return sense['glosses'][0]
                        else:
                            return qc, sense['glosses'][0]
                        mstr = '{}: {}'.format(qc, sense['glosses'][0])
                        if sense['glosses'][0] == 'en':
                              pdb.set_trace()
                        if return_mean:
                            return mstr
                        else:
                            return qc
    qc_new, qc_words = lemma_first(qc)
    if qc_new in wik_dict and qc_new != qc:
        return resolve_meaning(qc_new, wik_dict, return_mean=return_mean)
    qc_new = ''.join(qc_words)
    if qc_new in wik_dict and qc_new != qc:
        return resolve_meaning(qc_new, wik_dict, return_mean=return_mean)    
    qc_new = ' '.join(qc.split(' ')[1:])                         
    if qc_new == qc:
        pdb.set_trace()
    res = resolve_meaning(qc_new, wik_dict, return_mean=return_mean)
    if res is not None:
        return res
    qc_new = qc.translate(str.maketrans('', '', string.punctuation))
    if qc_new in wik_dict and qc_new != qc:
        return resolve_meaning(qc_new, wik_dict, return_mean=return_mean)
    qc_new = qc.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    if qc_new in wik_dict and qc_new != qc:
        return resolve_meaning(qc_new, wik_dict, return_mean=return_mean)


if __name__ == '__main__':
    #wikdict_fn = 'wik_dict.10.json'
    wikdict_fn = 'wik_dict.json'
    wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))

    for word in wikidata_descr:
        wik_dict[word] = None
    hard_words = []
    def resolve_meaning_total(qc, hard_words):
        res = resolve_meaning(qc, wik_dict)
        if res is None:
            print('hard word:', qc)
            hard_words.append(qc)
        return res
            
    print(resolve_meaning_total('orients', hard_words))
