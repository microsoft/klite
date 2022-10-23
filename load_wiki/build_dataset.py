import spacy
import json
import os.path as op
import os
from pathlib import Path
import nltk
import numpy as np
from typing import List
from query_wiktionary import resolve_meaning
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import glob
from nltk.corpus import wordnet as wn
import random
# python -m spacy download en_core_web_sm
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        # if e.errno != errno.EEXIST:
            # raise
        print(e)

def get_args_parser():
    parser = argparse.ArgumentParser('preprocess', add_help=False)

    # Dataset
    parser.add_argument('--dataset', default="web_questions", type=str, help='Training dataset.')
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')

    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')

    # File process
    parser.add_argument('--start_idx', type=int, default=0, help="The question index to star asking GPT3")
    parser.add_argument('--process_lines', type=int, default=5, help="The question index to star asking GPT3")

    parser.add_argument('--start_from_exists', type=int, default=0, help="Whether start from existing processed one, and continue")
    
    # Merge files
    parser.add_argument('--merge', type=int, default=0, help="Whether merge the files")
    parser.add_argument('--merge_prompt', type=int, default=0, help="Whether merge the files using entity prompt")
    parser.add_argument('--merge_span', type=int, default=0, help="Whether merge the files using entity span")

    parser.add_argument('--image_label', type=int, default=0, help="Whether the image label data or not")

    return parser


def tsv_writer(values, tsv_file, sep='\t', lineidx=False):
    # mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    if lineidx:
        with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
            assert values is not None
            for value in values:
                assert value is not None
                # this step makes sure python2 and python3 encoded img string are the same.
                # for python2 encoded image string, it is a str class starts with "/".
                # for python3 encoded image string, it is a bytes class starts with "b'/".
                # v.decode('utf-8') converts bytes to str so the content is the same.
                # v.decode('utf-8') should only be applied to bytes class type. 
                value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
                v = '{0}\n'.format(sep.join(map(str, value)))
                fp.write(v)
                fpidx.write(str(idx) + '\n')
                idx = idx + len(v)
        os.rename(tsv_file_tmp, tsv_file)
        os.rename(lineidx_file_tmp, lineidx_file)
    else:
        with open(tsv_file_tmp, 'w') as fp:
            assert values is not None
            for value in values:
                assert value is not None
                # this step makes sure python2 and python3 encoded img string are the same.
                # for python2 encoded image string, it is a str class starts with "/".
                # for python3 encoded image string, it is a bytes class starts with "b'/".
                # v.decode('utf-8') converts bytes to str so the content is the same.
                # v.decode('utf-8') should only be applied to bytes class type. 
                value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
                v = '{0}\n'.format(sep.join(map(str, value)))
                fp.write(v)
                idx = idx + len(v)
        os.rename(tsv_file_tmp, tsv_file)


def tsv_reader(tsv_file, sep='\t'):
    lines = []
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            # yield [x.strip() for x in line.split(sep)]
            lines.append( [x.strip() for x in line.split(sep)] )
    return lines

def find_noun_phrases_nltk(caption: str) -> List[str]:
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    # result.draw()
    return noun_phrases

def find_nn_spacy(nlp, caption: str) -> List[str]:
    caption = caption #.lower()
    # doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

    doc = nlp(caption)
    NN, entity, subjobj, subjobj_root, entity_label = list(), list(), list(), list(), list()
    for token in doc:
        # get NN with POS
        if "NN" in token.tag_:
            NN.append( token.text )
    
    for chunk in doc.noun_chunks:
        subjobj.append( chunk.text )
        subjobj_root.append( chunk.root.text )

    for ent in doc.ents:
        entity.append( ent.text )
        entity_label.append( ent.label_  )

    noun_phrases = []
    # reference from https://github.com/vacancy/SceneGraphParser/blob/master/sng_parser/backends/spacy_parser.py
    for noun_chunk in doc.noun_chunks:
        # Ignore pronouns such as "it".
        if noun_chunk.root.lemma_ == '-PRON-':
            continue

        ent = dict(
            span=noun_chunk.text,
            lemma_span=noun_chunk.lemma_,
            head=noun_chunk.root.text,
            lemma_head=noun_chunk.root.lemma_,
        )

        for x in noun_chunk.root.children:
            if x.dep_ == 'compound':  # compound nouns
                ent['head'] = x.text + ' ' + ent['head']
                ent['lemma_head'] = x.lemma_ + ' ' + ent['lemma_head']
        
        noun_phrases.append(ent['lemma_head'])

    return NN, entity, subjobj, subjobj_root, noun_phrases

def get_wiki_meaning(text, wik_dict, return_mean=True):
    try:
        res = resolve_meaning(text, wik_dict, return_mean=return_mean)
        if res is None: return '', ''
    except:
        # res = ['', '']
        return '', ''
    return res


def parse_tsv(datafile, datafield, nlp, outputdir, start_idx, process_lines=5, lookup=False, debug=False, tokenize=False):
    tsv_lines = tsv_reader(datafile)
    file_name = datafile.split('/')[-1]
    data_output = []
    end_idx = min(start_idx+process_lines, len(tsv_lines))
    if lookup:
        wikdict_fn = 'wik_dict.json'
        # wikdict_fn = 'wik_dict.10.json'
        # wikdict_fn = '/home/sheng/project/load_wiki/wik_dict.10.json'
        wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))

    for img_id, txt_json in tqdm(tsv_lines[start_idx:end_idx]):
        txt_json = json.loads( txt_json )
        caption = txt_json[ datafield ]

        if debug: print( caption )

        caption_is_not_none = caption[0] is None
        if caption_is_not_none:
            print(f"caption is None {img_id}")
            data_output.append( [ img_id, json.dumps( txt_json ) ] )
            continue
        _NN, _entity, _subjobj, _subjobj_root, _noun_phrases = [], [], [], [], []
        for c_ in caption:
            NN, entity, subjobj, subjobj_root, noun_phrases = find_nn_spacy( nlp, c_ )
            _NN.append(NN), _entity.append(entity), _subjobj.append(subjobj), _subjobj_root.append(subjobj_root), _noun_phrases.append(noun_phrases)
        
        wiki_field = {"captions_nn": _NN, "captions_entity":_entity, "captions_subjobj":_subjobj, "captions_subjobj_root":_subjobj_root, "captions_noun_phrases":_noun_phrases}
        txt_json.update( wiki_field )
        if debug: print( txt_json )

        # TODO: lookup the dictionary
        if lookup:
            txt_json_wiki = dict()
            for field in wiki_field:
                # if field != datafield:
                lookup_nn = np.unique( txt_json[ field ] )
                _wiki_nns = []
                for lookup_nns in txt_json[ field ]:
                    lookup_nn = np.unique( lookup_nns )
                    _wiki_nn = []
                    for _nn in lookup_nn:
                        wiki_nn_token, wiki_nn = get_wiki_meaning( _nn, wik_dict, return_mean=False )
                        if wiki_nn != '':
                            if tokenize:
                                # wiki_nn = ' '.join( word_tokenize( wiki_nn.lower() ) )
                                wiki_nn = ' '.join( word_tokenize( wiki_nn ) )
                            _wiki_nn.append( (_nn, wiki_nn_token, wiki_nn) )
                    _wiki_nns.append( _wiki_nn )
                txt_json_wiki[f"{field}_wiki"] =  _wiki_nns
                if debug: print(field, lookup_nn)
            txt_json.update(txt_json_wiki)
            if debug: print( txt_json )
            # if debug: exit()

        data_output.append( [ img_id, json.dumps( txt_json ) ] )
    
    if len(data_output) > 0:
        outputfile = f"{outputdir}/{start_idx}_{end_idx}_{file_name}"
        tsv_writer(data_output, outputfile)

def hypernyms_chain(concept, wn):
    ss = wn.synsets(concept)
    hypernyms_chain = []
    # chain_list = ss.hypernym_paths()
    
    while len(ss) > 0:
        ss = ss[0]
        
        lemma = random.choice( ss.lemmas() )
        # hypernyms_chain.append(ss.lemmas()[0].name() )
        hypernyms_chain.append( lemma.name() )

        # print(f'{ss.name()}, {ss.definition()}, {ss.hypernyms()}')
        ss = ss.hypernyms()

    hypernyms_chain = ' , '.join(hypernyms_chain)
    return hypernyms_chain

def parse_image_label_tsv(datafile, datafield, nlp, outputdir, start_idx, process_lines=5, lookup=False, debug=False, tokenize=False):
    tsv_lines = tsv_reader(datafile)
    file_name = datafile.split('/')[-1]
    data_output = []
    end_idx = min(start_idx+process_lines, len(tsv_lines))
    if lookup:
        wikdict_fn = 'wik_dict.json'
        # wikdict_fn = 'wik_dict.10.json'
        # wikdict_fn = '/home/sheng/project/load_wiki/wik_dict.10.json'
        wik_dict = json.load(open(wikdict_fn, encoding='utf-8'))
    # do wordnet as well
    for img_id, txt_json in tqdm(tsv_lines[start_idx:end_idx]):
        txt_json = json.loads( txt_json )
        class_name = txt_json[datafield]
        if lookup:
            # wiki_nn = get_wiki_meaning( class_name, wik_dict )
            wiki_nn_token, wiki_nn = get_wiki_meaning( class_name, wik_dict, return_mean=False )
            if wiki_nn != '':
                if tokenize:
                    # wiki_nn = ' '.join( word_tokenize( wiki_nn.lower() ) )
                    wiki_nn = ' '.join( word_tokenize( wiki_nn ) )

            txt_json["definition_wiki"] = wiki_nn
            txt_json["definition_wiki_token"] = wiki_nn_token
            # wordnet
            # use offset
            if 'class' in txt_json:
                class_id = txt_json['class']
                res_synset = wn.synset_from_pos_and_offset('n',int(class_id.split('n')[-1]))
                chain = hypernyms_chain(res_synset.name().split('.')[0], wn)
                definition = res_synset.definition()
                txt_json['hypernyms_chain'] = chain.lower()
                # txt_json['definition'] = ' '.join( word_tokenize( definition.lower() ) )
                txt_json['definition'] = ' '.join( word_tokenize( definition ) )

            # TODO: look it up
        # print(txt_json)
        data_output.append( [ img_id, json.dumps( txt_json ) ] )
    outputfile = f"{outputdir}/{start_idx}_{end_idx}_{file_name}"
    tsv_writer(data_output, outputfile)

def merge_image_label_tsv(datadir, datafield, outputdir, debug=False):
    datafiles = np.unique([ '_'.join(_.split('_')[2:]) for _ in os.listdir(datadir)])
    print(datafiles)
    # coverage
    # for datafile in glob.glob(datadir + "/*tsv"):
    #     print(datafile)
    #     tsv_lines = tsv_reader(datafile)
    #     for img_id, txt_json in tqdm(tsv_lines):
    #         txt_json = json.loads( txt_json )

    # select the most infrequent data then append
    for raw_datafile in datafiles:
        print(raw_datafile)
        data_output = []
        real_datafiles = glob.glob(datadir + f"/*{raw_datafile}")
        real_datafiles = sorted([(int( (_.split('/')[-1]).split('_')[1] ), _) for _ in real_datafiles])
        print(real_datafiles)
        for idx, datafile in real_datafiles:
            print(datafile)
            tsv_lines = tsv_reader(datafile)
            for img_id, txt_json in tqdm(tsv_lines):
                txt_json = json.loads( txt_json )
                data_output.append( [ img_id, json.dumps( txt_json ) ] )

        outputfile = f"{outputdir}/{raw_datafile}"
        tsv_writer(data_output, outputfile, lineidx=True)

def merge_tsv(datadir, datafield, outputdir, debug=False):
    datafiles = np.unique([_.split('_')[-1] for _ in os.listdir(datadir)])
    print(datafiles)
    # construct the frequency dict
    freq_dict, coverage = dict(), dict()
    for datafile in glob.glob(datadir + "/*tsv"):
        print(datafile)
        tsv_lines = tsv_reader(datafile)
        for img_id, txt_json in tqdm(tsv_lines):
            txt_json = json.loads( txt_json )
            for c_field in txt_json:
                if c_field != datafield and 'wiki' not in c_field:
                    if c_field not in freq_dict: freq_dict[c_field] = dict()
                    if c_field not in coverage: coverage[c_field] = [0, 0]
                    for c_datas in txt_json[c_field]:
                        for c_data in c_datas:
                            if c_data not in freq_dict[c_field]: freq_dict[c_field][c_data] = 0
                            freq_dict[c_field][c_data] += 1
                        if len(c_datas): coverage[c_field][0] += 1
                        coverage[c_field][1] += 1
            # if debug: print(freq_dict, txt_json, coverage)
            # if debug: break
    
    print("#############  coverage  #################")
    for item in coverage:
        print(item, coverage[item][0] / coverage[item][1], coverage[item])
    # exit()

    # select the most infrequent data then append
    for raw_datafile in datafiles:
        print(raw_datafile)
        data_output = []
        real_datafiles = glob.glob(datadir + f"/*{raw_datafile}")
        real_datafiles = sorted([(int(_.split('_')[-2]), _) for _ in real_datafiles])
        print(real_datafiles)
        for idx, datafile in real_datafiles:
            print(datafile)
            tsv_lines = tsv_reader(datafile)

            for img_id, txt_json in tqdm(tsv_lines):
                txt_json = json.loads( txt_json )
                result_txt_json = { datafield:  txt_json[ datafield ] }
                for c_field in txt_json:
                    if 'wiki' not in c_field: continue
                    c_field_no_wiki = c_field.replace("_wiki", "")
                    result_txt_json[ c_field ] = []
                    # for only one caption
                    freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1]) for c_item in txt_json[c_field] ]
                    if len(freq_c_data) > 0:
                        if debug: print(freq_c_data)
                        aug_c_token_wiki = min( freq_c_data )
                        aug_c_token_wiki = ' , '.join( [aug_c_token_wiki[1], aug_c_token_wiki[2]] )
                        aug_c_nn_np = ' ; '.join( [txt_json[datafield][0], aug_c_token_wiki] )
                    else:
                        aug_c_nn_np = txt_json[datafield][0]
                    result_txt_json[ c_field ].append( aug_c_nn_np )

                    # for c_caption, c_data in zip(txt_json[datafield], txt_json[c_field]):
                    #     if debug: print(txt_json[datafield], c_data, c_field, txt_json[c_field])
                    #     freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1]) for c_item in c_data ]
                    #     if len(freq_c_data):
                    #         aug_c_token_wiki = min( freq_c_data )
                    #         aug_c_token_wiki = ' , '.join( [aug_c_token_wiki[1], aug_c_token_wiki[2]] )
                    #         aug_c_nn_np = ' ; '.join( [c_caption, aug_c_token_wiki] )
                    #     else:
                    #         aug_c_nn_np = c_caption
                    #     result_txt_json[ c_field ].append( aug_c_nn_np )

                if debug: print(result_txt_json, txt_json, aug_c_token_wiki)
                if debug: break

                data_output.append( [ img_id, json.dumps( result_txt_json ) ] )

        outputfile = f"{outputdir}/{raw_datafile}"
        tsv_writer(data_output, outputfile, lineidx=True)


def merge_tsv_entity_prompt(datadir, datafield, outputdir, debug=False):
    from prompt_engineering import prompt_engineering
    datafiles = np.unique([_.split('_')[-1] for _ in os.listdir(datadir)])
    print(datafiles)
    # construct the frequency dict
    freq_dict, coverage = dict(), dict()
    for datafile in glob.glob(datadir + "/*tsv"):
        print(datafile)
        tsv_lines = tsv_reader(datafile)
        for img_id, txt_json in tqdm(tsv_lines):
            txt_json = json.loads( txt_json )
            for c_field in txt_json:
                if c_field != datafield and 'wiki' not in c_field and c_field != 'source':
                    if c_field not in freq_dict: freq_dict[c_field] = dict()
                    if c_field not in coverage: coverage[c_field] = [0, 0]
                    for c_datas in txt_json[c_field]:
                        for c_data in c_datas:
                            if c_data not in freq_dict[c_field]: freq_dict[c_field][c_data] = 0
                            freq_dict[c_field][c_data] += 1
                        if len(c_datas): coverage[c_field][0] += 1
                        coverage[c_field][1] += 1
            if debug: print(freq_dict, txt_json, coverage)
            if debug: break
        # if debug: break
    
    print("#############  coverage  #################")
    for item in coverage:
        print(item, coverage[item][0] / coverage[item][1], coverage[item])
    # exit()

    # select the most infrequent data then append
    for raw_datafile in datafiles:
        print(raw_datafile)
        data_output = []
        real_datafiles = glob.glob(datadir + f"/*{raw_datafile}")
        real_datafiles = sorted([(int(_.split('_')[-2]), _) for _ in real_datafiles])
        print(real_datafiles)
        for idx, datafile in real_datafiles:
            print(datafile)
            tsv_lines = tsv_reader(datafile)

            for img_id, txt_json in tqdm(tsv_lines):
                txt_json = json.loads( txt_json )
                result_txt_json = { datafield:  txt_json[ datafield ] }
                for c_field in txt_json:
                    if 'wiki' not in c_field: continue
                    c_field_no_wiki = c_field.replace("_wiki", "")
                    result_txt_json[ c_field ] = []
                    result_txt_json[ c_field + '_freq' ] = []
                    result_txt_json[ c_field_no_wiki ], result_txt_json[ c_field_no_wiki + '_freq' ] = [], []
                    # for only one caption
                    freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1]) for c_item in txt_json[c_field] ]
                    
                    if len(freq_c_data) > 0:
                        if debug: print(freq_c_data)
                        aug_c_token_wiki = min( freq_c_data )
                        aug_c_token_wiki_txt = ' , '.join( [aug_c_token_wiki[1], aug_c_token_wiki[2]] )

                        # ori_text = txt_json[datafield][0]
                        ori_text = prompt_engineering( aug_c_token_wiki[1] )
                        

                        # aug_c_nn_np = ' ; '.join( [ori_text, aug_c_token_wiki_txt] )
                        aug_c_token = aug_c_token_wiki[1]
                        aug_c_nn_np = aug_c_token_wiki_txt

                        aug_c_token_wiki = max( freq_c_data )
                        aug_c_token_wiki_txt = ' , '.join( [aug_c_token_wiki[1], aug_c_token_wiki[2]] )
                        aug_c_token_freq = aug_c_token_wiki[1]
                        aug_c_nn_np_freq = aug_c_token_wiki_txt
                        # aug_c_nn_np_freq = ' ; '.join( [ori_text, aug_c_token_wiki_txt] )

                        result_txt_json[ c_field_no_wiki ].append( aug_c_token )
                        result_txt_json[ c_field_no_wiki + '_freq' ].append( aug_c_token_freq )
                    else:
                        aug_c_nn_np = txt_json[datafield][0]
                        aug_c_nn_np_freq = txt_json[datafield][0]
                        # aug_c_nn_np = ori_text
                    result_txt_json[ c_field ].append( aug_c_nn_np )
                    result_txt_json[ c_field + '_freq' ].append( aug_c_nn_np_freq )

                    # for c_caption, c_data in zip(txt_json[datafield], txt_json[c_field]):
                    #     if debug: print(txt_json[datafield], c_data, c_field, txt_json[c_field])
                    #     freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1]) for c_item in c_data ]
                    #     if len(freq_c_data):
                    #         aug_c_token_wiki = min( freq_c_data )
                    #         aug_c_token_wiki = ' , '.join( [aug_c_token_wiki[1], aug_c_token_wiki[2]] )
                    #         aug_c_nn_np = ' ; '.join( [c_caption, aug_c_token_wiki] )
                    #     else:
                    #         aug_c_nn_np = c_caption
                    #     result_txt_json[ c_field ].append( aug_c_nn_np )

                if debug: print(result_txt_json, txt_json, aug_c_token_wiki)
                if debug: break

                data_output.append( [ img_id, json.dumps( result_txt_json ) ] )
            if debug: exit()

        outputfile = f"{outputdir}/{raw_datafile}"
        tsv_writer(data_output, outputfile, lineidx=True)


def merge_tsv_entity_span(datadir, datafield, outputdir, debug=False):
    import re
    from prompt_engineering import prompt_engineering
    datafiles = np.unique([_.split('_')[-1] for _ in os.listdir(datadir)])
    print(datafiles)
    # construct the frequency dict
    freq_dict, coverage = dict(), dict()
    for datafile in glob.glob(datadir + "/*tsv"):
        print(datafile)
        tsv_lines = tsv_reader(datafile)
        for img_id, txt_json in tqdm(tsv_lines):
            txt_json = json.loads( txt_json )
            for c_field in txt_json:
                if c_field != datafield and 'wiki' not in c_field and c_field != 'source':
                    if c_field not in freq_dict: freq_dict[c_field] = dict()
                    if c_field not in coverage: coverage[c_field] = [0, 0]
                    for c_datas in txt_json[c_field]:
                        for c_data in c_datas:
                            if c_data not in freq_dict[c_field]: freq_dict[c_field][c_data] = 0
                            freq_dict[c_field][c_data] += 1
                        if len(c_datas): coverage[c_field][0] += 1
                        coverage[c_field][1] += 1
                elif c_field == datafield:
                    if c_field not in freq_dict: freq_dict[c_field] = dict()
                    assert len(txt_json[datafield]) == 1
                    for c_data in txt_json[datafield][0].split():
                        if c_data not in freq_dict[c_field]: freq_dict[c_field][c_data] = 0
                        freq_dict[c_field][c_data] += 1
            if debug: print(freq_dict, txt_json, coverage)
            if debug: break
        # if debug: break
    
    print("#############  coverage  #################")
    for item in coverage:
        print(item, coverage[item][0] / coverage[item][1], coverage[item])
    # exit()

    # select the most infrequent data then append
    for raw_datafile in datafiles:
        print(raw_datafile)
        data_output = []
        real_datafiles = glob.glob(datadir + f"/*{raw_datafile}")
        real_datafiles = sorted([(int(_.split('_')[-2]), _) for _ in real_datafiles])
        print(real_datafiles)
        for idx, datafile in real_datafiles:
            print(datafile)
            tsv_lines = tsv_reader(datafile)

            for img_id, txt_json in tqdm(tsv_lines):
                txt_json = json.loads( txt_json )
                result_txt_json = { datafield:  txt_json[ datafield ] }
                for c_field in txt_json:
                    if 'wiki' not in c_field or 'source' in c_field: continue
                    c_field_no_wiki = c_field.replace("_wiki", "")

                    # freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1]) for c_item in txt_json[c_field] ]
                    freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1]) for c_item in txt_json[c_field][0] ]
                    result_txt_json[ c_field ] = []
                    if len(freq_c_data) > 0:
                        if debug: print(freq_c_data)
                        aug_c_token_wiki = min( freq_c_data )
                        aug_c_token_wiki = ' , '.join( [aug_c_token_wiki[1], aug_c_token_wiki[-1]] )
                        aug_c_nn_np = ' ; '.join( [txt_json[datafield][0], aug_c_token_wiki] )
                    else:
                        aug_c_nn_np = txt_json[datafield][0]

                    
                    result_txt_json[ c_field ].append( aug_c_nn_np )
                    
                    # for only one caption
                    # print(c_field, txt_json[c_field])
                    # freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1]) for c_item in txt_json[c_field] ]
                    freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1], c_item[2]) for c_item in txt_json[c_field][0] ]
                    # print(freq_c_data)
                    # exit()
                    # result_txt_json[ c_field ] = []

                    if len(freq_c_data) > 0:
                        if debug: print(freq_c_data)
                        
                        result_txt_json[ c_field_no_wiki + '_span' ] = [[]]
                        result_txt_json[ c_field_no_wiki + '_token' ] = [[]]

                        caption = txt_json[ datafield ][0]
                        # caption = aug_c_nn_np
                        assert len(txt_json[ datafield ]) == 1
                        # for freq, token, token_wiki in freq_c_data:

                        for freq, token, token_wiki, token_wiki_def in freq_c_data:
                            try:
                                token_span = [(a.start(), a.end()) for a in list(re.finditer(token, caption))]
                                # result_txt_json[ c_field_no_wiki + '_token' ][0] += [token_wiki] * len(token_span)
                                result_txt_json[ c_field_no_wiki + '_token' ][0] += [token] * len(token_span)
                                result_txt_json[ c_field_no_wiki + '_span' ][0] += token_span
                            except:
                                token_start = caption.find(token)
                                token_end = token_start + len(token)
                                result_txt_json[ c_field_no_wiki + '_span' ][0].append( (token_start, token_end) )
                                # result_txt_json[ c_field_no_wiki + '_token' ][0].append(token_wiki)
                                result_txt_json[ c_field_no_wiki + '_token' ][0].append(token)
                                print(caption, token, token_wiki, caption.find(token))
                                # exit()
                            # result_txt_json[ c_field_no_wiki + '_token' ][0].append(token)
                            # token_start = caption.find(token)
                            # print( caption.findall(token) )
                            # token_end = token_start + len(token)
                            # result_txt_json[ c_field_no_wiki + '_span' ][0].append( (token_start, token_end) )
                    else:
                        caption = txt_json[ datafield ][0]
                        # caption = aug_c_nn_np
                        assert len(txt_json[ datafield ]) == 1
                        result_txt_json[ c_field_no_wiki + '_span' ] = [[]]
                        # select the most infrequent word in the caption
                        # freq_c_data = [ (freq_dict[datafield][c_data], c_data) for c_data in caption.split() ]
                        # result_txt_json[ c_field_no_wiki + '_span' ][0].append( (0, 0) )
                        if len(freq_c_data) > 0:
                            aug_c_token_wiki = min( freq_c_data )
                            token = aug_c_token_wiki[1]
                            try:
                                token_span = [(a.start(), a.end()) for a in list(re.finditer(token, caption))]
                                result_txt_json[ c_field_no_wiki + '_span' ][0] += token_span
                            except:
                                print(caption, token, caption.find(token))
                                token_start = caption.find(token)
                                token_end = token_start + len(token)
                                result_txt_json[ c_field_no_wiki + '_span' ][0].append( (token_start, token_end) )
                                # exit()
                        #     # token_start = caption.find(token)
                        #     # token_end = token_start + len(token)
                        #     # result_txt_json[ c_field_no_wiki + '_span' ][0].append( (token_start, token_end) )
                        # else:
                        #     print(caption, caption.split())
                        #     result_txt_json[ c_field_no_wiki + '_span' ][0].append( (0, 0) )
                    
                    # else:
                    #     aug_c_nn_np = txt_json[datafield][0]
                    #     aug_c_nn_np_freq = txt_json[datafield][0]
                    #     # aug_c_nn_np = ori_text
                    # result_txt_json[ c_field ].append( aug_c_nn_np )
                    # result_txt_json[ c_field + '_freq' ].append( aug_c_nn_np_freq )

                    # for c_caption, c_data in zip(txt_json[datafield], txt_json[c_field]):
                    #     if debug: print(txt_json[datafield], c_data, c_field, txt_json[c_field])
                    #     freq_c_data = [ (freq_dict[c_field_no_wiki][c_item[0]], c_item[0], c_item[1]) for c_item in c_data ]
                    #     if len(freq_c_data):
                    #         aug_c_token_wiki = min( freq_c_data )
                    #         aug_c_token_wiki = ' , '.join( [aug_c_token_wiki[1], aug_c_token_wiki[2]] )
                    #         aug_c_nn_np = ' ; '.join( [c_caption, aug_c_token_wiki] )
                    #     else:
                    #         aug_c_nn_np = c_caption
                    #     result_txt_json[ c_field ].append( aug_c_nn_np )

                if debug: print(result_txt_json, txt_json, aug_c_token_wiki)
                if debug: break

                data_output.append( [ img_id, json.dumps( result_txt_json ) ] )
            if debug: exit()

        outputfile = f"{outputdir}/{raw_datafile}"
        tsv_writer(data_output, outputfile, lineidx=True)

if __name__ == '__main__':
    import argparse
    parser = get_args_parser()
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.merge:
        if args.image_label:
            merge_image_label_tsv(args.data_path, 'class_name', args.output_dir)
        else:
            # merge_tsv(args.data_path, 'captions', args.output_dir)
            if args.merge_prompt:
                merge_tsv_entity_prompt(args.data_path, 'captions', args.output_dir, debug=False)
            elif args.merge_span:
                merge_tsv_entity_span(args.data_path, 'captions', args.output_dir, debug=False)
            else:
                merge_tsv(args.data_path, 'captions', args.output_dir)
    else:
        if args.image_label:
            parse_image_label_tsv(args.data_path, 'class_name', nlp, args.output_dir, args.start_idx, args.process_lines, True, tokenize=True)
        else:
            parse_tsv(args.data_path, 'captions', nlp, args.output_dir, args.start_idx, args.process_lines, True, tokenize=True, debug=False)
# 
# cc3m
# #############  coverage  #################
# captions_nn 0.9989278846153846 [2077770, 2080000]
# captions_entity 0.1588144230769231 [330334, 2080000]
# captions_subjobj 0.9993076923076923 [2078560, 2080000]
# captions_subjobj_root 0.9993076923076923 [2078560, 2080000]
# captions_noun_phrases 0.9993076923076923 [2078560, 2080000]

# cc12m
#############  coverage  #################
# source 1.0 [56303450, 56303450]
# captions_nn 0.9989291064757133 [11248631, 11260690]
# captions_entity 0.4828423480266307 [5437138, 11260690]
# captions_subjobj 0.9976890403696399 [11234667, 11260690]
# captions_subjobj_root 0.9976890403696399 [11234667, 11260690]
# captions_noun_phrases 0.9976890403696399 [11234667, 11260690]

# yfcc13m
#  #############  coverage  #################
#  captions_nn 0.9974933540975578 [14155503, 14191075]
#  captions_entity 0.7346407513172892 [10425342, 14191075]
#  captions_subjobj 0.9950345551693582 [14120610, 14191075]
#  captions_subjobj_root 0.9950345551693582 [14120610, 14191075]
#  captions_noun_phrases 0.9950345551693582 [14120610, 14191075]