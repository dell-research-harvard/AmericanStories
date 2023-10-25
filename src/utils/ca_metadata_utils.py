import os
import json
import re
from rdflib import Graph, Namespace, URIRef
import urllib
from copy import deepcopy
import requests
from rdflib.namespace import RDF, DCTERMS, OWL, DC


# Load in ISO 639-2 language codes -- mapping from 2 or 3-letter code to English name of language. 
# Needed because languge is given by a URI in the RDF file, and the end of the URI is the 3-letter code.
LANGUAGES_FILE = './AmericanStories/src/utils/iso_639_2_languages.json'
with open(LANGUAGES_FILE, 'r') as f:
    LANGUAGE_CODES = json.load(f)

# Declare the FRBR namespace used in library of congress -- used for defining relationships between documents/collections
FRBR = Namespace('http://purl.org/vocab/frbr/core#')

# Declare the ORE namespace used in library of congress -- used for defining collections, aggregations, etc
ORE = Namespace('http://www.openarchives.org/ore/terms/')

# URL for the Chronicling America raw batches
CHRONICLING_AMERICA_DATA_STUB = 'https://chroniclingamerica.loc.gov/data/batches/'

# Template for metadata -- contains all the information we want to extract about a newspaper
BLANK_LCCN_METADATA = {'title': '',
                  'geonames_ids': [],
                  'dbpedia_ids': [],
                  'issn': '',
                  'lccn': '',
                  'start_year': '',
                  'end_year': '',
                  'languages': [],
                  'succeeds': [],
                  'successors': [],
                  'editions': [] }
                        
def get_lccn_metadata(lccn, edition_list = False):
    '''
    Get the metadata for a newspaper from its Library of Congress Control Number

    lccn: string
        Library of Congress Control Number of the newspaper
    edition_list: boolean
        Whether to get the list of all available editions of the newspaper
    '''
    # Set up a blank metadata dictionary
    metadata = deepcopy(BLANK_LCCN_METADATA)
    metadata['lccn'] = lccn

    # Get the RDF file for the newspaper
    g = Graph()
    g.parse(f'https://chroniclingamerica.loc.gov/lccn/{lccn}.rdf')
    paper = URIRef(f'https://chroniclingamerica.loc.gov/lccn/{lccn}')

    # Parse the RDF file to get the metadata
    for s, p, o in g.triples((None, DCTERMS['title'], None)):
        metadata['title'] = str(o)
    for s, p, o in g.triples((None, DCTERMS['coverage'], None)):
        if 'geonames' in o:
            metadata['geonames_ids'].append(o.split('/')[-2])
        elif 'dbpedia' in o:
            metadata['dbpedia_ids'].append(o.split('/')[-1])
    for s, p, o in g.triples((None, OWL['sameAs'], None)):
        if 'issn' in o:
            metadata['issn'] = o.split(':')[-1]
    for s, p, o in g.triples((None, FRBR['successorOf'], None)):
        metadata['succeeds'].append(o.split('/')[-1].split('#')[0])
    for s, p, o in g.triples((None, FRBR['successor'], None)):
        metadata['successors'].append(o.split('/')[-1].split('#')[0])
    for s, p, o in g.triples((None, DCTERMS['language'], None)):
        metadata['languages'].append(LANGUAGE_CODES[o.split('/')[-1]]['english'][0])
    for s, p, o in g.triples((None, DCTERMS['date'], None)):
        dates = o.split('>')[-1].split('<')[0].split('/')
        if len(dates) != 2:
            continue
        elif any(len(date) != 4 or not all([c.isdigit() or c == '?' for c in date]) for date in dates):
            continue
        else:
            metadata['start_year'] = dates[0]
            metadata['end_year'] = dates[1]
    if edition_list:
        for s, p, o in g.triples((None, ORE['aggregates'], None)):
            if '#issue' in o:
                metadata['editions'].append(o.split('#')[0].split(lccn)[1][1:])
    
    return metadata
    
def get_metadatas_from_lccn_list(lccn_list, edition_list = False):
    metadatas = {}
    for lccn in lccn_list:
        metadatas[lccn] = get_lccn_metadata(lccn, edition_list = edition_list)
    return metadatas

BLANK_EDITION_METADATA = {'lccn': '',
                          'edition': '',
                          'date': '',
                          'pages': [] }

def get_edition_metadata(lccn, edition):
    '''
    Get the metadata for a specific edition of a newspaper

    lccn: string
        Library of Congress Control Number of the newspaper
    edition: string
        Edition of the newspaper, in the format "YYYY-MM-DD/ed-NN"    
    '''
    
    # Validate the inputs
    if len(edition) == 10:
        edition = f'{edition[:4]}-{edition[4:6]}-{edition[6:8]}/ed-{edition[8:]}'

    if not re.match(r'\d{4}-\d{2}-\d{2}/ed-\d{1,2}', edition):
        raise ValueError('Edition must be in format "YYYY-MM-DD/ed-NN"')

    metadata = deepcopy(BLANK_EDITION_METADATA)
    metadata['lccn'] = lccn
    metadata['edition'] = edition.split('/')[-1]
    metadata['date'] = edition.split('/')[0]

    # Get the RDF file for the edition
    g = Graph()
    try:
        g.parse(f'https://chroniclingamerica.loc.gov/lccn/{lccn}/{edition}.rdf')
    except urllib.error.HTTPError:
        print(f'No RDF file found for https://chroniclingamerica.loc.gov/lccn/{lccn}/{edition}.rdf')
        return metadata
    
    # Parse the RDF file to get the pages
    for s, p, o in g.triples((None, ORE['aggregates'], None)):
        if '#page' in o:
            metadata['pages'].append(o.split('#')[0].split('/')[-1])

    return metadata

def get_edition_metadatas_from_list(lccn, edition_list):
    metadatas = {}
    for edition in edition_list:
        metadatas[edition] = get_edition_metadata(lccn, edition)
    return metadatas

def parse_loc_data_location(o):
    '''
    Parses out the location of the jp2 image of a scan from the rdf file object, 
    which will look like:
    https://chroniclingamerica.loc.gov/iiif/2/dlc_jamaica_ver01%2Fdata%2Fsn83030214%2F00175041217%2F1905011501%2F0318.jp2/full/200,/0/default.jpg
    '''
    return CHRONICLING_AMERICA_DATA_STUB + o.split('.jp2')[0].split('/')[-1].replace('%2F', '/') + '.jp2'

BLANK_SCAN_METADATA = {'lccn': '',
                       'edition': '',
                       'date': '',
                       'page': '',
                       'jp2_url': '',
                       'ocr_text_url': '',
                       'ocr_xml_url': '',
                       'raw_data_loc': '' }

def get_scan_metadata(lccn, date, edition, page):
    '''
    Get the metadata for a specific edition of a newspaper

    lccn: string
        Library of Congress Control Number of the newspaper
    edition: string
        Edition of the newspaper, in the format "YYYY-MM-DD/ed-N"
    page: string
        Page of the newspaper, in the format "seq-N"
    ''' 
    if len(date) == 8:
        date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
    if len(edition) == 2:
        edition = f'ed-{edition}'
    if len(page) <=3:
        page = f'seq-{page}'

    # Validate the inputs
    if not re.match(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError('date must be in format "YYYY-MM-DD"')
    if not re.match(r'ed-\d+', edition):
        raise ValueError('Edition must be in format "ed-N"')
    if not re.match(r'seq-\d+', page):
        raise ValueError('Page must be in format "seq-N"')
    
    metadata = deepcopy(BLANK_SCAN_METADATA)
    metadata['lccn'] = lccn
    metadata['edition'] = edition
    metadata['date'] = date
    metadata['page'] = page.split('-')[-1]

    # Get the RDF file for the page
    g = Graph()
    g.parse(f'https://chroniclingamerica.loc.gov/lccn/{lccn}/{date}/{edition}/{page}.rdf')

    # Parse the RDF file to get the pages
    for s, p, o in g.triples((None, ORE['aggregates'], None)):
        if 'ocr.txt' in o:
            metadata['ocr_text_url'] = str(o)
        elif 'ocr.xml' in o:
            metadata['ocr_xml_url'] = str(o)
        elif f'{page}.jp2' in o:
            metadata['jp2_url'] = str(o)
        elif 'full/200' in o:
            metadata['raw_data_loc'] = parse_loc_data_location(o)

    return metadata


if __name__ == '__main__':
    '''
    Test: Get metadata from lccn
    '''
    # lccn = 'sn86069872'
    # lccn = 'sn83045377'
    # lccn = 'sn84027107'
    # lccn = 'sn89066801'
    # metadata = get_lccn_metadata(lccn, edition_list=False)
    # print(metadata)

    '''
    Test: Get metadata from edition
    '''
    # lccn = 'sn83045555'
    # edition = '1889-11-21/ed-1'
    # metadata = get_edition_metadata(lccn, edition)
    # print(metadata)
    
    '''Test: Get metadata from page'''
    # lccn = 'sn83030214'
    # date = '1905-01-15'
    # edition = 'ed-1'
    # page = 'seq-25'
    # metadata = get_scan_metadata(lccn, date, edition, page)
    # print(metadata)