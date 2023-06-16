import requests
import json
import re
import pandas as pd
import os
from tqdm import tqdm

from ca_metadata_utils import get_lccn_metadata, get_edition_metadatas_from_list, get_scan_metadata

def get_lccn_list():
    '''
    Get the list of all Library of Congress Control Numbers (LCCNs) for newspapers in Chronicling America
    This is meant to be comprehensive, as long as the https://chroniclingamerica.loc.gov/newspapers.txt
    file stays up to date
    '''

    session = requests.Session()
    response = session.get('https://chroniclingamerica.loc.gov/newspapers.txt')

    # Convert response to a dataframe
    df = pd.DataFrame([[s.strip() for s in x.split('|')] for x in response.text.split('\n')[2:]], 
                                    columns = [x.strip() for x in response.text.split('\n')[0].split('|')])

    # Drop rows without an LCCN
    df = df[~df['LCCN'].isna()]
    
    return df['LCCN'].unique().tolist()

def get_list_on_date(date):
    '''
    Get the list of all Library of Congress Control Numbers (LCCNs) for newspapers in Chronicling America

    date: string
        Date in the format YYYY-MM-DD
    '''
    lccns = get_lccn_list()
    scans = []

    # Get the metadata for each newspaper
    for lccn in tqdm(lccns):
        lccn_data = get_lccn_metadata(lccn, edition_list=True)

        # Get the editions for the date
        editions = [x for x in lccn_data['editions'] if x.startswith(date)]

        # Get the metadata for each edition
        edition_data = get_edition_metadatas_from_list(lccn, editions)

        # Get the scans for each edition
        for edition, edition_info in edition_data.items():
            for page in edition_info['pages']:
                scans.append(get_scan_metadata(lccn, date, edition_info['edition'], page))

    return scans




if __name__ == '__main__':
    '''
    Test get_lccn_list()
    '''
    # lccn_list = get_lccn_list()
    # print(lccn_list[:10])

    '''
    Test get_list_on_date()
    '''
    date = '1900-01-01'
    page_list = get_list_on_date(date)
    print(len(page_list))
    with open(f'../../pipeline_ingress/new_gen_test/{date}.json', 'w') as f:
        json.dump(page_list, f, indent=4)



