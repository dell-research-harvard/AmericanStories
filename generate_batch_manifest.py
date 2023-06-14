'''
Generate a set of batch manifest files for a given chronicling america batch

Whichever batch is entered (must be a valid batch name for https://chroniclingamerica.loc.gov/data/batches/),
this script will generate a set of manifest files, each containing 100 scans from the batch. The scans are compiled in 
orde of their appearence in the batch manifest files, whcih is sorted by lccn and then date. Batch manifest files are found in
https://chroniclingamerica.loc.gov/data/batches/{batch}/data/batch.xml

Batches manifests are saved to a file named /{batch}, with manifests named manifest_0.txt, manifest_1.txt, etc.
There is also a file listing any editions that could not be retrieved from the batch manifest.
'''

import os
import requests
import argparse
from tqdm import tqdm
import re
import xml.etree.ElementTree as ET
from io import StringIO


BASE_URL = 'https://chroniclingamerica.loc.gov/data/batches/'

def get_edition_metadata(root_url):
    # get file list from url
    r = requests.get(root_url)
    xml = r.text

    # use regex to find a string in the text with at least six numbers then ".xml"
    try:
        manifest = re.search(r'<a href="[0-9]{6,}.xml">.*</a>', xml).group(0)
    except AttributeError:
        print(f'No manifest file found for {root_url}')
        return {}
        # raise AttributeError('No manifest file found')
    
    metadata = {'paper_name': '',
                'paper_date': '',
                'paper_city': '',
                'paper_state': '',
                'paper_url': root_url,
                'lccn': '',
                'volume': '',
                'issue': '',
                'edition': '',
                'pages': {} }
    
    manifest_file = manifest.split('>')[1].split("<")[0]

    # get xml file
    r = requests.get(root_url + '/' + manifest_file)
    print(root_url + '/' + manifest_file)
    xml = r.text

    root = ET.fromstring(xml)

    namespaces = dict([
         node for _, node in ET.iterparse(
             StringIO(xml), events=['start-ns']
         )
    ])
    paper_info = root.get('LABEL')
    
    try:
        metadata['paper_name'] = paper_info.split(' (')[0]
        metadata['paper_date'] = paper_info.split(' (')[1].split(')')[1][2:]
        metadata['paper_city'] = paper_info.split(' (')[1].split(')')[0].split(', ')[0]
        metadata['paper_state'] = paper_info.split(' (')[1].split(')')[0].split(', ')[1]
    except IndexError:
        try:
            metadata['paper_name'] = paper_info.split(' [')[0]
            metadata['paper_date'] = paper_info.split(' [')[1].split(']')[1][2:]
            metadata['paper_city'] = paper_info.split(' [')[1].split(']')[0].split(', ')[0]
            metadata['paper_state'] = paper_info.split(' [')[1].split(']')[0].split(', ')[1] 
        except IndexError:
            try:
                metadata['paper_name'] = paper_info.split(', ')[0]
                metadata['paper_date'] = paper_info.split(', ')[1]
            except IndexError:
                metadata['paper_name'] = paper_info

    if 'MODS' in namespaces:
        mods_str = 'MODS'
    else:
        mods_str = 'mods'
    print('Going hunting')
    for dmd in root.findall('dmdSec', namespaces):
        if dmd.get('ID') == 'issueModsBib':
            for mdwrap in dmd.findall('mdWrap', namespaces):
                for xml_data in mdwrap.findall('xmlData', namespaces):
                    try:
                        for mods in xml_data.findall('mods:mods', namespaces):
                            for item in mods.findall('mods:relatedItem', namespaces):
                                for identifier in item.findall('mods:identifier', namespaces):
                                    if identifier.get('type') == 'lccn':
                                        metadata['lccn'] = identifier.text
                                for part in item.findall(f'mods:part', namespaces):
                                    for detail in part.findall('mods:detail', namespaces):
                                        if detail.get('type') == 'volume':
                                            for number in detail.findall('mods:number', namespaces):
                                                metadata['volume'] = number.text
                                        elif detail.get('type') == 'issue':
                                            for number in detail.findall('mods:number', namespaces):
                                                metadata['issue'] = number.text
                                        elif detail.get('type') == 'edition':
                                            for number in detail.findall('mods:number', namespaces):
                                                metadata['edition'] = number.text
                    except SyntaxError:
                        pass

        elif dmd.get('ID').startswith('pageMods'):
            print('Have pages')
            page_metadata = {'page_number': '',
                             'page_url': '',
                             'page_ocr': '',
                             'reel_number': '',
                             'reel_sequence_number': ''}
            for mdwrap in dmd.findall('mdWrap', namespaces):
                for xml_data in mdwrap.findall('xmlData', namespaces):
                    try:
                        for mods in xml_data.findall(f'{mods_str}:mods', namespaces):
                            for part in mods.findall(f'{mods_str}:part', namespaces):
                                for extent in part.findall(f'{mods_str}:extent', namespaces):
                                    for start in extent.findall(f'{mods_str}:start', namespaces):
                                        page_metadata['page_number'] = start.text
                            for item in mods.findall(f'{mods_str}:relatedItem', namespaces):
                                    for identifier in item.findall(f'{mods_str}:identifier', namespaces):
                                        if identifier.get('type') == 'reel number':
                                            page_metadata['reel_number'] = identifier.text
                                        elif identifier.get('type') == 'reel sequence number':
                                            page_metadata['reel_sequence_number'] = identifier.text
                                            print(page_metadata['reel_sequence_number'])
                    except SyntaxError:
                        pass
            
            if len(page_metadata['reel_sequence_number']) < 4:
                file_number = '0' * (4 - len(page_metadata['reel_sequence_number'])) + page_metadata['reel_sequence_number']
            else:
                file_number = page_metadata['reel_sequence_number']
            page_metadata['page_url'] = root_url + file_number + '.jp2'
            page_metadata['page_ocr'] = root_url + file_number + '.xml'

            metadata['pages'][page_metadata['page_number']] = page_metadata

    return metadata


def get_list_for_batch(batch):
    batch_url = BASE_URL + batch + '/data/'
    batch_manifest = requests.get(batch_url + 'batch.xml').text
    if '404 Not Found' in batch_manifest:
        batch_manifest = requests.get(batch_url + 'BATCH.xml').text
        if '404 Not Found' in batch_manifest:
            batch_manifest = requests.get(batch_url + 'batchfile.xml').text
            if '404 Not Found' in batch_manifest:
                batch_manifest = requests.get(batch_url + 'batch_1.xml').text
                if '404 Not Found' in batch_manifest:
                    batch_manifest = requests.get(batch_url + 'BATCH_1.xml').text

    batch_editions = []
    for line in batch_manifest.split('\n'):

        if 'issue' not in line:
            continue
        elif len(line) < 20:
            continue

        try:
            fp_int = '/'.join(line.split('./')[1].split('.xml')[0].split('/')[:-1])
        except IndexError as e:
            try:
                fp_int = '/'.join(line.split('">')[1].split('.xml')[0].split('/')[:-1])
            except IndexError as e:
                print(line)
                raise e
                    
        if 'sn' != fp_int[:2] and 'SN' != fp_int[:2]:
            if len(fp_int.split('/')) == 3:
                pass
            else:
                fp_int = '/'.join(fp_int.split('/')[1:])
        elif len(fp_int.split('/')) == 2:
            print(fp_int)
                    
        batch_editions.append(batch_url + fp_int)
        if len(batch_editions) > 10:
            break

    
    print('Found {} editions in batch {}'.format(len(batch_editions), batch))
    all_scans = []
    error_scans = []
    for edition in tqdm(batch_editions):
        num_repeats = 0
        while True:
            try:
                ed_metadata = get_edition_metadata(edition + '/')
                break
            except Exception as e:
                num_repeats += 1
                if num_repeats > 5:
                    ed_metadata = {}
                error_scans.append(edition)
                
        try:
            for page in ed_metadata['pages'].keys():
                try:
                    all_scans.append(ed_metadata['pages'][page]['page_url'])
                except KeyError as e:
                    print('No scans found for edition {}'.format(edition))
        except KeyError as e:
            print('No scans found for edition {}'.format(edition))
            continue


    os.makedirs(f'./{batch}', exist_ok=True)
    scans_to_write = []
        
        
    for i, scan in enumerate(all_scans):
        scans_to_write.append(scan)

        if len(scans_to_write) == 100:
            print(f'Writing {len(scans_to_write)} scans to manifest')
            print(f'./{batch}/manifest_{str(i // 100)}.txt')
            with open(f'./{batch}/manifest_{str(i // 100)}.txt', 'w') as f:
                f.write('\n'.join(scans_to_write))
            scans_to_write = []
    
    print('Writing {} scans to manifest'.format(len(scans_to_write)))
    with open(f'./{batch}/manifest_{str((i + 100) // 100)}.txt', 'w') as f:
        f.write('\n'.join(scans_to_write))

    with open(f'./{batch}/error_manifest.txt', 'w') as f:
        f.write('\n'.join(error_scans))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=str, required=True)

    args = parser.parse_args()

    get_list_for_batch(args.batch)
