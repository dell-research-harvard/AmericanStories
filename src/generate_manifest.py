'''
Various functions for scraping chronicling america metadata and generating batch-level manifests,
lists of scans in each batch. 

An example batch can be seen at: https://chroniclingamerica.loc.gov/data/batches/ak_albatross_ver01/

Commented descriptions in the main function describe the various options for generating manifests
'''

import requests
import re
from tqdm import tqdm
import xml.etree.ElementTree as ET
from io import StringIO
import os
import json
# from utils.ca_metadata_utils import get_lccn_metadata, get_edition_metadata, get_scan_metadata
# from rdflib import Graph, Namespace, URIRef
# from rdflib.namespace import RDF, DCTERMS, OWL, DC


def parse_ocr_from_xml(url):
    r = requests.get(url)
    xml = r.text

    root = ET.fromstring(xml)
    xml_namespace = root.tag.split('}')[0] + '}'
    xml_str = ''

    for layout in root.findall(f'{xml_namespace}Layout'):
        for page in layout.findall(f'{xml_namespace}Page'):
            for printspace in page.findall(f"{xml_namespace}PrintSpace"):
                for textblock in printspace.findall(f"{xml_namespace}TextBlock"):
                    for textline in textblock.findall(f"{xml_namespace}TextLine"):
                        line = ' '.join([s.get(f'CONTENT') for s in textline.findall(f'{xml_namespace}String')])
                        xml_str += line + '\n'

    return xml_str
    
def get_file_list_from_url(url):
    # get file list from url
    r = requests.get(url)
    xml = r.text

    # use regex to find a string in the text with at least six numbers then ".xml"
    try:
        manifest = re.search(r'<a href="[0-9]{6,}.xml">.*</a>', xml).group(0)
    except AttributeError:
        print(f'No manifest file found for {url}')
        return []
    
    manifest_file = manifest.split('>')[1].split("<")[0]

    # get xml file
    r = requests.get(url + '/' + manifest_file)
    xml = r.text

    # get list of files in filesec section of xml
    filesec = xml.split('<fileSec>')[1].split('</fileSec>')[0]

    # get file names from the filesec
    files = [f.split('</file>')[0] for f in filesec.split('USE="service"')[1:]]
    filenames = [f.split('xlink:href="')[1].split('"')[0] for f in files]

    # if file name starts with "./", remove those characters
    filenames = [f[2:] if f[:2] == './' else f for f in filenames]

    return filenames

def find_page_number_from_filename(filename):
    root_url = '/'.join(filename.split('/')[:-1]) + '/'
    r = requests.get(root_url)
    xml = r.text
    manifest = re.search(r'<a href="[0-9]{6,}.xml">.*</a>', xml).group(0)
    manifest_file = manifest.split('>')[1].split("<")[0]

    r = requests.get(root_url + manifest_file)
    xml = r.text

    reel_number = filename.split('/')[-1].split('.')[0]
    reel_number = str(int(reel_number))

    root = ET.fromstring(xml)
    namespaces = dict([
         node for _, node in ET.iterparse(
             StringIO(xml), events=['start-ns']
         )
    ])

    for dmd in root.findall('dmdSec', namespaces):
        if dmd.get('ID').startswith('pageMods'):
            page_metadata = {'page_number': '',
                        'page_url': '',
                        'page_ocr': '',
                        'reel_number': '',
                        'reel_sequence_number': ''}
            for mdwrap in dmd.findall('mdWrap', namespaces):
                for xml_data in mdwrap.findall('xmlData', namespaces):
                    if 'mods' in namespaces.keys():
                        mods_tag = 'mods'
                    elif 'MODS' in namespaces.keys():
                        mods_tag = 'MODS'
                    else:
                        raise KeyError('No mods namespace found')

                    for mods in xml_data.findall(f'{mods_tag}:mods', namespaces):
                        for part in mods.findall(f'{mods_tag}:part', namespaces):
                            for extent in part.findall(f'{mods_tag}:extent', namespaces):
                                for start in extent.findall(f'{mods_tag}:start', namespaces):
                                    page_metadata['page_number'] = start.text
                        for item in mods.findall(f'{mods_tag}:relatedItem', namespaces):
                            for identifier in item.findall(f'{mods_tag}:identifier', namespaces):
                                if identifier.get('type') == 'reel number':
                                    page_metadata['reel_number'] = identifier.text
                                elif identifier.get('type') == 'reel sequence number':
                                    page_metadata['reel_sequence_number'] = identifier.text
            
            if page_metadata['reel_sequence_number'] == reel_number:
                return page_metadata['page_number']



def get_metadata(root_url):
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
            page_metadata = {'page_number': '',
                             'page_url': '',
                             'page_ocr': '',
                             'reel_number': '',
                             'reel_sequence_number': ''}
            for mdwrap in dmd.findall('mdWrap', namespaces):
                for xml_data in mdwrap.findall('xmlData', namespaces):
                    try:
                        for mods in xml_data.findall('mods:mods', namespaces):
                            for part in mods.findall('mods:part', namespaces):
                                for extent in part.findall('mods:extent', namespaces):
                                    for start in extent.findall('mods:start', namespaces):
                                        page_metadata['page_number'] = start.text
                            for item in mods.findall('mods:relatedItem', namespaces):
                                    for identifier in item.findall('mods:identifier', namespaces):
                                        if identifier.get('type') == 'reel number':
                                            page_metadata['reel_number'] = identifier.text
                                        elif identifier.get('type') == 'reel sequence number':
                                            page_metadata['reel_sequence_number'] = identifier.text
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


def get_list_on_date(date='1896-03-01'):
    print(date)
    base_url = 'https://chroniclingamerica.loc.gov/data/batches/'
    r = requests.get(base_url)
    xml = r.text
    
    all_batches = re.findall(r'<a href=".*">.*</a>', xml)[1:-1]
    batches = [b.split('>')[1].split('<')[0] for b in all_batches]
    edition_urls = []
    for batch in tqdm(batches):
        batch_url = base_url + batch + 'data/'
        batch_manifest = requests.get(batch_url + '/batch.xml').text
        for line in batch_manifest.split('\n')[2:]:
            if f'issueDate="{date}"' in line:
                try:
                    try:
                        fp_int = '/'.join(line.split('./')[1].split('.xml')[0].split('/')[:-1])
                    except IndexError as e:
                        fp_int = '/'.join(line.split('">')[1].split('.xml')[0].split('/')[:-1])
                    
                    if 'sn' != fp_int[:2]:
                        fp_int = '/'.join(fp_int.split('/')[1:])
                    
                    edition_urls.append(batch_url + fp_int)
    
                except IndexError as e:
                    print(line)
                    
    os.chdir(r'C:\Users\bryan\Documents\NBER\chronicling_america\day_manifests')
    error_metadatas = []
    ed_metadatas = []
    for ed in tqdm(edition_urls):
        try:
            ed_metadata = get_metadata(ed + '/')
            ed_metadatas.append(ed_metadata)
        except AttributeError:
            error_metadatas.append(ed)
            continue
        except SyntaxError:
            error_metadatas.append(ed)
            continue

    with open(f'./manifest_{date}.json', 'w') as f:
        json.dump(ed_metadatas, f, indent=4)
    with open(f'./error_manifest_{date}.json', 'w') as f:
        json.dump(error_metadatas, f, indent=4)

def get_lists_on_dates(dates):
    base_url = 'https://chroniclingamerica.loc.gov/data/batches/'
    r = requests.get(base_url)
    xml = r.text
    date_editions = {date: [] for date in dates}
    print(date_editions)
    all_batches = re.findall(r'<a href=".*">.*</a>', xml)[1:-1]
    batches = [b.split('>')[1].split('<')[0] for b in all_batches]
    for batch in tqdm(batches):
        batch_url = base_url + batch + 'data/'
        batch_manifest = requests.get(batch_url + '/batch.xml').text
        for line in batch_manifest.split('\n')[2:]:
            for date in dates:
                if f'issueDate="{date}"' in line:
                    try:
                        try:
                            fp_int = '/'.join(line.split('./')[1].split('.xml')[0].split('/')[:-1])
                        except IndexError as e:
                            fp_int = '/'.join(line.split('">')[1].split('.xml')[0].split('/')[:-1])
                        
                        if 'sn' != fp_int[:2]:
                            fp_int = '/'.join(fp_int.split('/')[1:])
                        
                        date_editions[date].append(batch_url + fp_int)
        
                    except IndexError as e:
                        print(line)

    print({date: len(date_editions[date]) for date in dates})
    os.chdir(r'C:\Users\bryan\Documents\NBER\chronicling_america\day_manifests')
    for date in dates:
        error_metadatas = []
        ed_metadatas = []
        for ed in tqdm(date_editions[date]):
            try:
                ed_metadata = get_metadata(ed + '/')
                ed_metadatas.append(ed_metadata)
            except AttributeError:
                error_metadatas.append(ed)
                continue
            except SyntaxError:
                error_metadatas.append(ed)
                continue

        with open(f'./manifest_{date}.json', 'w') as f:
            json.dump(ed_metadatas, f, indent=4)
        with open(f'./error_manifest_{date}.json', 'w') as f:
            json.dump(error_metadatas, f, indent=4)
    
def get_lists_on_years(years):
    base_url = 'https://chroniclingamerica.loc.gov/data/batches/'
    r = requests.get(base_url)
    xml = r.text
    year_editions = {year: [] for year in years}
    print(year_editions)
    all_batches = re.findall(r'<a href=".*">.*</a>', xml)[1:-1]
    batches = [b.split('>')[1].split('<')[0] for b in all_batches]

    for batch in tqdm(batches):
        batch_url = base_url + batch + 'data/'
        batch_manifest = requests.get(batch_url + '/batch.xml').text
        for line in batch_manifest.split('\n')[2:]:
            for year in years:
                if f'issueDate="{year}' in line:
                    try:
                        try:
                            fp_int = '/'.join(line.split('./')[1].split('.xml')[0].split('/')[:-1])
                        except IndexError as e:
                            fp_int = '/'.join(line.split('">')[1].split('.xml')[0].split('/')[:-1])
                        
                        if 'sn' != fp_int[:2]:
                            fp_int = '/'.join(fp_int.split('/')[1:])
                        
                        year_editions[year].append(batch_url + fp_int)
        
                    except IndexError as e:
                        print(line)

    print({year: len(year_editions[year]) for year in years})
    os.chdir(r'C:\Users\bryan\Documents\NBER\ca_manifests\full_years')
    for year in years:
        error_metadatas = []
        ed_metadatas = []
        for ed in tqdm(year_editions[year]):
            try:
                ed_metadata = get_metadata(ed + '/')
                ed_metadatas.append(ed_metadata)
            except AttributeError:
                error_metadatas.append(ed)
                continue
            except SyntaxError:
                error_metadatas.append(ed)
                continue

        with open(f'./manifest_{year}.json', 'w') as f:
            json.dump(ed_metadatas, f, indent=4)
        with open(f'./error_manifest_{year}.json', 'w') as f:
            json.dump(error_metadatas, f, indent=4)

def get_list_for_batch(batch):
    batch_url = base_url + batch + 'data/'
    batch_manifest = requests.get(batch_url + '/batch.xml').text
    batch_editions = []
    for line in batch_manifest.split('\n')[2:]:
        try:
            fp_int = '/'.join(line.split('./')[1].split('.xml')[0].split('/')[:-1])
        except IndexError as e:
            try:
                fp_int = '/'.join(line.split('">')[1].split('.xml')[0].split('/')[:-1])
            except IndexError as e:
                continue
                    
        if 'sn' != fp_int[:2]:
            fp_int = '/'.join(fp_int.split('/')[1:])
                    
        batch_editions.append(batch_url + fp_int)
    
    print('Found {} editions in batch {}'.format(len(batch_editions), batch))
    all_scans = []
    error_scans = []
    for edition in tqdm(batch_editions):
        num_repeats = 0
        while True:
            try:
                ed_metadata = get_metadata(edition + '/')
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

    
def get_lists_for_batches(start_batch, n_scans):
    base_url = 'https://chroniclingamerica.loc.gov/data/batches/'
    r = requests.get(base_url)
    xml = r.text
    all_batches = re.findall(r'<a href=".*">.*</a>', xml)[1:-1]
    batches = [b.split('>')[1].split('<')[0] for b in all_batches]
    os.chdir(r'C:\Users\bryan\Documents\NBER\chronicling_america\batch_manifests')
    total_added = 0
    for batch in batches[start_batch:]:
        batch_url = base_url + batch + 'data/'
        batch_manifest = requests.get(batch_url + '/batch.xml').text
        batch_editions = []
        for line in batch_manifest.split('\n')[2:]:
            try:
                fp_int = '/'.join(line.split('./')[1].split('.xml')[0].split('/')[:-1])
            except IndexError as e:
                try:
                    fp_int = '/'.join(line.split('">')[1].split('.xml')[0].split('/')[:-1])
                except IndexError as e:
                    continue
                        
            if 'sn' != fp_int[:2]:
                fp_int = '/'.join(fp_int.split('/')[1:])
                        
            batch_editions.append(batch_url + fp_int)
        
        print('Found {} editions in batch {}'.format(len(batch_editions), batch))
        all_scans = []
        error_scans = []
        for edition in tqdm(batch_editions):
            num_repeats = 0
            while True:
                try:
                    ed_metadata = get_metadata(edition + '/')
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

        total_added += len(all_scans)

        with open(f'./{batch}/error_manifest.txt', 'w') as f:
            f.write('\n'.join(error_scans))

    if total_added >= n_scans:
        print(f'All {n_scans} scans scraped')


if __name__ == '__main__':
    '''
    Test get_file_list_from_url
    '''
    # url = 'https://chroniclingamerica.loc.gov/data/batches/ak_albatross_ver01/data/sn84020657/0027952665A/1916030101/'
    # get_file_list_from_url(url)

    '''
    Test get_list_on_date
    '''
    # date_list = ['1916-01-01', '1916-02-01', '1916-03-01', '1916-04-01', '1916-05-01', '1916-06-01', '1916-07-01', '1916-08-01', '1916-09-01', '1916-10-01', '1916-11-01', '1916-12-01']
    # get_lists_on_dates(date_list)

    # with open(r'./urls_1916-03-01.txt', 'w') as f:
    #     for url in urls_date
    #         f.write(url + '\n')

    '''
    Test parse_ocr_from_xml
    '''
    # xml_url = 'https://chroniclingamerica.loc.gov/data/batches/ak_albatross_ver01/data/sn84020657/0027952665A/1916030101/0007.xml'
    # parse_ocr_from_xml(xml_url)

    '''
    Test get_metadata (old version)
    '''
    # metadata_url = 'https://chroniclingamerica.loc.gov/data/batches/mdu_denton_ver01/data/sn84026758/00279522515/1881102201/0131.jp2'
    # # metadata = get_metadata(metadata_url)
    # lccn = metadata_url.split('/')[-4]
    # year_ed = metadata_url.split('/')[-2]
    # print(get_lccn_metadata('sn84026758'))
    # print(get_edition_metadata('sn84026758', '1881102201'))
    # page_num = find_page_number_from_filename(metadata_url)
    # print(get_scan_metadata('sn84026758', '18811022', '01', page_num))
    # print(metadata)

    '''
    Run: Get all manifests for a list of dates
    '''
    # date_list = ['1846-03-01', '1856-03-01', '1866-03-01', '1876-03-01', '1886-03-01', '1896-03-01', '1906-03-01', '1916-03-01', '1926-03-01']
    # os.chdir(YOUR_DIR_HERE)
    # i = 0
    # for date in date_list:
    #     with open(f'./manifest_{date}.json', 'r') as f:
    #         data = json.load(f)
    #     scans = []
    #     for ed in data:
    #         scans.extend([ed['pages'][k]['page_url'] for k in ed['pages'].keys()])
    #     j = 0
    #     for scan in scans:
    #         while j + 20 < len(scans):
    #             with open(f'manifest_{i}.txt', 'w') as outfile:
    #                 outfile.write('\n'.join(scans[j: j+20]))
    #                 j += 20
    #                 i += 1

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
    
    # '''Test: Get metadata from page'''
    # lccn = 'sn83030214'
    # edition = '1905-01-15/ed-1'
    # page = 'seq-25'
    # metadata = get_scan_metadata(lccn, edition, page)
    # print(metadata)

    '''
    Get manifests for years
    '''
    # years = ['1914', '1915', '1916', '1917', '1918', '1919', '1920']
    # get_list_on_years(years)


    '''
    Get manifests for batches
    '''
    # get_lists_for_batches(20, 1000000)

    '''
    Get list of batches
    '''
    # base_url = 'https://chroniclingamerica.loc.gov/data/batches/'
    # r = requests.get(base_url)
    # xml = r.text
    # all_batches = re.findall(r'<a href=".*">.*</a>', xml)[1:-1]
    # batches = [b.split('>')[1].split('<')[0] for b in all_batches]
    # with open(YOUR_DIR_HERE, 'w') as outfile:
    #     outfile.write('\n'.join(batches))