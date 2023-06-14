import os
from tqdm import tqdm
from datetime import datetime
import time


def create_dbx_file_path(fields):

    # e.g. organized_pdfs/north-carolina/graham-alamance-gleaner/Aug-1921/366929632-graham-alamance-gleaner-Aug-04-1921-p-7.pdf
    date = datetime.strptime(fields[5].split()[0], '%Y-%m-%d').strftime('%b-%d-%Y')
    monthyear = datetime.strptime(fields[5].split()[0], '%Y-%m-%d').strftime('%b-%Y')
    dbx_path = os.path.join(
        'organized_pdfs',
        fields[2],
        fields[3],
        monthyear,
        '-'.join([fields[0], fields[3], date, 'p', str(fields[-3]) + '.pdf'])
    )
    edition = '_'.join([fields[3], fields[5]]) 
    return dbx_path, edition


def write_na_ingress_files(dbxinfo, save_dir, size=50, max_ims=200): 

    # get years
    years = sorted(list(set([ed.split('_')[1][:4] for _, ed in dbxinfo])))
    print(years)
    
    # create edition dict
    edition_dict = {}
    for urlpath, edition in dbxinfo:
        if edition in edition_dict:
            edition_dict[edition].append(urlpath)
        else:
            edition_dict[edition] = [urlpath]
            
    print(sum(len(v) for k, v in edition_dict.items()))

    # make sure save dir exists
    os.makedirs(save_dir, exist_ok=True)
    
    # write
    for year in years:

        url_counter = 0
        urlpaths_to_write = []
        batch_counter = 0
        year_save_dir = os.path.join(save_dir, year)
        os.makedirs(year_save_dir, exist_ok=True)
        
        year_keys = [yk for yk in list(edition_dict.keys()) if yk.split('_')[1][:4] == year]
        for k in tqdm(year_keys):
            
            if len(urlpaths_to_write) >= max_ims or url_counter >= size:
                with open(os.path.join(year_save_dir, f"mini_batch_{batch_counter}.txt"), 'w') as f:
                    f.write("\n".join(urlpaths_to_write))
                urlpaths_to_write = edition_dict[k]
                url_counter = 1
                batch_counter += 1
            else:
                urlpaths_to_write.extend(edition_dict[k])
                url_counter += 1
        
        if len(urlpaths_to_write) > 0:
            with open(os.path.join(year_save_dir, f"mini_batch_{batch_counter}.txt"), 'w') as f:
                f.write("\n".join(urlpaths_to_write))


if __name__ == '__main__':

    merged_imageid_metadata_path = "/home/jscarlson/Downloads/local_merged_6772.csv"
    save_dir = "/home/jscarlson/Downloads/newspaper_archive_67_72_gaps"
    dbxinfo = []

    with open(merged_imageid_metadata_path) as f:
        for line in tqdm(f):
            if line.startswith("imageid"):
                continue
            csv_fields = line.split(',')
            if csv_fields[-1].startswith("missing"):
                dbxinfo.append(create_dbx_file_path(csv_fields))
            
    write_na_ingress_files(dbxinfo, save_dir, size=50, max_ims=200)

            
    