import argparse
import csv
import os
from tqdm import tqdm


def parse_loc_csv(csvpath):
    hashcodes = []
    urlpaths = []
    header_seen = False
    with open(csvpath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if header_seen:
                if len(row) == 2:
                    hashcode, urlpath = row
                elif len(row) == 3:
                    _, hashcode, urlpath = row
                else:
                    raise Exception
                urlpaths.append(urlpath)
                hashcodes.append(hashcode)
            else:
                header_seen = True
    return hashcodes, urlpaths


def extract_loc_url_metadata(dbxpath):
    components = dbxpath.split('/')
    scandate = components[-2]
    newspapercode = components[-4]
    newspaperedition = newspapercode + '_' + scandate
    return dbxpath, newspaperedition


def get_loc_dbx_paths_and_metadata(urlpaths):
    dbxpaths = [x.replace("https://chroniclingamerica.loc.gov", "loc_scraper/scans") for x in urlpaths]
    dbxinfo = [extract_loc_url_metadata(dbxpath) for dbxpath in dbxpaths]
    return dbxinfo


def write_loc_ingress_files(dbxinfo, save_dir, size=50, max_ims=200): 

    # get years
    years = sorted(list(set([ed.split('_')[1][:4] for _, ed in dbxinfo])))
    
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_csv_path", help="")
    parser.add_argument("--save_dir", help="")
    args = parser.parse_args()

    loc_csv_path = "/media/jscarlson/ADATASE800/loc_scraper/queries/210716_1850-1859fp/urls.csv"
    save_dir = "/home/jscarlson/Downloads/loc_1850_1859_fp_batches_by_year_correct"

    _, urlpaths = parse_loc_csv(loc_csv_path)
    dbxinfo = get_loc_dbx_paths_and_metadata(urlpaths)
    write_loc_ingress_files(dbxinfo, save_dir)
