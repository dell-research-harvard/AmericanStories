import os
from tqdm import tqdm


def extract_info_from_full_path_manifest(fpmrow):
    fpmrow = fpmrow.strip()
    try:
        info = fpmrow.split(",")
        if len(info) == 2:
            dbx_path, imageid = info
        else:
            imageid = info[-1]
            dbx_path = ",".join(info[:-1]).replace("\"", "")
            print(dbx_path)
    except Exception:
        print(fpmrow)
        exit(1)
    state, newspaper, monthyear, filend = dbx_path.split("/")
    month, day, year = filend.split("-")[-5:-2]
    edition = f"{newspaper}-{month}-{day}-{year}"
    return imageid, os.path.join("organized_pdfs", dbx_path), edition, year


def write_na_ingress_files(dbxinfo, save_dir, size=50, max_ims=200): 

    # get years
    years = sorted(list(set([ed.split('-')[-1] for _, ed in dbxinfo])))
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
        
        year_keys = [yk for yk in list(edition_dict.keys()) if yk.split('-')[-1] == year]
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


def ingest_imageid_audit(audit_path):

    with open(audit_path) as f:
        audited_imageids = [line.strip() for line in tqdm(f)]

    print(f"Number of proc images: {len(audited_imageids)}")

    return audited_imageids


def generate_batching_info(full_path_manifest_path, audited_imageids, start_year=1967, end_year=1972):

    dbx_info = []
    with open(full_path_manifest_path) as f:
        next(f)
        for line in tqdm(f):
            imageid, dbx_path, edition, year = extract_info_from_full_path_manifest(line)
            if (start_year <= int(year) <= end_year):
                if not (imageid in audited_imageids):
                    dbx_info.append((dbx_path, edition))

    return dbx_info


if __name__ == '__main__':

    root_path = "/Users/jscarlson/Downloads"
    full_path_manifest_path = os.path.join(root_path, "full_path_manifest.csv")
    detailed_manifest_path = os.path.join(root_path, "detailed_on_remote_manifest.csv")
    imageid_audit_path = os.path.join(root_path, "dbx_audit_results_all_67_72/all_proc_imageids.txt")
    save_dir = os.path.join(root_path, "newspaper_archive_67_72_remaining")
    audited_imageids = ingest_imageid_audit(imageid_audit_path)
    audited_imageids = set(audited_imageids)
    dbxinfo = generate_batching_info(full_path_manifest_path, audited_imageids)
    write_na_ingress_files(dbxinfo, save_dir, size=50, max_ims=200)
