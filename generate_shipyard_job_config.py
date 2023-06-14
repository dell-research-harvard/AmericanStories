import yaml
import argparse
import os
import numpy as np
import subprocess
import pandas as pd
from itertools import compress
import time


class literal(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='>')


def create_job_yaml(
        ingress_text_files, 
        ingress_dropbox_path, 
        egress_dropbox_path, 
        save_path, 
        jobid, 
        priority,
        vm_total, 
        container_ver,
        data_source="newspaper_archive",
        failed_subset_from_csv=None,
        manual_selection=None
    ):

    # start up params
    all_tasks = []
    sleep_seconds = -30
    remote_id = 1
    VM_TOTAL = vm_total

    # failed subset list
    if failed_subset_from_csv:
        job_stats_df = pd.read_csv(failed_subset_from_csv)
        task_ids = job_stats_df['Task id'].tolist()
        prev_failed_task_ids = list(
            compress(
                job_stats_df['Task id'].tolist(), 
                [c != 0 for c in job_stats_df['Exit code'].tolist()]
            )
        )
        print(f'Prev failed tasks: {prev_failed_task_ids}')
        other_term_task_ids = [id for id in range(len(ingress_text_files)) if id not in task_ids]
        print(f'Other term tasks: {other_term_task_ids}')
        prev_failed_task_ids += other_term_task_ids
        print(f'All failed tasks: {prev_failed_task_ids}')
        ft_idx = -1

    if manual_selection:
        ingress_text_files = []
        with open(manual_selection) as f:
            for line in f:
                ingress_text_files.append(line.strip() + '.txt')

    for idx, text_file in enumerate(ingress_text_files):

        # if in failed subset mode, only recreate tasks that prev failed
        if failed_subset_from_csv:
            if idx not in prev_failed_task_ids: 
                continue
            else:
                ft_idx += 1
                idx = ft_idx

        # stagger start first VMs
        if idx < VM_TOTAL:
            if idx % int(VM_TOTAL / 10) == 0:
                sleep_seconds += 30
        else:
            sleep_seconds = 0

        # every 10 VMs share a remote (Dropbox app)
        if idx % 20 == 0:
            remote_id = 1
        else:
            remote_id += 1
        
        # task command
        task_command = literal(
            "/bin/bash -c $'"
            "export OMP_THREAD_LIMIT=2 && "
            "rm -rf /img2txt_pipeline/pdf && "
            "rm -rf /img2txt_pipeline/mb && "
            f"sleep {sleep_seconds} && "
            f"echo {idx} && "
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest --tpslimit 10 pipelineapp{remote_id}:{ingress_dropbox_path}/{text_file} /img2txt_pipeline/mb && "
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest --tpslimit 10 --include-from /img2txt_pipeline/mb/{text_file} pipelineapp{remote_id}: /img2txt_pipeline/pdf && "
            "python3.8 /img2txt_pipeline/run_img2txt_pipeline.py " 
            "--pdf_source_path /img2txt_pipeline/pdf " 
            "--img_save_path /img2txt_pipeline/jpg " 
            "--output_save_path /img2txt_pipeline/out "
            "--config_path /img2txt_pipeline/model_files/config.yaml "
            "--checkpoint_path /img2txt_pipeline/model_files/model_final.pth "  
            "--label_map_path /img2txt_pipeline/label_maps/label_map.json " 
            "--tessdata_path /usr/local/share/tessdata "
            f"--data_source {data_source} "
            "--classifier_head_checkpoint_path /img2txt_pipeline/model_files/full_edition_scan_classifier_head.pth "
            "--filter_duplicates --ocr_padding && "
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest --dropbox-batch-mode off --tpslimit 10 /img2txt_pipeline/jpg/error_table.csv pipelineapp{remote_id}:{egress_dropbox_path}/{text_file.split('.')[0]} && " 
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest --dropbox-batch-mode off --tpslimit 10 /img2txt_pipeline/out pipelineapp{remote_id}:{egress_dropbox_path}/{text_file.split('.')[0]}"
            "'"
        )

        # first VMs have no task dependencies; otherwise, task dependencies exist to enforce order
        if idx < VM_TOTAL:
            task = {
                'id': literal(str(idx).zfill(6)),
                'command': task_command,
                'docker_image': f'img2txtregistry.azurecr.io/img2txt_pipeline_container:v{container_ver}'
            }
        else:
            task = {
                'id': literal(str(idx).zfill(6)),
                'depends_on_range': [idx % VM_TOTAL, idx % VM_TOTAL],
                'command': task_command,
                'docker_image': f'img2txtregistry.azurecr.io/img2txt_pipeline_container:v{container_ver}'
            }
        
        # append task
        all_tasks.append(task)

    # create single job
    job = {'job_specifications': [{
        'id': 'err' + jobid if failed_subset_from_csv else jobid,
        'auto_complete': True,
        'priority': priority,
        'tasks': all_tasks
    }]}

    # convert python obj to yaml file, write
    yaml.add_representer(literal, literal_presenter)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'jobs.yaml'), 'w') as f:
        job_yaml = yaml.dump(job, f)

    return job_yaml


def create_job(
        byear,
        local_batch_path,
        env,
        yaml_save_path = f"/Users/jscarlson/config",
        manual_selection = None,
        failed_subset_from_csv = None,
        priority = 999,
        vm_total = 250,
        container_ver = 17,
        data_source = "loc",
    ):

    # NA remaining
    text_file_dir = f"/Users/jscarlson/Downloads/{local_batch_path}/{byear}"
    dropbox_ingress_path = f"pipeline_ingress/na_remaining_{byear}"
    dropbox_egress_path = f"pipeline_egress/na_remaining_{byear}"
    failed_subset_from_csv = f"/Users/jscarlson/Downloads/narem{byear}.csv" if failed_subset_from_csv else None
    jobid = f"narem{byear}"

    # NA gaps
    """
    text_file_dir = f"/home/jscarlson/Downloads/{local_batch_path}/{byear}"
    dropbox_ingress_path = f"pipeline_ingress/na_gap_{byear}"
    dropbox_egress_path = f"pipeline_egress/na_gap_{byear}"
    failed_subset_from_csv = f"/home/jscarlson/Downloads/na_gap_{byear}.csv" if failed_subset_from_csv else None
    jobid = f"nagap{byear}"
    """

    # LOC
    """
    text_file_dir = f"/home/jscarlson/Downloads/{local_batch_path}/{byear}"
    dropbox_ingress_path = f"pipeline_ingress/loc_{byear}_fp"
    dropbox_egress_path = f"pipeline_egress/loc_{byear}_fp"
    failed_subset_from_csv = f"/home/jscarlson/Downloads/locjob{byear}.csv" if failed_subset_from_csv else None
    jobid = f"locjob{byear}"
    """

    # Standard (Front Page, Editorials) NA
    """
    text_file_dir = f'/home/jscarlson/Downloads/minibatches{byear}_50'
    dropbox_ingress_path = f'pipeline_ingress/all{byear}_size50_marchManifest'
    dropbox_egress_path = f'pipeline_egress/all{byear}_size50_marchManifest'
    yaml_save_path = '/home/jscarlson/config_fullscale/'
    failed_subset_from_csv = None # '/home/jscarlson/Downloads/img2txtjob1960.csv'
    manual_selection = None
    jobid = f'img2txtjob{byear}'
    priority = 999
    vm_total = 250
    container_ver = 17
    """

    # Full Edition NA
    """
    text_file_dir = f'/content/drive/MyDrive/all_scans_dbx_text_files_{byear}'
    dropbox_ingress_path = f'pipeline_ingress/all_scans_dbx_text_files_{byear}'
    dropbox_egress_path = f'pipeline_egress/all_scans_dbx_text_files_{byear}'
    yaml_save_path = '/content/drive/MyDrive/config_remote'
    failed_subset_from_csv = None # '/content/fulleditionjob1971.csv'
    manual_selection = None
    jobid = f'fulleditionjob{byera}'
    priority = 999
    vm_total = 250
    container_ver = 16
    """

    # DUP
    """
    text_file_dir = f'/media/jscarlson/ADATASE800/pipeline_redo_ingress/all{byear}_size50_marchManifest'
    dropbox_ingress_path = f'pipeline_redo_ingress/all{byear}_size50_marchManifest'
    dropbox_egress_path = f'pipeline_redo_egress/all{byear}_size50_marchManifest'
    yaml_save_path = '/home/jscarlson/config_dup/'
    failed_subset_from_csv = None # f'/home/jscarlson/Downloads/dupimg2txtjob{byear}.csv'
    manual_selection = None
    jobid = f'errordupimg2txtjob{byear}' # f'errordupimg2txtjob{byear}'
    priority = int(input("Priority: "))
    """

    job_yaml = create_job_yaml(
        os.listdir(text_file_dir), 
        ingress_dropbox_path=dropbox_ingress_path,
        egress_dropbox_path=dropbox_egress_path, 
        save_path=yaml_save_path,
        jobid=jobid,
        priority=priority,
        failed_subset_from_csv=failed_subset_from_csv,
        manual_selection=manual_selection,
        vm_total=vm_total,
        container_ver=container_ver,
        data_source=data_source
    )

    if not failed_subset_from_csv and not manual_selection:
        print(f"***\n{y}\n***")
        subprocess.run(["rclone", "copy", "-P", text_file_dir, "vn:" + dropbox_ingress_path, "--transfers", "8", "--checkers", "8"])
        time.sleep(10)
        subprocess.run(["/Users/jscarlson/github_repos/batch-shipyard/shipyard", "jobs", "add"], env=env)
        time.sleep(30)


if __name__ == '__main__':

    env = os.environ.copy()   # Make a copy of the current environment
    # env['SHIPYARD_CONFIGDIR'] = '/Users/jscarlson/config'
    subprocess.run([r"C:\Users\bryan\Documents\NBER\batch-shipyard.cmd", "pool", "add"], env=env)
    time.sleep(60)

    for idx, y in enumerate(range(1969, 1970)):
        create_job(y, local_batch_path="newspaper_archive_67_72_remaining", 
            priority = 999 - (10*idx), data_source="newspaper_archive", env=env)
    
    # for idx, y in enumerate(range(1850, 1860)):
    #    create_job(y, local_batch_path="loc_1850_1859_fp_batches_by_year_correct", priority = 999 - (10*idx))

    # create_job(1893, vm_total=20, failed_subset_from_csv=True)