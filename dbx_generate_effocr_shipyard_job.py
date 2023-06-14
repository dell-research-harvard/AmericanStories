import yaml
import argparse
import os
import numpy as np
import subprocess
import pandas as pd
from itertools import compress
import time
import paramiko
import json
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
        batch_manifests, 
        save_path,
        batches,
        jobid, 
        priority,
        vm_total, 
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

    idx = 0
    for batch, batch_manifest in zip(batches, batch_manifests):
        for batch_idx, text_file in enumerate(batch_manifest):
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

            # every 20 VMs share a remote (Dropbox app)
            if idx % 20 == 0:
                remote_id = 1
            else:
                remote_id += 1
            
            # task command
            task_command = literal(
                "/bin/bash -c $'"
                f"sleep {sleep_seconds} && "
                f"echo {idx} && "
                "rm -rf /img2txt_pipeline/end-to-end-pipeline && "
                "rm -rf /img2txt_pipeline/pdf && "
                "rm -rf /img2txt_pipeline/mb && "
                "rm -rf /img2txt_pipeline/output && "
                "git clone -b updated_effocr git@github.com:dell-research-harvard/end-to-end-pipeline.git /img2txt_pipeline/end-to-end-pipeline && "
                "cp -a /img2txt_pipeline/end-to-end-pipeline/images_to_embeddings_pipeline/. /img2txt_pipeline/ && "
                f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest --tpslimit 10 remote:e2e2e/end-to-end-pipeline/pipeline_ingress/batch_manifests_dropbox_q/{batch}/{text_file} /img2txt_pipeline/mb && "
                f"rclone copyto -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest --tpslimit 10 --include-from /img2txt_pipeline/mb/{text_file} pipelineapp{remote_id + 20}: /img2txt_pipeline/pdf && "
                "ls -l /img2txt_pipeline/pdf && "
                "python3.8 /img2txt_pipeline/dbx_run_img2txt_yolo_pipeline.py "
                f"--manifest_path /img2txt_pipeline/pdf/ "
                "--checkpoint_path_layout /img2txt_pipeline/layout_model_new.onnx "
                "--checkpoint_path_line /img2txt_pipeline/line_model_new.onnx "
                "--label_map_path_layout /img2txt_pipeline/label_maps/label_map_layout.json "
                "--label_map_path_line /img2txt_pipeline/label_maps/label_map_line.json "
                "--effocr_char_recognizer_dir /img2txt_pipeline/char_recognizer "
                "--effocr_recognizer_dir /img2txt_pipeline/word_recognizer_new "
                "--effocr_localizer_dir /img2txt_pipeline/ "
                "--legibility-classifier /img2txt_pipeline/legibility_model_new.onnx "
                f"--output_save_path /img2txt_pipeline/output/{batch_idx} "
                "--line_model_backend yolov8 "
                "--localizer_model_backend yolov8 "
                "--layout_model_backend yolov8 "
                "--word-level-effocr "
                "--recognizer_word_thresh 0.83 && "
                f"ls -l /img2txt_pipeline/output/{batch_idx} && "
                f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest --dropbox-batch-mode off --tpslimit 10 /img2txt_pipeline/output/ remote:e2e2e/end-to-end-pipeline/pipeline_egress/dbx_pipeline/{batch}"
                "'"
            )

            # first VMs have no task dependencies; otherwise, task dependencies exist to enforce order
            if idx < VM_TOTAL:
                task = {
                    'id': literal(str(idx).zfill(6)),
                    'command': task_command,
                    'docker_image': f'caimg2txtregistry.azurecr.io/pipelineappv1:dropbox2',
                }
            else:
                task = {
                    'id': literal(str(idx).zfill(6)),
                    'depends_on_range': [idx % VM_TOTAL, idx % VM_TOTAL],
                    'command': task_command,
                    'docker_image': f'caimg2txtregistry.azurecr.io/pipelineappv1:dropbox2',
                }
            
            # append task
            all_tasks.append(task)
            idx += 1

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
        jobidx,
        batches,
        env,
        yaml_save_path = rf"C:\Users\bryan\Documents\NBER\batch-shipyard\config",
        manual_selection = None,
        failed_subset_from_csv = None,
        priority = 999,
        vm_total = 1000,
    ):

    # NA remaining
    """
    text_file_dir = f"/Users/jscarlson/Downloads/{local_batch_path}/{byear}"
    dropbox_ingress_path = f"pipeline_ingress/na_remaining_{byear}"
    dropbox_egress_path = f"pipeline_egress/na_remaining_{byear}"
    failed_subset_from_csv = f"/Users/jscarlson/Downloads/narem{byear}.csv" if failed_subset_from_csv else None
    jobid = f"narem{byear}"
    """

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
    jobid = f"locjob{byear}
    """

    # LOC new
    # manifest_path = f"/e2e2e/end-to-end-pipeline/pipeline_ingress/loc_manifests/{batch}"
    # failed_subset_from_csv = f"/home/jscarlson/Downloads/locjob{batch}.csv" if failed_subset_from_csv else None
    failed_subset_from_csv = None

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
    client = paramiko.SSHClient()
    host_keys = client.load_system_host_keys()
    client.connect('140.247.116.167', port=9287, username='tombryan', password='password')
    
    batch_manifests = []
    for batch in batches:
        stdin, stdout, stderr = client.exec_command(
            f'ls /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_ingress/batch_manifests_dropbox_q/{batch}'
        )
        
        batch_manifests.append(stdout.read().decode('utf-8').splitlines())
    
    jobid = f"locjob{jobidx}"

    job_yaml = create_job_yaml(
        batch_manifests, 
        save_path=yaml_save_path,
        batches = batches,
        jobid=jobid,
        priority=priority,
        failed_subset_from_csv=failed_subset_from_csv,
        manual_selection=manual_selection,
        vm_total=vm_total,
    )

    if not failed_subset_from_csv and not manual_selection:
        # subprocess.run(["rclone", "copy", "-P", text_file_dir, "vn:" + dropbox_ingress_path, "--transfers", "8", "--checkers", "8"])
        time.sleep(10)
        subprocess.run([r"C:\Users\bryan\Documents\NBER\batch-shipyard\shipyard.cmd", "jobs", "add"], env=env)
        time.sleep(30)


if __name__ == '__main__':

    env = os.environ.copy()   # Make a copy of the current environment
    env['SHIPYARD_CONFIGDIR'] = r'C:\Users\bryan\Documents\NBER\batch-shipyard\config_central_india'
    # subprocess.run([r"C:\Users\bryan\Documents\NBER\batch-shipyard\shipyard.cmd", "pool", "add"], env=env)
    # time.sleep(60)

    # for idx, y in enumerate(range(1969, 1970)):
    #    create_job(y, local_batch_path="newspaper_archive_67_72_remaining", 
    #        priority = 999 - (10*idx), data_source="newspaper_archive", env=env)
    
    # Get a list of batces to run
    batches = json.load(open(r'C:\Users\bryan\Documents\NBER\chronicling_america\dbx_batches.json', 'r'))
        
    job_dict ={i : batches[i * 20 : min((i + 1) * 20, len(batches))] for i in range((len(batches) + 20 - 1) // 20 )}
    print(len(job_dict))
    for job_idx, batch_set in job_dict.items():
        if job_idx <= 73:
            continue
        create_job(job_idx,
            batches=batch_set, 
            env=env, priority = 999,
            yaml_save_path = rf"C:\Users\bryan\Documents\NBER\batch-shipyard\config_central_india",
            manual_selection = None,
            failed_subset_from_csv = None,
            vm_total = 1250
        )
        if job_idx > 79:
            break
        

        # Get number of active shipyard jobs
        
            


    # create_job(1893, vm_total=20, failed_subset_from_csv=True)