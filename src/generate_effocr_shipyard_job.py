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

class literal(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='>')

def create_batch_job_yaml(
        env, 
        yaml_save_path,
        batch,
        n_manifests
    ):

    # start up params
    all_tasks = []
    sleep_seconds = 3
    ssh_key = '\n'
    with open(r'C:\Users\bryan\.ssh\id_rsa', 'r') as f:
        count = 0
        for line in f:
            if count != 0 and 'END' not in line:
                ssh_key += line.strip()
                ssh_key += '\n'
            count += 1

    for i in range(10):
        # task command
        if i == n_manifests - 1:
            i += 1
        task_command = literal(
            "/bin/bash -c $'"
            "git clone -b updated_effocr git@github.com:dell-research-harvard/end-to-end-pipeline.git /img2txt_pipeline/end-to-end-pipeline && "
            "cp -a /img2txt_pipeline/end-to-end-pipeline/images_to_embeddings_pipeline/. /img2txt_pipeline/ && "
            "mkdir /img2txt_pipeline/manifest && "
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest remote:e2e2e/end-to-end-pipeline/pipeline_ingress/batch_manifests/{batch}/manifest_{i}.txt /img2txt_pipeline && "
            "rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest remote:e2e2e/end-to-end-pipeline/images_to_embeddings_pipeline/utils/iso_639_2_languages.json /img2txt_pipeline/utils && "
            "ls -l /img2txt_pipeline && "
            "rm -rf /img2txt_pipeline/end-to-end-pipeline && "
            "rm -rf /img2txt_pipeline/pdf && "
            "rm -rf /img2txt_pipeline/mb && "
            "rm -rf /img2txt_pipeline/output && "
            f"sleep {2 * i + 5} && "
            f"echo {i} && "
            "python3.8 /img2txt_pipeline/run_img2txt_yolo_pipeline.py "
            f"--manifest_path /img2txt_pipeline/manifest_{i}.txt "
            "--checkpoint_path_layout /img2txt_pipeline/layout_model_new.onnx "
            "--checkpoint_path_line /img2txt_pipeline/line_model_new.onnx "
            "--label_map_path_layout /img2txt_pipeline/label_maps/label_map_layout.json "
            "--label_map_path_line /img2txt_pipeline/label_maps/label_map_line.json "
            "--effocr_char_recognizer_dir /img2txt_pipeline/char_recognizer "
            "--effocr_recognizer_dir /img2txt_pipeline/word_recognizer_new "
            "--effocr_localizer_dir /img2txt_pipeline/ "
            "--legibility-classifier /img2txt_pipeline/legibility_model_new.onnx "
            f"--output_save_path /img2txt_pipeline/output/{i} "
            "--line_model_backend yolov8 "
            "--localizer_model_backend yolov8 "
            "--layout_model_backend yolov8 "
            "--word-level-effocr "
            "--first_n 5 "
            "--recognizer_word_thresh 0.83 && "
            f"ls -l /img2txt_pipeline/output/{i} && "
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest /img2txt_pipeline/output/ remote:e2e2e/end-to-end-pipeline/pipeline_egress/ca_pipeline/{batch}"
            "'"
        )


        # first VMs have no task dependencies; otherwise, task dependencies exist to enforce order
        task = {
            'id': literal(str(i).zfill(6)),
            'command': task_command,
            'docker_image': f'caimg2txtregistry.azurecr.io/pipelineappv1:finalmodels',
            'max_wall_time': '0.04:30:00',
            'retention_time': '0.03:00:00',
        }

            
        # append task
        all_tasks.append(task)

    # create single job
    job = {'job_specifications': [{
        'id': f'{batch}',
        'auto_complete': True,
        'tasks': all_tasks,
        'allow_run_on_missing_image': True
    }]}

    # convert python obj to yaml file, write
    yaml.add_representer(literal, literal_presenter)
    os.makedirs(yaml_save_path, exist_ok=True)
    with open(os.path.join(yaml_save_path, 'jobs.yaml'), 'w') as f:
        job_yaml = yaml.dump(job, f)

    return job_yaml

def create_job_yaml(
        env, 
        yaml_save_path,
        vm_total=1,
    ):

    # start up params
    all_tasks = []
    VM_TOTAL = vm_total
    sleep_seconds = 3
    ssh_key = '\n'
    with open(r'C:\Users\bryan\.ssh\id_rsa', 'r') as f:
        count = 0
        for line in f:
            if count != 0 and 'END' not in line:
                ssh_key += line.strip()
                ssh_key += '\n'
            count += 1

    for i in range(171):
        # task command
        task_command = literal(
            "/bin/bash -c $'"
            "git clone -b updated_effocr git@github.com:dell-research-harvard/end-to-end-pipeline.git /img2txt_pipeline/end-to-end-pipeline && "
            "cp -a /img2txt_pipeline/end-to-end-pipeline/images_to_embeddings_pipeline/. /img2txt_pipeline/ && "
            "mkdir /img2txt_pipeline/manifest && "
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest remote:e2e2e/end-to-end-pipeline/pipeline_ingress/sample_manifests_dpd/manifest_{i}.txt /img2txt_pipeline && "
            "rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest remote:e2e2e/end-to-end-pipeline/images_to_embeddings_pipeline/utils/iso_639_2_languages.json /img2txt_pipeline/utils && "
            "ls -l /img2txt_pipeline && "
            "rm -rf /img2txt_pipeline/end-to-end-pipeline && "
            "rm -rf /img2txt_pipeline/pdf && "
            "rm -rf /img2txt_pipeline/mb && "
            "rm -rf /img2txt_pipeline/output && "
            f"sleep {sleep_seconds} && "
            f"echo {i} && "
            "python3.8 /img2txt_pipeline/run_img2txt_yolo_pipeline.py "
            f"--manifest_path /img2txt_pipeline/manifest_{i}.txt "
            "--checkpoint_path_layout /img2txt_pipeline/layout_model_new.onnx "
            "--checkpoint_path_line /img2txt_pipeline/line_model_new.onnx "
            "--label_map_path_layout /img2txt_pipeline/label_maps/label_map_layout.json "
            "--label_map_path_line /img2txt_pipeline/label_maps/label_map_line.json "
            "--effocr_char_recognizer_dir /img2txt_pipeline/char_recognizer "
            "--effocr_recognizer_dir /img2txt_pipeline/word_recognizer_new "
            "--effocr_localizer_dir /img2txt_pipeline/ "
            "--legibility-classifier /img2txt_pipeline/legibility_model_new.onnx "
            f"--output_save_path /img2txt_pipeline/output/{i} "
            "--line_model_backend yolov8 "
            "--localizer_model_backend yolov8 "
            "--layout_model_backend yolov8 "
            "--word-level-effocr "
            "--recognizer_word_thresh 0.82 && "
            f"ls -l /img2txt_pipeline/output/{i} && "
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest /img2txt_pipeline/output/ remote:e2e2e/end-to-end-pipeline/pipeline_egress/dpd_final_sample"
            "'"
        )


        # first VMs have no task dependencies; otherwise, task dependencies exist to enforce order
        task = {
            'id': literal(str(i).zfill(6)),
            'command': task_command,
            'docker_image': f'caimg2txtregistry.azurecr.io/pipelineappv1:finalmodels'
        }

            
        # append task
        all_tasks.append(task)

    # create single job
    job = {'job_specifications': [{
        'id': 'dpp_final_run',
        'tasks': all_tasks,
        'allow_run_on_missing_image': True
    }]}

    # convert python obj to yaml file, write
    yaml.add_representer(literal, literal_presenter)
    os.makedirs(yaml_save_path, exist_ok=True)
    print(yaml_save_path)
    with open(os.path.join(yaml_save_path, 'jobs.yaml'), 'w') as f:
        job_yaml = yaml.dump(job, f)

    return job_yaml

def create_test_job_yaml(
        env, 
        yaml_save_path,
        vm_total=1,
    ):

    # start up params
    all_tasks = []
    VM_TOTAL = vm_total
    sleep_seconds = 3
    ssh_key = '\n'
    with open(r'C:\Users\bryan\.ssh\id_rsa', 'r') as f:
        count = 0
        for line in f:
            if count != 0 and 'END' not in line:
                ssh_key += line.strip()
                ssh_key += '\n'
            count += 1

    for i in range(1):
        # task command
        task_command = literal(
            "/bin/bash -c $'"
            "rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest remote:e2e2e/end-to-end-pipeline/pipeline_ingress/batch_manifests/nvln_hamilton_ver01/manifest_0.txt /img2txt_pipeline && "
            "git clone -b updated_effocr git@github.com:dell-research-harvard/end-to-end-pipeline.git /img2txt_pipeline/end-to-end-pipeline && "
            "cp -a /img2txt_pipeline/end-to-end-pipeline/images_to_embeddings_pipeline/. /img2txt_pipeline/ && "
            f"ls -h /img2txt_pipeline && "
            f"python3.8 /img2txt_pipeline/img_download_test.py "
            "'"
        )


        # first VMs have no task dependencies; otherwise, task dependencies exist to enforce order
        task = {
            'id': literal(str(i).zfill(6)),
            'command': task_command,
            'docker_image': f'caimg2txtregistry.azurecr.io/pipelineappv1:finalmodels'
        }

            
        # append task
        all_tasks.append(task)

    # create single job
    job = {'job_specifications': [{
        'id': 'test_download_2',
        'tasks': all_tasks,
        'allow_run_on_missing_image': True
    }]}

    # convert python obj to yaml file, write
    yaml.add_representer(literal, literal_presenter)
    os.makedirs(yaml_save_path, exist_ok=True)
    print(yaml_save_path)
    with open(os.path.join(yaml_save_path, 'jobs.yaml'), 'w') as f:
        job_yaml = yaml.dump(job, f)

    return job_yaml

def create_job(
        env,
        yaml_save_path,
        batch = None,
        n_manifests = None
    ):

    if batch is None:
        job_yaml = create_test_job_yaml(
            env=env,
            yaml_save_path=yaml_save_path,
        )
    else:
        job_yaml = create_batch_job_yaml(
            env=env, 
            yaml_save_path=yaml_save_path,
            batch=batch,
            n_manifests=n_manifests
        )

    testing = "TESTING"
    print(f"***\n{testing}\n***")
    subprocess.run([r"C:\Users\bryan\Documents\NBER\batch-shipyard\shipyard.cmd", "jobs", "add"], env=env)


if __name__ == '__main__':

    shipyard_dir = r'C:\Users\bryan\Documents\NBER\batch-shipyard'
    env = os.environ.copy()   # Make a copy of the current environment
    env['SHIPYARD_CONFIGDIR'] = r'C:\Users\bryan\Documents\NBER\batch-shipyard\config_hdsi'
    
    with open(r'C:\Users\bryan\Documents\NBER\chronicling_america\truncated_batches.txt') as infile:
        batches = infile.read().splitlines()
        
    # Get the list of batches completed stored on the remote server at /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress
    # from ssh -p 9287 tombryan@140.247.116.167
    # ls /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress

    # Get the list of batches completed stored on the remote server at /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress    
    client = paramiko.SSHClient()
    host_keys = client.load_system_host_keys()
    client.connect('140.247.116.167', port=9287, username='tombryan', password='password')

    stdin, stdout, stderr = client.exec_command(
        'ls /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress/ca_pipeline'
    )
    completed_batches = stdout.read().decode('utf-8').splitlines()
    
    n_to_run = 1
    to_run_batches = []
    for batch in batches:
        if batch[:-1] not in completed_batches:
            try:
                manifest_count = client.exec_command(
                                f'ls /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_ingress/batch_manifests/{batch}'
                                )[1].read().decode('utf-8').count('.txt') - 1
                if manifest_count > 0:
                    to_run_batches.append((batch, manifest_count))
            except Exception as e:
                print(e)
                continue
            
            if len(to_run_batches) == n_to_run:
                break
    
    for batch, manifest_count in to_run_batches:
        print(batch, manifest_count)
        create_job(env=env, yaml_save_path=r'C:\Users\bryan\Documents\NBER\batch-shipyard\config_hdsi', batch=batch[:-1], n_manifests=manifest_count)
        time.sleep(5)
    
    
    # Create the job
    # batch = 'ak_dallsheep_ver02'
    # n_manifests = 98
    # create_job(env=env, yaml_save_path=r'C:\Users\bryan\Documents\NBER\batch-shipyard\config_central_india')
