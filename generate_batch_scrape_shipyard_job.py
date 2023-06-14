import yaml
import argparse
import os
import numpy as np
import subprocess
import pandas as pd
from itertools import compress
import time
import json

class literal(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='>')

def create_batch_job_yaml(
        env, 
        yaml_save_path,
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

    for i, batch in enumerate(batches[1:]):
        # task command
        task_command = literal(
            "/bin/bash -c $'"
            "git clone -b updated_effocr git@github.com:dell-research-harvard/end-to-end-pipeline.git /img2txt_pipeline/end-to-end-pipeline && "
            "cp -a /img2txt_pipeline/end-to-end-pipeline/images_to_embeddings_pipeline/. /img2txt_pipeline/ && "
            "ls -l /img2txt_pipeline && "
            "rm -rf /img2txt_pipeline/end-to-end-pipeline && "
            "rm -rf /img2txt_pipeline/pdf && "
            "rm -rf /img2txt_pipeline/mb && "
            "rm -rf /img2txt_pipeline/output && "
            f"sleep {sleep_seconds} && "
            f"echo {batch} && "
            f"python3.8 /img2txt_pipeline/generate_batch_manifest.py --batch {batch} && "
            f"ls -l /img2txt_pipeline/ && "
            f"ls -l / && "
            f"rclone copy -P --config /root/.config/rclone/rclone.conf -vv --no-check-dest ./{batch}/ remote:e2e2e/end-to-end-pipeline/pipeline_ingress/batch_manifests/{batch}"
            "'"
        )

        # first VMs have no task dependencies; otherwise, task dependencies exist to enforce order
        task = {
            'id': literal(str(i).zfill(6)),
            'command': task_command,
            'docker_image': f'caimg2txtregistry.azurecr.io/pipelineappv1:latest'
        }

        # append task
        all_tasks.append(task)

    # create single job
    job = {'job_specifications': [{
        'id': f'redo_scrape_batches',
        'tasks': all_tasks,
    }]}

    # convert python obj to yaml file, write
    yaml.add_representer(literal, literal_presenter)
    os.makedirs(yaml_save_path, exist_ok=True)
    with open(os.path.join(yaml_save_path, 'jobs.yaml'), 'w') as f:
        job_yaml = yaml.dump(job, f)

    return job_yaml



def create_job(
        env,
        yaml_save_path,
        batch = None,
    ):

    job_yaml = create_batch_job_yaml(
        env=env, 
        yaml_save_path=yaml_save_path,
    )

    testing = "TESTING"
    print(f"***\n{testing}\n***")
    subprocess.run([r"C:\Users\bryan\Documents\NBER\batch-shipyard\shipyard.cmd", "jobs", "add"], env=env)


if __name__ == '__main__':

    shipyard_dir = r'C:\Users\bryan\Documents\NBER\batch-shipyard'
    env = os.environ.copy()   # Make a copy of the current environment
    env['SHIPYARD_CONFIGDIR'] = r'C:\Users\bryan\Documents\NBER\batch-shipyard\config'
    
    with open(r'C:\Users\bryan\Documents\NBER\chronicling_america\redo_batches.txt', 'r') as f:
        batches = f.read().splitlines()
    batches = [b for b in batches if not b.startswith('ak')]

    # Create the job
    create_job(env=env, yaml_save_path=r'C:\Users\bryan\Documents\NBER\batch-shipyard\config')
    
