class literal(str):
    pass


task_command = literal(
    "/bin/bash -c $'"
    f"ls -h /img2txt_pipeline && "
    f"python3.8 -c \"import requests; url = \'google.com\'; print(requests.get(url).text)\""
    "'"
)

print(task_command)