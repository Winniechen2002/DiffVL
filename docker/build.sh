docker build . -t task --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa_task_annotator)" # --no-cache
