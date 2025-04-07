#!/bin/bash
docker rm -f ai-train
docker build -t vk/ai-train .
docker run -d --restart always --net=host -v /stor:/stor -v /etc/localtime:/etc/localtime:ro \
        --log-opt max-size=500m --log-opt max-file=5 \
        -e APP_HOST='0.0.0.0' \
        -e APP_PORT='8088' \
        -e LOGGER_DIR='/stor/data/logs/amd-train/' \
        -e FILE_DIR='/stor/data/amd-train/' \
        -e SQLALCHEMY_DATABASE_URI='mysql://root:root@127.0.0.1/amd_train' \
        -e USER_DEFAULT_PASSWORD='' \
        -e PYTHONUNBUFFERED=0 \
        --name ai-interface vk/ai-train
