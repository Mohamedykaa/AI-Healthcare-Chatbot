#!/bin/bash
# Scan local image with Trivy
# Requires trivy installed on host or via docker
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    -v $HOME/Library/Caches:/root/.cache/ \
    aquasecurity/trivy:latest image \
    --severity HIGH,CRITICAL \
    --exit-code 1 \
    medical_bot_backend:latest
