#!/bin/bash
mkdir -p certs
# Generate Self Signed Certs for Postgres
openssl req -new -x509 -days 365 -nodes -text -out certs/server.crt \
  -keyout certs/server.key -subj "/CN=db"
chmod 600 certs/server.key
