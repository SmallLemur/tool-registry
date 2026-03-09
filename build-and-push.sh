#!/bin/bash
set -e
cd "$(dirname "$0")"

IMAGE="registry.gitlab.com/pensante1/tool-registry"

# Load credentials from parent .env if it exists
if [ -f "../.env" ]; then
    source "../.env"
fi

GITLAB_USER="${GITLAB_USER:-}"
GITLAB_TOKEN="${GITLAB_TOKEN:-}"

if [ -z "$GITLAB_USER" ]; then
    read -p "GitLab username: " GITLAB_USER
fi
if [ -z "$GITLAB_TOKEN" ]; then
    read -sp "GitLab Personal Access Token: " GITLAB_TOKEN
    echo
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "${GITLAB_TOKEN}" | docker login registry.gitlab.com -u "${GITLAB_USER}" --password-stdin

docker build --no-cache --platform linux/amd64 \
    -t "${IMAGE}:latest" \
    -t "${IMAGE}:${TIMESTAMP}" \
    -f Dockerfile .

docker push "${IMAGE}:latest"
docker push "${IMAGE}:${TIMESTAMP}"

echo "Done! Image: ${IMAGE}:latest (${TIMESTAMP})"
