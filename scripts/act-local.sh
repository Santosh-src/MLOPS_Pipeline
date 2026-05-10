#!/usr/bin/env bash
# Run the GitHub Actions workflow locally via act, with your kubeconfig
# mounted into the container so the cd/undeploy jobs can talk to
# Docker Desktop's K8s API.
#
# Usage:
#   ./scripts/act-local.sh push                                                  # full CI + CD
#   ./scripts/act-local.sh -j ci                                                 # CI only
#   ./scripts/act-local.sh workflow_dispatch -j undeploy --input action=undeploy # tear down
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="$PROJECT_DIR/.cache"

if [[ ! -f "${HOME}/.kube/config" ]]; then
  echo "ERROR: ${HOME}/.kube/config not found." >&2
  echo "Enable Kubernetes in Docker Desktop and ensure kubectl works locally." >&2
  exit 1
fi

ARCH=$(uname -m)
case "$ARCH" in
  x86_64)        K8S_ARCH="amd64" ;;
  aarch64|arm64) K8S_ARCH="arm64" ;;
  *) echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

KUBECTL_LINUX="$CACHE_DIR/kubectl-linux-$K8S_ARCH"
if [[ ! -x "$KUBECTL_LINUX" ]]; then
  echo "Downloading Linux kubectl for the act container (one-time)..."
  mkdir -p "$CACHE_DIR"
  K8S_VER=$(curl --connect-timeout 10 --max-time 30 -sL https://dl.k8s.io/release/stable.txt)
  curl -fL --progress-bar -o "$KUBECTL_LINUX" \
    "https://dl.k8s.io/release/$K8S_VER/bin/linux/$K8S_ARCH/kubectl"
  chmod +x "$KUBECTL_LINUX"
  echo "Cached at $KUBECTL_LINUX"
fi

CONTAINER_OPTS="-v ${HOME}/.kube:/root/.kube -v ${KUBECTL_LINUX}:/usr/local/bin/kubectl"

exec act --container-options "$CONTAINER_OPTS" "$@"
