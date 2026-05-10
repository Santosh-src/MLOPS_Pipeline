#!/usr/bin/env bash
# Tear down the Heart Disease MLOps stack from Docker Desktop Kubernetes.
# Usage: ./scripts/undeploy.sh [--all]
#   --all  Also remove the local heart-disease-api Docker image.
set -euo pipefail

DELETE_IMAGE="false"
[[ "${1:-}" == "--all" ]] && DELETE_IMAGE="true"

kubectl config use-context docker-desktop || true

for ns in mlops-app mlops-tracking mlops-monitoring; do
  if kubectl get namespace "$ns" &>/dev/null; then
    kubectl delete namespace "$ns" --wait=true --timeout=60s || true
  fi
done

if [[ "$DELETE_IMAGE" == "true" ]]; then
  docker rmi heart-disease-api:latest 2>/dev/null || true
fi

echo "Undeploy complete."
