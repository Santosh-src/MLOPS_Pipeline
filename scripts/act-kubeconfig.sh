#!/usr/bin/env bash
# Patch the kubeconfig inside an `act` container so it can reach Docker Desktop K8s.
# Sourced (or executed) by ci.yml CD/undeploy steps. Sets KUBECONFIG to a patched copy.
set -euo pipefail

KUBECONFIG_SRC="${KUBECONFIG_SRC:-/root/.kube/config}"
KUBECONFIG_OUT="${KUBECONFIG_OUT:-/tmp/kubeconfig-patched}"

if [[ ! -f "$KUBECONFIG_SRC" ]]; then
  echo "ERROR: kubeconfig not found at $KUBECONFIG_SRC" >&2
  echo "Run act through the wrapper that mounts your kubeconfig:" >&2
  echo "  ./scripts/act-local.sh push" >&2
  exit 1
fi

cp "$KUBECONFIG_SRC" "$KUBECONFIG_OUT"
export KUBECONFIG="$KUBECONFIG_OUT"

CLUSTER_NAME=$(kubectl config view -o jsonpath='{.clusters[0].name}')
CURRENT_SERVER=$(kubectl config view -o jsonpath='{.clusters[0].cluster.server}')
NEW_SERVER=$(echo "$CURRENT_SERVER" | sed \
  -e 's|://localhost|://host.docker.internal|' \
  -e 's|://127\.0\.0\.1|://host.docker.internal|' \
  -e 's|://\[::1\]|://host.docker.internal|')

kubectl config set-cluster "$CLUSTER_NAME" --server="$NEW_SERVER" --insecure-skip-tls-verify=true >/dev/null
kubectl config unset "clusters.${CLUSTER_NAME}.certificate-authority-data" >/dev/null

kubectl cluster-info >/dev/null
echo "$KUBECONFIG_OUT"
