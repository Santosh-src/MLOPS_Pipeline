#!/usr/bin/env bash
# Deploy the Heart Disease MLOps stack to Docker Desktop Kubernetes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_DIR/k8s"

kubectl config use-context docker-desktop

if ! docker image inspect heart-disease-api:latest &>/dev/null; then
  docker build -t heart-disease-api:latest "$PROJECT_DIR"
fi

kubectl apply -f "$K8S_DIR/namespaces.yaml"
kubectl apply -f "$K8S_DIR/deployment.yaml"
kubectl apply -f "$K8S_DIR/service.yaml"

if kubectl get ingressclass nginx &>/dev/null; then
  kubectl apply -f "$K8S_DIR/ingress.yaml"
else
  echo "NGINX Ingress Controller not found -- skipping ingress."
fi

kubectl apply -f "$K8S_DIR/mlflow.yaml"
kubectl apply -f "$K8S_DIR/prometheus.yaml"
kubectl apply -f "$K8S_DIR/grafana.yaml"

kubectl rollout status deployment/heart-disease-api -n mlops-app        --timeout=120s
kubectl rollout status deployment/mlflow-server     -n mlops-tracking   --timeout=180s
kubectl rollout status deployment/prometheus        -n mlops-monitoring --timeout=120s
kubectl rollout status deployment/grafana           -n mlops-monitoring --timeout=120s

cat <<EOF

Services:
  API:        http://localhost:8080      (mlops-app)
  API docs:   http://localhost:8080/docs
  MLflow UI:  http://localhost:5001      (mlops-tracking)
  Prometheus: http://localhost:9090      (mlops-monitoring)
  Grafana:    http://localhost:3000      (mlops-monitoring, admin/admin)

Tear down: ./scripts/undeploy.sh
EOF
