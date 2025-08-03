#!/bin/bash
# Setup production-ready K3s deployment with monitoring and error handling

set -e

NAMESPACE="sejm-whiz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="$(dirname "$SCRIPT_DIR")/manifests"

echo "🚀 Setting up production-ready Sejm-Whiz K3s deployment..."

# Function to wait for deployment rollout
wait_for_deployment() {
    local deployment=$1
    echo "⏳ Waiting for deployment $deployment to be ready..."
    kubectl rollout status deployment/$deployment -n $NAMESPACE --timeout=300s
}

# Function to check pod health
check_pod_health() {
    local app_label=$1
    echo "🔍 Checking health of pods with label app=$app_label..."

    # Wait for pods to be running
    kubectl wait --for=condition=Ready pod -l app=$app_label -n $NAMESPACE --timeout=300s

    # Check readiness probe
    local pod_name=$(kubectl get pods -n $NAMESPACE -l app=$app_label -o jsonpath='{.items[0].metadata.name}')
    echo "✅ Pod $pod_name is ready"
}

# Function to setup monitoring
setup_monitoring() {
    echo "📊 Setting up monitoring and metrics collection..."

    # Apply monitoring configuration
    kubectl apply -f "$MANIFESTS_DIR/k3s-monitoring.yaml"

    # Create metrics collection service if not exists
    if ! kubectl get service prometheus-server -n monitoring 2>/dev/null; then
        echo "⚠️  Prometheus not found. Consider installing kube-prometheus-stack:"
        echo "   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts"
        echo "   helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace"
    fi
}

# Function to verify database connectivity
verify_database() {
    echo "🗄️  Verifying database connectivity..."

    local postgres_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=postgresql-pgvector -o jsonpath='{.items[0].metadata.name}')

    if [ -n "$postgres_pod" ]; then
        kubectl exec -n $NAMESPACE $postgres_pod -- psql -U sejm_whiz_user -d sejm_whiz -c "SELECT 1;" > /dev/null
        echo "✅ Database connectivity verified"
    else
        echo "❌ PostgreSQL pod not found"
        exit 1
    fi
}

# Function to test GPU availability
test_gpu() {
    echo "🔧 Testing GPU availability..."

    local processor_pod=$(kubectl get pods -n $NAMESPACE -l app=sejm-whiz-processor-gpu -o jsonpath='{.items[0].metadata.name}')

    if [ -n "$processor_pod" ]; then
        kubectl exec -n $NAMESPACE $processor_pod -- nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null
        echo "✅ GPU access verified"
    else
        echo "❌ Processor GPU pod not found"
        exit 1
    fi
}

# Function to setup error recovery
setup_error_recovery() {
    echo "🔄 Setting up error recovery mechanisms..."

    # Add restart policy and backoff limits to existing deployments
    kubectl patch deployment sejm-whiz-processor-gpu -n $NAMESPACE -p '{
        "spec": {
            "template": {
                "spec": {
                    "restartPolicy": "Always"
                }
            }
        }
    }'

    echo "✅ Error recovery mechanisms configured"
}

# Function to validate deployment
validate_deployment() {
    echo "✅ Validating production deployment..."

    # Check all deployments are ready
    local deployments=("postgresql-pgvector" "redis" "sejm-whiz-processor-gpu" "sejm-whiz-web-ui")

    for deployment in "${deployments[@]}"; do
        if kubectl get deployment $deployment -n $NAMESPACE &>/dev/null; then
            wait_for_deployment $deployment
        else
            echo "⚠️  Deployment $deployment not found, skipping..."
        fi
    done

    # Verify core services
    verify_database
    test_gpu

    # Check web UI health
    echo "🌐 Checking Web UI health..."
    local web_ui_pod=$(kubectl get pods -n $NAMESPACE -l app=sejm-whiz-web-ui -o jsonpath='{.items[0].metadata.name}')
    if [ -n "$web_ui_pod" ]; then
        kubectl exec -n $NAMESPACE $web_ui_pod -- curl -f http://localhost:8000/health > /dev/null
        echo "✅ Web UI health check passed"
    fi
}

# Main execution
main() {
    echo "🔧 Production setup starting..."

    # Ensure namespace exists
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

    # Setup monitoring
    setup_monitoring

    # Setup error recovery
    setup_error_recovery

    # Validate deployment
    validate_deployment

    echo ""
    echo "🎉 Production setup completed successfully!"
    echo ""
    echo "📊 Monitoring endpoints:"
    echo "   - Web UI: http://192.168.0.200:30800/dashboard"
    echo "   - Health: http://192.168.0.200:30800/health"
    echo "   - Logs: curl -N http://192.168.0.200:30800/api/logs/stream"
    echo ""
    echo "🔍 Useful commands:"
    echo "   - Check deployment status: kubectl get all -n $NAMESPACE"
    echo "   - View processor logs: kubectl logs -f -n $NAMESPACE deployment/sejm-whiz-processor-gpu"
    echo "   - Monitor GPU usage: kubectl exec -n $NAMESPACE deployment/sejm-whiz-processor-gpu -- nvidia-smi"
    echo "   - Check database: kubectl exec -n $NAMESPACE deployment/postgresql-pgvector -- psql -U sejm_whiz_user -d sejm_whiz -c '\\dt'"
}

# Error handling
trap 'echo "❌ Production setup failed. Check the logs above for details."' ERR

# Run main function
main "$@"
