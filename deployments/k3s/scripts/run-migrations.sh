#!/bin/bash
# Run Alembic migrations on K3s PostgreSQL database

set -e

NAMESPACE="sejm-whiz"
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=postgresql-pgvector -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "Error: PostgreSQL pod not found in namespace $NAMESPACE"
    exit 1
fi

echo "Found PostgreSQL pod: $POD_NAME"

# Create a temporary pod to run migrations
echo "Creating migration job..."

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: sejm-whiz-migrations-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: migrations
        image: sejm-whiz-processor:optimized
        command: ["/bin/bash", "-c"]
        args:
        - |
          echo "Setting up database connection..."
          export DATABASE_URL="postgresql://sejm_whiz_user:sejm_whiz_password@postgresql-pgvector.$NAMESPACE:5432/sejm_whiz"

          echo "Running Alembic migrations..."
          cd /app

          # Navigate to database component directory
          if [ -f components/sejm_whiz/database/alembic.ini ]; then
              cd components/sejm_whiz/database
              echo "Running migrations from components/sejm_whiz/database..."
          else
              echo "Error: components/sejm_whiz/database/alembic.ini not found"
              exit 1
          fi

          # Run migrations
          python -m alembic upgrade head

          echo "Migrations completed successfully!"
        env:
        - name: DATABASE_HOST
          value: "postgresql-pgvector.$NAMESPACE"
        - name: DATABASE_PORT
          value: "5432"
        - name: DATABASE_NAME
          value: "sejm_whiz"
        - name: DATABASE_USER
          value: "sejm_whiz_user"
        - name: DATABASE_PASSWORD
          value: "sejm_whiz_password"
        - name: PYTHONPATH
          value: "/app:/app/components"
  backoffLimit: 3
EOF

echo "Waiting for migration job to complete..."
kubectl wait --for=condition=complete --timeout=120s -n $NAMESPACE job/$(kubectl get jobs -n $NAMESPACE --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')

echo "Migration job completed!"

# Show job logs
JOB_NAME=$(kubectl get jobs -n $NAMESPACE --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
echo "Migration logs:"
kubectl logs -n $NAMESPACE job/$JOB_NAME

# Verify tables were created
echo ""
echo "Verifying database tables..."
kubectl exec -n $NAMESPACE $POD_NAME -- psql -U sejm_whiz_user -d sejm_whiz -c "\dt"

echo ""
echo "Database migrations completed successfully!"
