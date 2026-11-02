# 8. Deployment and CI/CD

## 8.1 Azure resources

| Resource | Name (example) | Purpose | Cost tip |
|----------|----------------|---------|----------|
| Resource group | `rg-taxlens` | Everything in one place, so cleanup is a single delete | — |
| Storage account | `sttaxlens<suffix>` | Blob container `documents`, queues `ingest-jobs`, `ingest-poison` | Standard_LRS, costs very little |
| Event Grid system topic + subscription | `blob-created-to-queue` | `BlobCreated` on `documents/public/` → `ingest-jobs` | Free tier covers this volume |
| Azure Cosmos DB for NoSQL | `cosmos-taxlens-<suffix>` | App data: users, workspaces, memberships, documents, audit log | Free tier (1000 RU/s + 25 GB, one account per subscription) |
| Pinecone (SaaS, not Azure) | indexes `taxlens-ci`, `taxlens-prod` | Vectors + masked chunk text | Starter plan is free; one project, separate indexes per environment |
| Key Vault | `kv-taxlens-<suffix>` | API keys (Gemini, Groq, Pinecone, LangSmith), JWT secret | Pennies |
| User-assigned managed identity | `id-taxlens` | Key Vault, Storage and Cosmos data-plane access, no keys | Free |
| Container Apps environment | `cae-taxlens` | Hosts the app + job, Log Analytics | Consumption plan has a monthly free grant |
| Container App | `ca-taxlens-api` | FastAPI | `min-replicas 0` (cold start) or 1 for demos |
| Container Apps Job | `caj-taxlens-indexer` | Queue-triggered indexer | Only bills while running |
| Container App (optional) | `ca-taxlens-phoenix` | Hosted Phoenix | Or keep Phoenix local and use Phoenix Cloud |
| Image registry | GitHub Container Registry (`ghcr.io`) | Docker images | Free; Azure Container Registry Basic is the alternative |

**Set a budget alert** on the resource group, e.g. ₹1,000/month with alerts at 50/80/100%.
Delete the resource group when you are not demoing.

Region: `centralindia` is the natural choice for Indian tax data. Check that every service above
is available there and fall back to `southindia` if one isn't.

## 8.2 Provisioning script (`infra/deploy.sh`)

Run it in Git Bash or WSL. Every command is idempotent enough to re-run. Once it works, port it
to Bicep (`infra/main.bicep`).

```bash
#!/usr/bin/env bash
set -euo pipefail

SUFFIX=${SUFFIX:-$(openssl rand -hex 3)}
LOC=centralindia
RG=rg-taxlens
ST=sttaxlens$SUFFIX
COSMOS=cosmos-taxlens-$SUFFIX
KV=kv-taxlens-$SUFFIX
ENV=cae-taxlens
ID=id-taxlens
IMAGE=ghcr.io/<your-github-user-lowercase>/taxlens:latest

az extension add --name containerapp --upgrade -y
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.EventGrid \
          Microsoft.DocumentDB Microsoft.KeyVault Microsoft.Storage; do
  az provider register --namespace $ns --wait
done

az group create -n $RG -l $LOC

# ── Storage: blob container + queues ────────────────────────────
az storage account create -n $ST -g $RG -l $LOC --sku Standard_LRS --kind StorageV2 \
  --allow-blob-public-access false --min-tls-version TLS1_2
ST_ID=$(az storage account show -n $ST -g $RG --query id -o tsv)
ME=$(az ad signed-in-user show --query id -o tsv)
az role assignment create --assignee $ME --role "Storage Blob Data Contributor"  --scope $ST_ID
az role assignment create --assignee $ME --role "Storage Queue Data Contributor" --scope $ST_ID
sleep 30   # RBAC propagation
az storage container create --account-name $ST -n documents     --auth-mode login
az storage queue     create --account-name $ST -n ingest-jobs   --auth-mode login
az storage queue     create --account-name $ST -n ingest-poison --auth-mode login

# ── Event Grid: public corpus uploads -> queue ─────────────────
az eventgrid event-subscription create \
  --name blob-created-to-queue \
  --source-resource-id $ST_ID \
  --endpoint-type storagequeue \
  --endpoint "$ST_ID/queueservices/default/queues/ingest-jobs" \
  --included-event-types Microsoft.Storage.BlobCreated \
  --subject-begins-with /blobServices/default/containers/documents/blobs/public/

# ── Cosmos DB for NoSQL (app data: users, workspaces, documents, audit) ──
# Free tier: one account per subscription. If yours is used up, drop --enable-free-tier
# (or use --capabilities EnableServerless and remove --throughput below).
az cosmosdb create -n $COSMOS -g $RG \
  --locations regionName=$LOC failoverPriority=0 isZoneRedundant=False \
  --default-consistency-level Session \
  --enable-free-tier true
az cosmosdb sql database create -a $COSMOS -g $RG -n taxlens --throughput 1000   # shared RU/s
for spec in "users:/email" "workspaces:/id" "memberships:/user_id" \
            "documents:/workspace_id" "audit_log:/workspace_id"; do
  az cosmosdb sql container create -a $COSMOS -g $RG -d taxlens \
    -n "${spec%%:*}" --partition-key-path "${spec#*:}"
done
COSMOS_ENDPOINT=$(az cosmosdb show -n $COSMOS -g $RG --query documentEndpoint -o tsv)

# ── Pinecone (SaaS, outside Azure) ─────────────────────────────
# Create the prod index once from your machine:
#   PINECONE_API_KEY=... PINECONE_INDEX=taxlens-prod uv run python scripts/init_pinecone.py
# The API key is stored in Key Vault below.

# ── Key Vault + secrets ─────────────────────────────────────────
az keyvault create -n $KV -g $RG -l $LOC --enable-rbac-authorization true
KV_ID=$(az keyvault show -n $KV --query id -o tsv)
az role assignment create --assignee $ME --role "Key Vault Secrets Officer" --scope $KV_ID
sleep 30
az keyvault secret set --vault-name $KV -n pinecone-api-key  --value "$PINECONE_API_KEY" > /dev/null
az keyvault secret set --vault-name $KV -n jwt-secret        --value "$(openssl rand -base64 48)" > /dev/null
az keyvault secret set --vault-name $KV -n google-api-key    --value "$GOOGLE_API_KEY"  > /dev/null
az keyvault secret set --vault-name $KV -n groq-api-key      --value "$GROQ_API_KEY"    > /dev/null
az keyvault secret set --vault-name $KV -n langsmith-api-key --value "${LANGSMITH_API_KEY:-none}" > /dev/null
# KEDA's queue scaler needs a connection string (or managed identity, see note below)
ST_CONN=$(az storage account show-connection-string -n $ST -g $RG --query connectionString -o tsv)
az keyvault secret set --vault-name $KV -n storage-conn --value "$ST_CONN" > /dev/null

# ── Managed identity + roles ────────────────────────────────────
az identity create -n $ID -g $RG -l $LOC
ID_RES=$(az identity show -n $ID -g $RG --query id -o tsv)
ID_PRINCIPAL=$(az identity show -n $ID -g $RG --query principalId -o tsv)
ID_CLIENT=$(az identity show -n $ID -g $RG --query clientId -o tsv)
for role in "Storage Blob Data Contributor" "Storage Queue Data Contributor"; do
  az role assignment create --assignee-object-id $ID_PRINCIPAL \
    --assignee-principal-type ServicePrincipal --role "$role" --scope $ST_ID
done
az role assignment create --assignee-object-id $ID_PRINCIPAL \
  --assignee-principal-type ServicePrincipal --role "Key Vault Secrets User" --scope $KV_ID

# ── Cosmos DB data-plane RBAC (no account keys in the app) ──────
# Built-in "Cosmos DB Built-in Data Contributor" on the read/write containers
for c in users workspaces memberships documents; do
  az cosmosdb sql role assignment create -a $COSMOS -g $RG \
    --role-definition-id 00000000-0000-0000-0000-000000000002 \
    --principal-id $ID_PRINCIPAL --scope "/dbs/taxlens/colls/$c"
done
# Custom append-only role for audit_log: create + read only, no replace/upsert/delete
cat > /tmp/audit-role.json <<'JSON'
{
  "RoleName": "TaxLensAuditAppender",
  "Type": "CustomRole",
  "AssignableScopes": ["/"],
  "Permissions": [{
    "DataActions": [
      "Microsoft.DocumentDB/databaseAccounts/readMetadata",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/create",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/items/read",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/executeQuery",
      "Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers/readChangeFeed"
    ]
  }]
}
JSON
AUDIT_ROLE=$(az cosmosdb sql role definition create -a $COSMOS -g $RG \
  -b @/tmp/audit-role.json --query name -o tsv)
az cosmosdb sql role assignment create -a $COSMOS -g $RG \
  --role-definition-id $AUDIT_ROLE --principal-id $ID_PRINCIPAL \
  --scope "/dbs/taxlens/colls/audit_log"
# Yourself, account-wide, so scripts/create_user.py works with your `az login`
az cosmosdb sql role assignment create -a $COSMOS -g $RG \
  --role-definition-id 00000000-0000-0000-0000-000000000002 \
  --principal-id $ME --scope "/"

# ── Container Apps environment ─────────────────────────────────
az containerapp env create -n $ENV -g $RG -l $LOC

KVURL="https://$KV.vault.azure.net/secrets"
SECRETS=(
  "pinecone-api-key=keyvaultref:$KVURL/pinecone-api-key,identityref:$ID_RES"
  "jwt-secret=keyvaultref:$KVURL/jwt-secret,identityref:$ID_RES"
  "google-api-key=keyvaultref:$KVURL/google-api-key,identityref:$ID_RES"
  "groq-api-key=keyvaultref:$KVURL/groq-api-key,identityref:$ID_RES"
  "langsmith-api-key=keyvaultref:$KVURL/langsmith-api-key,identityref:$ID_RES"
  "storage-conn=keyvaultref:$KVURL/storage-conn,identityref:$ID_RES"
)
ENVVARS=(
  APP_ENV=prod
  DB_BACKEND=cosmos
  COSMOS_ENDPOINT=$COSMOS_ENDPOINT
  COSMOS_DATABASE=taxlens
  VECTOR_BACKEND=pinecone
  PINECONE_INDEX=taxlens-prod
  PINECONE_API_KEY=secretref:pinecone-api-key
  JWT_SECRET=secretref:jwt-secret
  GOOGLE_API_KEY=secretref:google-api-key
  GROQ_API_KEY=secretref:groq-api-key
  LANGSMITH_API_KEY=secretref:langsmith-api-key
  AZURE_STORAGE_ACCOUNT_URL=https://$ST.blob.core.windows.net
  AZURE_STORAGE_QUEUE_URL=https://$ST.queue.core.windows.net
  AZURE_CLIENT_ID=$ID_CLIENT
  LLM_PROVIDER=gemini
  LLM_MODEL=gemini-2.5-flash
)

# ── API ─────────────────────────────────────────────────────────
az containerapp create -n ca-taxlens-api -g $RG --environment $ENV \
  --image $IMAGE --user-assigned $ID_RES \
  --ingress external --target-port 8000 \
  --cpu 1 --memory 2Gi --min-replicas 0 --max-replicas 2 \
  --secrets "${SECRETS[@]}" --env-vars "${ENVVARS[@]}"
# If the GHCR package is private add:
#   --registry-server ghcr.io --registry-username <user> --registry-password <PAT with read:packages>

# ── Indexer job (event-driven, KEDA azure-queue scaler) ─────────
az containerapp job create -n caj-taxlens-indexer -g $RG --environment $ENV \
  --trigger-type Event \
  --replica-timeout 3600 --replica-retry-limit 1 --replica-completion-count 1 --parallelism 1 \
  --min-executions 0 --max-executions 3 --polling-interval 30 \
  --scale-rule-name ingest-queue --scale-rule-type azure-queue \
  --scale-rule-metadata "accountName=$ST" "queueName=ingest-jobs" "queueLength=1" \
  --scale-rule-auth "connection=storage-conn" \
  --image $IMAGE --user-assigned $ID_RES \
  --cpu 1 --memory 2Gi \
  --secrets "${SECRETS[@]}" --env-vars "${ENVVARS[@]}" \
  --command "python" --args "-m" "taxlens.worker.indexer" "--once"

echo "API: https://$(az containerapp show -n ca-taxlens-api -g $RG --query properties.configuration.ingress.fqdn -o tsv)/docs"
```

Notes:

- **No migrations in the cloud.** Cosmos DB is schemaless, and this script creates the containers.
  Item shapes are versioned with `schema_version`. The repository upgrades old items when it reads
  them, and you can add a one-off backfill script if a change really needs it. Alembic is only
  for the local/CI SQLite schema.
- **Cosmos auth:** the app uses `DefaultAzureCredential` (managed identity). If the SDK gets a 403
  while reading account metadata, also assign the identity a custom role containing only
  `.../readMetadata` at scope `/`.
- **First admin user:** `DB_BACKEND=cosmos COSMOS_ENDPOINT=... uv run python scripts/create_user.py --admin`
  from your laptop. It authenticates with your `az login`, using the role assigned to `$ME` above.
- **Scaler auth:** newer Container Apps CLI versions can authenticate KEDA scale rules with a managed
  identity (`--scale-rule-identity`). If yours supports it, use that and delete the `storage-conn`
  secret.
- **Storage clients in code:** if `AZURE_STORAGE_CONNECTION_STRING` is set, use it (Azurite).
  Otherwise use `DefaultAzureCredential(managed_identity_client_id=AZURE_CLIENT_ID)` with the
  account URLs.
- **Bulk-loading the public corpus:**
  `az storage blob upload-batch -d documents --destination-path public/ -s data/raw --account-name $ST --auth-mode login`
  Each file triggers Event Grid → queue → indexer job.

## 8.3 GitHub → Azure with OIDC (no stored Azure secrets)

```bash
APP_ID=$(az ad app create --display-name gh-taxlens-deploy --query appId -o tsv)
az ad sp create --id $APP_ID
SUB=$(az account show --query id -o tsv)
az role assignment create --assignee $APP_ID --role Contributor \
  --scope /subscriptions/$SUB/resourceGroups/rg-taxlens
az ad app federated-credential create --id $APP_ID --parameters '{
  "name": "gh-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:<github-user>/TaxLens:environment:production",
  "audiences": ["api://AzureADTokenExchange"]
}'
```

GitHub repository settings:

- **Secrets:** `AZURE_CLIENT_ID` (= `$APP_ID`), `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
  `GOOGLE_API_KEY`, `GROQ_API_KEY`, `PINECONE_API_KEY`, `LANGSMITH_API_KEY`
- **Environment** `production`, with yourself as required reviewer if you want a manual approval
  before deploys
- **Branch protection** on `main`: require PRs and the `lint`, `test` and `eval` checks

## 8.4 Pipeline: `.github/workflows/ci.yml`

```yaml
name: ci

on:
  pull_request:
  push:
    branches: [main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

env:
  PYTHON_VERSION: "3.12"

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with: { enable-cache: true }
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy taxlens

  test:
    needs: lint
    runs-on: ubuntu-latest
    env:
      APP_ENV: ci
      DB_BACKEND: sqlite
      SQLITE_PATH: var/ci.db
      PHOENIX_COLLECTOR_ENDPOINT: ""
      # unit tests use FakeVectorStore; tests marked `integration` are skipped without PINECONE_API_KEY
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with: { enable-cache: true }
      - run: uv sync --frozen
      - run: uv run alembic upgrade head
      - run: uv run pytest --cov=taxlens --cov-report=term --cov-fail-under=70

  eval:
    needs: test
    # Fork PRs don't get secrets, so skip the eval there instead of failing
    if: github.event_name == 'push' || github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    env:
      APP_ENV: ci
      DB_BACKEND: sqlite
      SQLITE_PATH: var/ci.db
      PHOENIX_COLLECTOR_ENDPOINT: ""
      VECTOR_BACKEND: pinecone
      PINECONE_INDEX: taxlens-ci
      PINECONE_NAMESPACE_PREFIX: ci-${{ github.run_id }}-     # isolates parallel runs
      PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
      GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
      GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
      LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with: { enable-cache: true }
      - run: uv sync --frozen
      - uses: actions/cache@v4
        with:
          path: data/eval_corpus
          key: eval-corpus-${{ hashFiles('data/sources.yaml') }}
      - run: uv run python scripts/download_sources.py --subset ci --dest data/eval_corpus
      - run: uv run alembic upgrade head
      - run: uv run python scripts/seed_index.py data/eval_corpus --wait   # polls describe_index_stats
      - run: uv run python -m eval.run_eval --subset ci --out eval/results.json
      - name: Gate
        run: uv run python eval/check_thresholds.py eval/results.json
      - name: Publish summary
        if: always()
        run: cat eval/summary.md >> "$GITHUB_STEP_SUMMARY"
      - name: Comment on PR
        if: always() && github.event_name == 'pull_request'
        run: gh pr comment ${{ github.event.pull_request.number }} --body-file eval/summary.md --edit-last || gh pr comment ${{ github.event.pull_request.number }} --body-file eval/summary.md
        env: { GH_TOKEN: "${{ github.token }}" }
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: eval-results, path: eval/results.json }
      - name: Delete this run's Pinecone namespaces
        if: always()
        run: uv run python scripts/cleanup_pinecone.py --prefix "$PINECONE_NAMESPACE_PREFIX"

  build:
    needs: eval
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v5          # lower-cases the image name for GHCR
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=sha,format=long
            type=raw,value=latest,enable=${{ github.ref == 'refs/heads/main' }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    permissions:
      contents: read
      id-token: write                            # OIDC
    steps:
      - uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
      - name: Deploy new image
        run: |
          IMAGE=ghcr.io/$(echo "${{ github.repository }}" | tr '[:upper:]' '[:lower:]'):sha-${{ github.sha }}
          az containerapp update     -n ca-taxlens-api      -g rg-taxlens --image "$IMAGE"
          az containerapp job update -n caj-taxlens-indexer -g rg-taxlens --image "$IMAGE"
      - name: Smoke test
        run: |
          FQDN=$(az containerapp show -n ca-taxlens-api -g rg-taxlens --query properties.configuration.ingress.fqdn -o tsv)
          for i in $(seq 1 20); do curl -fsS "https://$FQDN/ready" && exit 0; sleep 10; done
          exit 1
```

Points to explain in an interview:

1. The **eval job is a required status check**, so a PR that drops `hit@5` below 0.85, or drops more
   than 3 points from the baseline, cannot be merged.
2. The PR gets a **comment with the metrics table**, so reviewers see the quality impact next to the diff.
3. Images are tagged with the **commit SHA**. A rollback is
   `az containerapp update --image ...:sha-<previous>` or reactivating the previous revision.
4. **No Azure credentials in GitHub**: OIDC tokens are issued per run and scoped to the
   `production` environment.

## 8.5 Nightly eval (`.github/workflows/nightly-eval.yml`)

Same steps as the `eval` job, plus the repository contract tests against the Cosmos DB emulator
(run as a service container, `pytest -m cosmos`), on `schedule: - cron: "30 20 * * *"` (02:00 IST) with
`--subset all --langsmith --experiment nightly-${{ github.run_number }}`. It never blocks anything,
but it builds the trend line for your README.

## 8.6 Updating the baseline

A `workflow_dispatch` workflow runs the CI subset on `main`, writes `eval/baseline.json`, and opens
a PR titled `chore(eval): update baseline`. The baseline therefore only changes through a reviewed PR.
