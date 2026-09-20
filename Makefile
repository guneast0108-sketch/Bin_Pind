.PHONY: dev dev-api dev-web verify verify-api verify-js test test-api test-web test-api-integration migrate migrate-down gen-types db-up db-down db-logs help

# ── Settings ──────────────────────────────────────────────────────────────────
API_DIR   := apps/api
WEB_DIR   := apps/web
OPENAPI   := $(API_DIR)/openapi.json
TYPES_OUT := packages/shared-types/src/api.ts
# api 도구는 venv 활성화 없이도 돌도록 .venv/bin 명시 (cd $(API_DIR) 이후 상대경로)
VBIN      := .venv/bin

# ── Development ───────────────────────────────────────────────────────────────
# DB는 Supabase Cloud 직접 연결 → 로컬 docker 불필요.
# (로컬 Postgres+PostGIS가 필요하면 `make db-up`을 별도로 실행)
dev:
	@echo "→ Web(:3000) 백그라운드 + API(:8000) 포그라운드. 중단: Ctrl-C"
	pnpm --filter=./apps/web dev & cd $(API_DIR) && $(VBIN)/uvicorn app.main:app --reload --port 8000

dev-api:
	cd $(API_DIR) && $(VBIN)/uvicorn app.main:app --reload --port 8000

dev-web:
	pnpm --filter=./apps/web dev

# ── Quality / CI ──────────────────────────────────────────────────────────────
# verify는 모노레포 전체를 커버: api(ruff/mypy) + js(web·extension·ui·shared-types)
verify: verify-api verify-js

verify-api:
	cd $(API_DIR) && $(VBIN)/ruff check . && $(VBIN)/ruff format --check . && $(VBIN)/mypy app

# 루트 recursive 스크립트 = web·extension·ui·shared-types lint + typecheck
verify-js:
	pnpm lint
	pnpm typecheck

# ── Tests ─────────────────────────────────────────────────────────────────────
test: test-api test-web

test-api:
	cd $(API_DIR) && $(VBIN)/pytest -x -q --ignore=tests/integration

test-api-integration:
	cd $(API_DIR) && $(VBIN)/pytest -x -q tests/integration

test-web:
	@echo "  (web 테스트는 Phase 2+에서 추가 — 현재 스킵)"

# ── Type Generation ───────────────────────────────────────────────────────────
# 1. Export OpenAPI spec from running FastAPI (없으면 앱에서 직접 덤프)
# 2. Generate TS types with openapi-typescript
gen-types:
	@echo "→ Exporting OpenAPI spec..."
	curl -sf http://localhost:8000/openapi.json -o $(OPENAPI) \
		|| (cd $(API_DIR) && $(VBIN)/python -c "import json; from app.main import app; print(json.dumps(app.openapi()))" > ../../$(OPENAPI))
	@echo "→ Generating TypeScript types..."
	pnpm --filter=@pind/shared-types generate
	@echo "✓ Types written to $(TYPES_OUT)"

# ── Database ──────────────────────────────────────────────────────────────────
# 평소엔 Supabase Cloud 사용. 아래는 로컬 docker-compose Postgres가 필요할 때만.
db-up:
	docker compose up -d db db_test

db-down:
	docker compose down

db-logs:
	docker compose logs -f db

migrate:
	cd $(API_DIR) && $(VBIN)/alembic upgrade head

migrate-down:
	cd $(API_DIR) && $(VBIN)/alembic downgrade -1

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make dev               Web(:3000) + API(:8000) 동시 실행 (DB는 Supabase Cloud)"
	@echo "  make dev-api/dev-web   API 또는 Web만 단독 실행 (두 터미널 워크플로우)"
	@echo "  make verify            전체 검증: api(ruff/mypy) + js(web·extension·ui·shared-types)"
	@echo "  make test              단위 테스트 (api)"
	@echo "  make gen-types         OpenAPI 추출 → TS 타입 생성 (shared-types/src/api.ts)"
	@echo "  make migrate           Alembic 마이그레이션 (head)"
	@echo "  make db-up / db-down   로컬 docker-compose DB (선택)"
	@echo ""
