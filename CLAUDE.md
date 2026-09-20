# Pind — 프로젝트 컨텍스트

영상에서 장소를 추출해 지도로 보여주는 서비스. 학부 풀스택 프로젝트.

> **세션 시작 시 반드시 `PROGRESS.md`를 먼저 읽을 것. 작업 종료 시 갱신할 것.**

---

## 핵심 의사결정 (변경 시 별도 논의 + ADR 갱신 필수)

- **영상 소스**: YouTube URL만 지원. `yt-dlp` 사용. TikTok/Instagram은 v2.
- **Supabase ↔ FastAPI 역할 분담**
  - **Supabase**: Auth(JWT), PostgreSQL+PostGIS, Realtime, Storage, RLS
  - **FastAPI**: AI 파이프라인 (Whisper, Gemini, 지오코딩). 무거운 일 전담.
  - **연결 흐름**: 클라이언트 → Supabase `videos` INSERT → Database Webhook → FastAPI 비동기 처리 → `places` INSERT → Supabase Realtime이 클라이언트로 push
- **AI 전략**: 하이브리드. MVP는 Gemini 단일 비디오 입력(A안), 모듈 인터페이스는 멀티모달 앙상블(B안) 확장 가능하도록 분리 설계.
- **비동기 처리**: FastAPI `BackgroundTasks` + `asyncio`. 동시 처리 5개 초과 시 Redis+RQ로 교체 검토.
- **비용 캡**: 영상당 API 호출 비용 상한을 환경변수로 강제 (`MAX_FRAMES_PER_VIDEO`, `MAX_VIDEO_DURATION_SEC` 등). 호출 전 `cost_guard.check()`.

---

## 디렉토리 구조

```
pind/
├── apps/
│   ├── api/             # FastAPI 백엔드 (Python 3.11+)
│   ├── web/             # Next.js App Router + Tailwind + shadcn/ui
│   └── extension/       # Plasmo Chrome Extension
├── packages/
│   ├── shared-types/    # OpenAPI에서 자동 생성된 TS 타입 (수동 작성 금지)
│   └── ui/              # next/* 의존 없는 순수 React 컴포넌트 (web/extension 공유)
├── supabase/            # 마이그레이션(RLS/함수), seed
├── docker-compose.yml   # 로컬 Postgres+PostGIS
├── Makefile             # make verify, make gen:types, make dev
├── CLAUDE.md            # 이 파일
└── PROGRESS.md          # 진행 현황
```

---

## 운영 원칙 (모든 세션에 적용)

1. **세션 시작**: `PROGRESS.md`를 먼저 읽고 현재 상태 파악.
2. **세션 종료**: 작업한 내용을 `PROGRESS.md`에 반영하고 commit.
3. **타입은 자동 생성**: TS 인터페이스 손으로 작성 금지. Pydantic 변경 → `make gen:types`.
4. **변경 후 검증**: 코드 수정 후 반드시 `make verify` 실행. 실패 시 수정 후 재실행.
5. **컨텍스트 분리**: 한 세션에서 백엔드와 프론트를 동시에 깊게 다루지 말 것. 작은 수직 슬라이스만 예외.
6. **commit 단위**: 한 commit = 한 논리적 변경. 마이그레이션 + 모델 + 스키마는 같은 commit 가능.

---

## 기술 스택 요약

- **Backend**: FastAPI, SQLAlchemy 2.0, Alembic, GeoAlchemy2, Pydantic v2, pytest, ruff, mypy
- **AI**: `google-genai` (Gemini), `faster-whisper`, `yt-dlp`, Google Places API, `sentence-transformers` (dedup)
- **Frontend Web**: Next.js 14+ (App Router, 단순화 규칙 적용), React 18, TypeScript, Tailwind, shadcn/ui, Zustand, TanStack Query, Leaflet
- **Extension**: Plasmo, React, Zustand, `@plasmohq/storage`, Leaflet
- **DB**: PostgreSQL 15 + PostGIS 3 (Supabase Cloud / 로컬은 Supabase CLI)
- **Infra**: Docker Compose (로컬), Render or Fly.io (FastAPI 배포), Vercel (web), Supabase Cloud (DB/Auth)

---

## 절대 금지

- TypeScript 인터페이스 손으로 작성 (자동 생성만 사용)
- Alembic 마이그레이션 없이 SQLAlchemy 모델 변경
- Server Component / Server Action 사용 (App Router이지만 단순화 규칙: 모든 페이지 `'use client'`)
- `packages/ui` 컴포넌트에서 `next/*` 임포트
- API 키 하드코딩 (반드시 환경변수)
- RLS 정책 없는 테이블 (Phase 5 전이라도 최소 정책 동시 추가)
- `service_role` 키를 브라우저/Extension에 노출

---

## 더 자세한 컨벤션

- Backend: `apps/api/CLAUDE.md`
- Web frontend: `apps/web/CLAUDE.md`
- Extension: `apps/extension/CLAUDE.md`
- Supabase: `supabase/CLAUDE.md`

---

## bkit 사용 정책
- 복잡한 feature (Phase 3, 4)는 `/pdca plan <feature>` 부터 시작.
- 단순 scaffolding (Phase 0, 대부분의 Phase 1/2)은 PDCA 생략.
- bkit이 생성한 `docs/0{1..4}/features/*.md` 는 PROGRESS.md의 해당 Phase 항목에 링크.
- PROGRESS.md가 마스터 진행 기록. bkit 문서는 deep-dive.