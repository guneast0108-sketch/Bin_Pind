# Supabase / Database 컨벤션

루트 `CLAUDE.md` 먼저.

> **테이블 스키마 자체는 Alembic(`apps/api/alembic/`)에서 관리. 여기서는 RLS 정책, DB 함수/트리거, Webhook 설정만 다룬다.**

---

## 디렉토리 구조

```
supabase/
├── config.toml              # supabase CLI 설정
├── migrations/              # SQL 마이그레이션 (RLS, 함수, 트리거)
│   └── <timestamp>_<name>.sql
├── seed.sql                 # 로컬 시드 데이터
└── functions/               # Edge Functions (현재는 미사용)
```

---

## RLS (Row Level Security)

- **모든 테이블에 RLS 활성화. 예외 없음.**
- 신규 테이블 생성 시 같은 마이그레이션에 RLS enable + 기본 정책 동시 추가.

### 기본 정책 패턴

```sql
-- videos: 본인 소유만 SELECT/INSERT/UPDATE/DELETE
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;

CREATE POLICY "videos_owner_select" ON videos
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "videos_owner_insert" ON videos
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "videos_owner_update" ON videos
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "videos_owner_delete" ON videos
  FOR DELETE USING (auth.uid() = user_id);

-- places: 본인 video 소속만 SELECT (write는 FastAPI service_role만)
ALTER TABLE places ENABLE ROW LEVEL SECURITY;

CREATE POLICY "places_owner_select" ON places
  FOR SELECT USING (EXISTS (
    SELECT 1 FROM videos
    WHERE videos.id = places.video_id
      AND videos.user_id = auth.uid()
  ));
```

### FastAPI의 권한

- **FastAPI는 `service_role` 키 사용** (RLS 우회).
- 단, FastAPI 내부에서 **JWT 검증 + user_id 권한 체크 필수** (e.g. webhook에 들어온 video가 정말 그 user의 것인지).
- service_role 키는 **서버 환경변수만**. 절대 클라이언트에 노출 X.

---

## PostGIS

### 초기화 마이그레이션 (Phase 1-1, Alembic에서 수행)
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

### 규칙
- **SRID 4326** (WGS84) 고정.
- 좌표 저장은 `geography(POINT, 4326)` (정확한 거리 계산을 위해 `geometry`가 아니라 `geography`).
- **공간 인덱스 필수**:
  ```sql
  CREATE INDEX places_geom_idx ON places USING GIST (geom);
  ```
- 반경 검색 예시:
  ```sql
  SELECT * FROM places
  WHERE ST_DWithin(geom, ST_MakePoint(:lng, :lat)::geography, :meters);
  ```

---

## Database Webhook

- `videos` 테이블 **INSERT** 트리거 시 FastAPI `POST /api/v1/webhooks/video-created` 호출.
- Supabase Dashboard → Database → Webhooks 에서 설정.
- **Headers**: `Authorization: Bearer <SUPABASE_WEBHOOK_SECRET>`.
- FastAPI 측에서 secret 검증 후 처리. 미검증 요청은 401.
- 페이로드는 Supabase 표준 형식 (`type: INSERT`, `record: {...}`, `old_record: null`).

---

## Realtime

- `videos.status` 변경을 클라이언트가 구독.
- `places` INSERT도 구독 (지도에 마커 실시간 추가).
- Publication에 테이블 추가:
  ```sql
  ALTER PUBLICATION supabase_realtime ADD TABLE videos, places;
  ```
- RLS가 Realtime에도 적용됨 → 본인 데이터만 받음.

---

## 로컬 개발 워크플로우

```bash
# 1. 로컬 Supabase 띄우기 (한 번만)
supabase start

# 2. Alembic 마이그레이션 적용 (스키마 변경 시)
cd apps/api && alembic upgrade head

# 3. RLS / 함수 마이그레이션 적용
cd ../.. && supabase db push   # supabase/migrations/ 의 .sql 적용

# 4. 시드 데이터
psql $SUPABASE_DB_URL -f supabase/seed.sql
```

---

## 절대 금지

- RLS 비활성화 상태로 production 배포
- `service_role` 키를 브라우저/Extension에 노출
- DB 함수에서 `SECURITY DEFINER` 사용 시 정당화 없이 (권한 상승 위험 — 필요 시 반드시 commit 메시지에 사유)
- `auth.uid()` 검증 없는 정책
- Realtime publication에 테이블만 넣고 RLS 미설정
