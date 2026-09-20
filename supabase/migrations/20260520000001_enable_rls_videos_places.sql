-- Migration: RLS policies for videos and places
--
-- 적용 순서:
--   1. Phase 1-1: Alembic이 videos, places 테이블 생성
--   2. 이 파일: supabase db push 로 RLS 활성화 + 정책 추가
--
-- FastAPI는 service_role 키를 사용하므로 RLS를 자동 우회함.
-- 클라이언트(anon / authenticated)는 아래 정책으로만 접근 가능.

-- ─────────────────────────────────────────────────────────────────────────────
-- videos
--   소유자(user_id = auth.uid())만 CRUD 허용.
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE videos ENABLE ROW LEVEL SECURITY;

CREATE POLICY "videos_owner_select" ON videos
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "videos_owner_insert" ON videos
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "videos_owner_update" ON videos
  FOR UPDATE USING (auth.uid() = user_id)
              WITH CHECK (auth.uid() = user_id);

CREATE POLICY "videos_owner_delete" ON videos
  FOR DELETE USING (auth.uid() = user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- places
--   SELECT 전용: 본인 video에 속한 place만 조회 가능.
--   INSERT/UPDATE/DELETE: 정책 없음 → RLS 기본 deny (FastAPI service_role만 write).
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE places ENABLE ROW LEVEL SECURITY;

CREATE POLICY "places_owner_select" ON places
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM videos
       WHERE videos.id   = places.video_id
         AND videos.user_id = auth.uid()
    )
  );


-- ─────────────────────────────────────────────────────────────────────────────
-- Realtime publication
--   RLS가 Realtime에도 적용되므로 본인 row 변경만 수신됨.
-- ─────────────────────────────────────────────────────────────────────────────

ALTER PUBLICATION supabase_realtime ADD TABLE videos, places;
