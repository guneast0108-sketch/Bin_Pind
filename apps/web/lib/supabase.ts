import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!url || !anonKey) {
  throw new Error(
    "Missing NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY (.env.local 확인)",
  );
}

// 브라우저용 Supabase 클라이언트. anon 키만 사용 (service_role 노출 금지).
// auth 세션은 supabase-js가 자동 관리하므로 localStorage 직접 접근 금지.
export const supabase = createClient(url, anonKey);
