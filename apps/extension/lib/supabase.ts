import { Storage } from "@plasmohq/storage"
import { createClient } from "@supabase/supabase-js"

const url = process.env.PLASMO_PUBLIC_SUPABASE_URL
const anonKey = process.env.PLASMO_PUBLIC_SUPABASE_ANON_KEY

if (!url || !anonKey) {
  throw new Error(
    "Missing PLASMO_PUBLIC_SUPABASE_URL / PLASMO_PUBLIC_SUPABASE_ANON_KEY (.env.local 확인)"
  )
}

const storage = new Storage({ area: "local" })

// supabase-js 세션을 chrome.storage(local)에 영속화하는 어댑터.
// Service Worker는 persistent가 아니므로 in-memory 의존 금지 → storage로 복원.
const chromeStorageAdapter = {
  getItem: (key: string) => storage.get<string>(key).then((v) => v ?? null),
  setItem: async (key: string, value: string) => {
    await storage.set(key, value)
  },
  removeItem: async (key: string) => {
    await storage.remove(key)
  }
}

export const supabase = createClient(url, anonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: false,
    storage: chromeStorageAdapter
  }
})
