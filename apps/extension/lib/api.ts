import { supabase } from "./supabase"

const API_URL = process.env.PLASMO_PUBLIC_API_URL ?? "http://localhost:8000"

// web의 lib/api.ts와 동일 인터페이스. JWT는 supabase 세션에서 자동 첨부.
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message)
    this.name = "ApiError"
  }
}

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(await authHeader()),
    ...((init?.headers as Record<string, string> | undefined) ?? {})
  }

  const res = await fetch(`${API_URL}${path}`, { ...init, headers })

  if (!res.ok) {
    throw new ApiError(res.status, `API ${res.status}: ${res.statusText}`)
  }

  return res.json() as Promise<T>
}
