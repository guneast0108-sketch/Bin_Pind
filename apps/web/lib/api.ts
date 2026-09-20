import { supabase } from "./supabase";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// API 호출은 반드시 이 wrapper를 통해서만. fetch 직접 호출 금지.
// JWT는 여기서 자동 첨부한다 (supabase 세션 → Authorization 헤더).
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(await authHeader()),
    ...((init?.headers as Record<string, string> | undefined) ?? {}),
  };

  const res = await fetch(`${API_URL}${path}`, { ...init, headers });

  if (!res.ok) {
    throw new ApiError(res.status, `API ${res.status}: ${res.statusText}`);
  }

  return res.json() as Promise<T>;
}
