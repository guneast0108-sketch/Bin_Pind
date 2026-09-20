import { QueryClient } from "@tanstack/react-query";

// 서버 상태(우리 API 응답) 캐싱용 TanStack Query 클라이언트 팩토리.
export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60_000,
        refetchOnWindowFocus: false,
      },
    },
  });
}
