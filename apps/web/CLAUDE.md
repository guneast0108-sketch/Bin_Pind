# Next.js Web 프론트엔드 컨벤션

루트 `CLAUDE.md` 먼저. 웹 작업 시 추가 규칙.

---

## 디렉토리 구조

```
apps/web/
├── app/                     # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx             # 홈 (URL 입력)
│   ├── places/page.tsx      # 지도 + 장소 목록
│   └── (auth)/login/page.tsx
├── components/              # 이 앱 전용 컴포넌트
├── hooks/                   # 이 앱 전용 훅
├── lib/
│   ├── supabase.ts          # supabase-js 클라이언트
│   ├── api.ts               # 우리 FastAPI 호출 wrapper
│   └── query-client.ts      # TanStack Query
├── stores/                  # Zustand
└── tailwind.config.ts
```

공유 자원은 `packages/ui`, `packages/shared-types` 에서 import.

---

## App Router 단순화 규칙 (가장 중요)

- **모든 페이지 파일 최상단에 `'use client'`** 박을 것.
- **Server Component 사용 금지.**
- **Server Action 사용 금지.** 데이터 호출은 클라이언트에서 (`lib/api.ts` 또는 `supabase-js`).
- 이유: 학부 프로젝트 복잡도 회피. RSC의 학습 비용 ↔ 실익이 안 맞음.
- 예외 둘 필요 생기면 commit 메시지에 명시적 정당화.

이 규칙으로 Next.js를 사실상 **"Vite + 좋은 라우터 + Vercel 배포"**로 활용.

---

## UI 워크플로우 (v0 / Figma)

### v0 → 프로젝트 이식 순서

1. v0에서 컴포넌트 생성 → `apps/web/components/` 에 우선 붙임.
2. **변환 체크리스트**:
   - `'use client'` 유지 (혹은 추가)
   - `next/link` → 일반 `<a href>` (필요 시 props로 `onClick` 받게)
   - `next/image` → 일단 `<img>` (Plasmo 호환 위해)
   - `next/navigation`의 `useRouter` 는 web에서만 사용 → 공유 컴포넌트에는 props로 콜백 받기
   - 색상은 `tailwind.config.ts` 의 디자인 토큰만 사용. 임의 hex 금지.
3. **재사용 명확해진 컴포넌트만** `packages/ui` 로 승격. 처음부터 추상화 X (YAGNI).
4. 승격 기준: web과 extension 양쪽에서 쓰일 게 분명할 때.

### shadcn/ui

- 추가: `npx shadcn@latest add <component>` (예: `button`, `dialog`, `form`)
- 설치 위치: `apps/web/components/ui/` (Next.js 측), 필요 시 `packages/ui/primitives/` 로 복사 후 next 의존 제거.
- 컴포넌트를 직접 수정하면 commit 메시지에 `[shadcn-edit]` 태그.

---

## 상태 관리

- **Zustand** (클라이언트 UI 상태). Redux / Context API 금지.
- **TanStack Query** (서버 상태 — 우리 API 응답 캐싱).
- **Supabase Realtime** 구독은 Custom Hook으로 캡슐화 (예: `useVideoStatus(videoId)`).
- Store 위치: `stores/<domain>Store.ts`. 도메인별 분리.

---

## API 통신

- **TS 타입은 `packages/shared-types/api.ts` 에서 import.** 손으로 작성 절대 X.
- API 호출은 `lib/api.ts` wrapper만 통해서. fetch 직접 호출 금지.
- JWT 헤더는 wrapper에서 자동 첨부 (`supabase.auth.getSession()` → Authorization 헤더).
- 에러 처리: wrapper에서 표준 에러 객체로 변환 → TanStack Query의 `onError`에서 toast.

---

## 지도 (Leaflet)

- **Leaflet + OpenStreetMap**. Mapbox 금지 (유료).
- Next.js SSR 이슈 회피: `dynamic(() => import('./Map'), { ssr: false })` 패턴.
- 마커 클러스터링: 100개 이상부터 `leaflet.markercluster` 활성화.
- 좌표는 항상 `{ lat: number, lng: number }` 객체로 다룰 것 (API와 일치).

---

## 검증

```bash
pnpm typecheck && pnpm lint && pnpm test
```

- 변경 후 실행. tsc 에러는 즉시 수정.
- `ts-ignore` 사용 시 commit 메시지에 사유 명시.

---

## 절대 금지

- TS 인터페이스 손으로 작성
- Server Component / Server Action 사용
- `packages/ui`에 next/* 의존 추가
- API 키 (NEXT_PUBLIC_* 외) 클라이언트 노출
- localStorage 직접 사용 (Supabase auth 상태는 supabase-js가 알아서 관리)
