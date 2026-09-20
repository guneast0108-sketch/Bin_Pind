# Plasmo Chrome Extension 컨벤션

루트 `CLAUDE.md` 먼저. Extension 환경의 특수성을 항상 의식하고 작업.

---

## 디렉토리 구조

```
apps/extension/
├── popup.tsx                # 팝업 UI (380×600px 기준)
├── background.ts            # Service Worker (메시지 라우팅, 인증 토큰)
├── contents/
│   └── youtube.tsx          # 유튜브 페이지에 "Pind에 저장" 버튼 주입
├── components/              # 이 앱 전용 컴포넌트 (재사용 가능해지면 packages/ui로)
├── hooks/
├── lib/
│   ├── supabase.ts          # Extension용 Supabase 클라이언트
│   ├── storage.ts           # @plasmohq/storage 래퍼
│   └── api.ts               # 우리 FastAPI 호출 (web과 동일 인터페이스)
├── stores/                  # Zustand (chrome.storage 백엔드)
└── package.json
```

---

## Extension 환경 특수 규칙

- **CSP 엄격**: inline script, `eval()` 사용 금지. Tailwind는 JIT/AOT 모드만 (런타임 생성 X).
- **localStorage 직접 사용 금지**: 반드시 `@plasmohq/storage` 사용 (chrome.storage 추상화, web/extension 양쪽 동작).
- **popup 크기**: 380×600px 가정하고 레이아웃. `overflow-y-auto` 항상 고려.
- **Service Worker는 persistent 아님**: 모든 상태는 storage에 영속화. in-memory 상태에 의존 금지.
- **외부 도메인 호출**: `manifest.json`(Plasmo는 `package.json`에 명시)의 `host_permissions`에 Supabase, FastAPI 도메인 명시.

---

## 컴포넌트 재사용 정책

- UI 컴포넌트는 **우선 `packages/ui` 에서 import 시도**. 없으면 다음 순서로 진행:
  1. `apps/web` 에서 만들고 검증
  2. 재사용 가치 확인되면 `packages/ui` 로 승격 (next/* 의존 제거)
  3. Extension에서 import

- **`next/*` import 절대 금지**. Plasmo는 Next.js가 아님.
- Extension 전용 임시 컴포넌트는 `apps/extension/components/` 에 두되, 가능한 한 빨리 `packages/ui`로 보내거나 web과 일관성 확보.

---

## 상태 / 인증

### Zustand
- store는 web과 별도. **chrome.storage 백엔드** (`@plasmohq/storage` 어댑터).
- 동기화가 필요한 상태(인증 토큰 등)는 `chrome.storage.sync`, 일반은 `chrome.storage.local`.

### Supabase 인증 흐름
1. Popup에서 "로그인" 버튼 → 새 탭으로 web 로그인 페이지 열기
2. Web에서 로그인 완료 → 토큰을 `postMessage`로 Extension에 전달, 혹은 web이 chrome.storage에 직접 쓰기 (확장 ID 권한 필요)
3. Popup 재로드 → storage에서 토큰 읽어 상태 복원
4. 토큰 갱신은 background script에서 주기적으로 처리

### Supabase 클라이언트
- 별도 인스턴스 (`lib/supabase.ts`).
- `auth` 옵션에서 `persistSession: true`, storage adapter는 chrome.storage.

---

## 진입점별 역할

- **popup.tsx**: URL 붙여넣기 + 빠른 결과 미리보기 + "지도에서 열기" 버튼 (web으로 새 탭)
- **contents/youtube.tsx**: 유튜브 영상 페이지에 플로팅 버튼 주입. 클릭 시 background에 메시지 → 처리 큐 등록
- **background.ts**: content↔popup 메시지 라우팅, 인증 토큰 갱신 스케줄러

---

## 검증

```bash
pnpm typecheck && pnpm lint
pnpm build
# chrome://extensions → "압축해제된 확장 프로그램 로드" → build/chrome-mv3-prod
# popup 동작 수동 확인
```

---

## 절대 금지

- `next/*` import
- localStorage 직접 사용 (chrome.storage / @plasmohq/storage 만)
- API 키 manifest나 코드에 하드코딩
- Service Worker가 persistent라고 가정한 코드 (전역 변수에 상태 저장 등)
- CSP에 `unsafe-inline`, `unsafe-eval` 추가
