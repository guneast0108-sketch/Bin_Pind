import { Storage } from "@plasmohq/storage"

// chrome.storage 추상화. localStorage 직접 사용 금지.
// 동기화 필요한 상태(인증 토큰 등)는 sync, 일반 상태는 local.
export const localStore = new Storage({ area: "local" })
export const syncStore = new Storage({ area: "sync" })
