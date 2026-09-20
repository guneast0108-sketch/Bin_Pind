import { useState } from "react"

// Plasmo popup (380×600 기준). CSP 엄격: inline <script>/eval 금지. next/* import 금지.
// 인라인 style은 허용. Tailwind 도입은 후속(JIT/AOT) — 부트스트랩은 인라인 style.
function IndexPopup() {
  const [url, setUrl] = useState("")

  return (
    <div
      style={{
        width: 380,
        minHeight: 200,
        padding: 16,
        boxSizing: "border-box",
        fontFamily: "system-ui, -apple-system, sans-serif"
      }}>
      <h1 style={{ fontSize: 18, fontWeight: 700, margin: 0 }}>Pind</h1>
      <p style={{ fontSize: 13, color: "#666", marginTop: 4 }}>
        유튜브 URL을 붙여넣어 장소를 추출하세요
      </p>
      <input
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        placeholder="https://youtube.com/watch?v=..."
        style={{
          width: "100%",
          marginTop: 12,
          padding: 8,
          boxSizing: "border-box"
        }}
      />
      <p style={{ fontSize: 12, color: "#999", marginTop: 12 }}>
        부트스트랩 완료 · 처리/지도 연결은 Phase 4
      </p>
    </div>
  )
}

export default IndexPopup
