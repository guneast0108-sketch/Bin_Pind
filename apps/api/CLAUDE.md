# FastAPI 백엔드 컨벤션

루트 `CLAUDE.md` 를 먼저 읽었다고 가정. 백엔드 작업 시 추가로 적용되는 규칙.

---

## 디렉토리 구조

```
apps/api/
├── app/
│   ├── main.py                  # FastAPI 앱 entry
│   ├── settings.py              # pydantic-settings (모든 env 변수)
│   ├── deps.py                  # 공통 의존성 (get_db, get_current_user)
│   ├── exceptions.py            # 도메인 예외
│   ├── models/                  # SQLAlchemy 모델 (DB 스키마)
│   │   ├── base.py
│   │   ├── video.py
│   │   └── place.py
│   ├── schemas/                 # Pydantic 스키마 (DTO)
│   ├── routers/                 # FastAPI 라우터 (/api/v1/...)
│   ├── pipeline/                # AI 파이프라인 — 모두 순수 함수
│   │   ├── types.py             # PlaceCandidate, Transcript, Frame 등
│   │   ├── cost_guard.py        # 비용 캡 가드
│   │   ├── download.py          # yt-dlp
│   │   ├── audio.py             # ffmpeg 오디오 추출
│   │   ├── transcribe.py        # faster-whisper (B안용)
│   │   ├── frames.py            # ffmpeg keyframe 추출 (B안용)
│   │   ├── analyze_video.py     # Gemini 단일 비디오 입력 (A안)
│   │   ├── analyze_audio.py     # Gemini 텍스트 분석 (B안 인터페이스)
│   │   ├── analyze_vision.py    # Gemini Vision (B안 인터페이스)
│   │   ├── resolve.py           # 후보 dedup + Google Places 지오코딩
│   │   ├── orchestrator.py      # 파이프라인 전체 흐름
│   │   └── vision/              # 비전 프론트엔드 — VLM 앞단 프레임 선별·복원
│   │       ├── types.py         # QualityMetrics, FrameStats, Keyframe, VisionFrontendConfig
│   │       ├── decode.py        # ffmpeg 균일 샘플링
│   │       ├── quality.py       # 무참조 화질 지표 + readability_score
│   │       ├── textness.py      # MSER 간판 문자 saliency
│   │       ├── isp.py           # 미니 ISP (WB/감마/디노이즈/CLAHE/언샤프)
│   │       ├── embed.py         # Embedder 프로토콜 (pHash+HSV / DINOv3)
│   │       ├── dedup.py         # 샷 분할 + 중복 샷 제거
│   │       └── select.py        # select_keyframes 오케스트레이터
│   └── webhooks/                # Supabase Webhook 핸들러
├── benchmarks/                  # 성능 측정 (bench_keyframes · validate_recall · youtube_fetch)
│   └── youtube/                 # 유튜브 검증 매니페스트·정답 (영상은 .cache/, 커밋 안 함)
├── alembic/                     # 마이그레이션
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                # sample.mp4 (5초 더미) 등
├── pyproject.toml               # ruff, mypy, pytest 설정
└── Dockerfile
```

---

## SQLAlchemy 규칙

- **SQLAlchemy 2.0 스타일만**. `DeclarativeBase` + `Mapped[T]` annotation. 1.x 레거시 스타일 금지.
- **PK는 UUID**: `Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)`.
- **모든 테이블에 `created_at`, `updated_at`** (`timestamp with timezone`, UTC).
- **좌표**: GeoAlchemy2 `Geography(POINT, srid=4326)`. **GIST 인덱스 필수**.
- **FK 컬럼**: 명시적 인덱스 추가.
- **Enum**: SQLAlchemy `Enum` 대신 Python `enum.StrEnum` + `String` 컬럼. 마이그레이션 호환성 위해.
- 변경은 **반드시 Alembic 마이그레이션**과 함께. 모델만 바꾸고 마이그레이션 빼먹지 말 것.

---

## Pydantic 규칙

- Pydantic v2 사용.
- 스키마는 **`Read`, `Create`, `Update` 분리** (예: `PlaceRead`, `PlaceCreate`, `PlaceUpdate`).
- `Read` 스키마: `model_config = ConfigDict(from_attributes=True)`.
- 좌표는 API 표면에 **`lat: float, lng: float`**으로 노출 (geom → lat/lng 변환 함수 작성).
- `datetime`은 UTC, ISO 8601.
- 컬럼명 ↔ 필드명 매핑: snake_case 통일 (DB도 API도).

---

## AI 파이프라인 규칙

### 함수 시그니처 원칙
- 모든 파이프라인 함수는 **순수 함수** (DB 의존 X). 입출력은 Pydantic 모델.
- 파일 의존은 `pathlib.Path`로 명시. 임시파일은 함수 내부에서 `tempfile.TemporaryDirectory`.

### A안 / B안 인터페이스 분리 (중요)
```python
# pipeline/types.py
class PlaceCandidate(BaseModel):
    name: str
    category: str | None
    context_start_sec: int
    context_end_sec: int
    confidence: float
    source_modality: Literal["audio", "vision", "text", "video"]
    raw_extracted_text: str

# A안 (MVP 구현)
async def analyze_video(video_path: Path) -> list[PlaceCandidate]: ...

# B안 (인터페이스만 정의, 구현은 v2)
async def analyze_audio(transcript: Transcript) -> list[PlaceCandidate]: ...
async def analyze_vision(frames: list[Frame]) -> list[PlaceCandidate]: ...
```
나중에 정확도 비교 실험 시 orchestrator만 갈아끼우면 되도록.

### 비전 프론트엔드 (`pipeline/vision/`)

- 공개 진입점은 `select_keyframes` 하나. 나머지 모듈은 순수 함수 집합.
- **저조도 복원은 화질 게이트보다 먼저 실행한다.** 이 순서를 뒤집으면 실내 저조도 프레임이
  전부 탈락해 실내 장소가 결과에서 사라진다. 게이트는 "품질 낮은 프레임"이 아니라
  "복원해도 못 읽는 프레임"만 거른다.
- 디노이징 판정은 **감마 후** σ 기준. 원본 σ는 어두운 프레임에서 과소평가된다.
- 프레임 픽셀은 청크(32장) 단위로만 메모리에 두고, 최종 선별된 프레임만 다시 읽는다.
- **감축률을 성능 지표로 쓰지 말 것.** 프레임을 1장만 남기면 감축률은 98%가 된다. 임계값이나
  선별 로직을 바꿨으면 `benchmarks/validate_recall.py`로 **장소 보존률**을 재고, 오라클
  (선별 없이 전 프레임 OCR)과 비교한다. `bench_keyframes.py`의 표는 비용 쪽만 말해준다.
- **판정을 파이프라인 안에서 하지 말 것.** 프레임이 읽히는지의 판정은 외부 OCR(tesseract)이
  한다. `textness`로 판정하면 선별 점수에 이미 들어간 값으로 자기를 채점하는 셈이다.
- **선명도 임계는 시간상 이웃 기준**이다(전역 퍼센타일 아님). 선명도는 흔들림만의 함수가
  아니라 장면의 텍스처 양에도 좌우되므로, 전역 하위 N% 컷은 단조로운 장면을 통째로 버린다.
- **장면 유사도 임계는 영상별로 보정된다**(`max_shot_cut_rate`). 고정값은 패닝 영상에서
  한 장소를 여러 샷으로 쪼개고, 예산이 흩어져 다른 장소가 사라진다.
- **예산 컷은 점수 순이 아니라 장면 다양성 순**(k-center greedy). 예산의 목적은 "가장 예쁜
  프레임 K장"이 아니라 "서로 다른 장소 K곳"이다.
- **보정이 항상 개선이라고 가정하지 말 것.** 전체 ISP는 문자 영역 점수가 떨어지면 값싼
  구제본으로 되돌린다(`enhance_textness_drop_limit`). 저조도에서 디노이징이 간판 획을 지운다.
- 설계 근거·측정값·한계: `docs/vision-frontend.md`

### 외부 API 호출
- `tenacity`로 재시도 데코레이트:
  - Gemini 429 / 503: exponential backoff, max 3회
  - 4xx (429 제외): 재시도 X
  - 5xx: 재시도 OK
- 호출 전 `cost_guard.check(video_id, estimated_cost)` 필수. 초과 시 `CostLimitExceeded` 예외.

---

## FastAPI 컨벤션

- 라우터 prefix는 **`/api/v1/`**. 버저닝 처음부터.
- `response_model=...` 로 응답 스키마 명시.
- 페이지네이션은 **cursor-based** (offset 금지). `limit`, `cursor` 쿼리 파라미터.
- 에러는 도메인 예외(`PlaceNotFound`, `CostLimitExceeded` 등) → 미들웨어에서 HTTP 응답으로 변환.
- 모든 엔드포인트에 `tags` 지정.
- **Webhook 엔드포인트는 Bearer 토큰(서비스 시크릿) 검증 필수**.

---

## 테스트

- 단위 테스트: `pytest` + `pytest-asyncio`. 외부 API는 `respx`로 mock.
- 통합 테스트: `httpx.AsyncClient` + docker-compose의 test DB.
- 새 라우터: happy path + 최소 1개 에러 케이스.
- `tests/fixtures/sample.mp4` (5초, <1MB)를 파이프라인 통합 테스트에 사용.
- AI 호출 통합 테스트는 별도 마커(`@pytest.mark.live`) — 평소엔 skip, 수동으로만 실행.

---

## 검증

```bash
make verify  # = ruff check . && ruff format --check . && mypy app && pytest
```

- 코드 변경 후 반드시 실행.
- 자동 수정 가능한 건 `ruff format .`, `ruff check --fix .`.

---

## 절대 금지

- 모델 변경 없이 마이그레이션 작성, 또는 그 반대
- 파이프라인 함수 안에서 DB 직접 접근
- API 키를 코드에 하드코딩 (반드시 `settings.py` 경유)
- Webhook 엔드포인트에 시크릿 검증 누락
- `print()` 로 로깅 (`structlog` 사용)

---

### Gemini 모델 버전

- **현재 사용 모델**: `gemini-2.5-flash` (stable GA)
- **절대 사용 금지**: `gemini-2.0-*` (2026-06-01 shutdown), `gemini-1.5-*` (이미 404)
- **Gemini 3 시리즈**: 현재 대부분 preview — GA 전환 전까지 사용 금지.
  Google I/O 2026 (2026-05-19) 이후 `gemini-3.x-flash` GA 여부 확인 후 재검토.
- 모델명은 환경변수 `GEMINI_MODEL`로 관리. 코드 하드코딩 금지.