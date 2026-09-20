# 유튜브 영상 장소 보존률 검증

합성 장면(`make_recall_scene.py`)에서 오라클 동률(8/8)을 확인한 하네스를 **Pind 가 실제로 받는
입력인 유튜브 여행 영상**에 돌린다. 직접 찍은 영상 대신 유튜브를 쓰는 이유는 분포다 — 서비스
입력은 재압축(720p H.264 수 Mbps)·편집 컷·편집 자막·색보정을 거친 영상이고, 촬영 원본은 그중
어느 것도 갖고 있지 않다.

```bash
cd apps/api
brew install tesseract tesseract-lang        # 한글 OCR 판정자 (Ubuntu: tesseract-ocr-kor)

python benchmarks/youtube_fetch.py           # 1) 다운로드 · 콘택트 시트 · 정답 초안
#                                              2) truth/<key>.json 의 texts·segments 채우기
python benchmarks/validate_recall.py \
  --manifest benchmarks/youtube/manifest.json --out-dir recall_out/youtube   # 3) 평가
```

## 개발 세트와 테스트 세트

`manifest.json` 의 `split` 이 `dev` 인 3편은 **프레임 선별 방법을 고르는 데 이미 썼다** (실측 4/26 → 원인 분해 →
`text_nms` 선택). 방법이 나아졌는지는 `split: test` 영상으로만 판단한다. 비교 설정·지표·채택 규칙은
[`PREREGISTRATION.md`](PREREGISTRATION.md)에 결과를 보기 전에 고정했다.

```bash
pip install -e '.[bench,vision-textdet]' && python -m app.pipeline.vision.textdet --fetch
python benchmarks/youtube_fetch.py --split test          # 테스트 영상 + 콘택트 시트
# → 제외 기준 확인 · 구간 결정 · 정답 라벨 커밋 (결과 보기 전)
python benchmarks/validate_recall.py --manifest benchmarks/youtube/manifest.json --split test \
  --ablations full,text-det --embedder perceptual --full-table --out-dir recall_out/youtube-test
```

## 파일

| 경로 | 커밋 | 내용 |
|---|---|---|
| `manifest.json` | O | 영상 URL · 포맷 선택자 · 평가 구간(`clip`) · 역할 |
| `truth/<key>.json` | O | 사람이 채운 정답. 첫 fetch 때 챕터로 초안이 생긴다 |
| `.cache/<key>/source.mp4` | X | 받은 영상(영상 트랙만). 저작권 때문에 커밋하지 않는다 |
| `.cache/<key>/info.json` | X | 제목·채널·챕터·포맷 id·해상도·코덱·**sha256** |
| `.cache/<key>/sheets/` | X | 라벨링용 콘택트 시트(3초 간격 4×4, 원본 시각 표기) + `index.md` |
| `.cache/ocr/` | X | OCR 캐시. 한글 모델이 느려 ablation 4종 × 오라클을 캐시 없이 돌리면 오래 걸린다 |

## 라벨링 규칙 — 결과를 보기 전에 확정한다

**정답은 파이프라인을 돌리기 전에 채우고 커밋한다.** 결과를 본 뒤 라벨을 고치면(놓친 장소를
지우거나 구간을 넓히면) 측정이 오염된다. 라벨을 고쳐야 할 이유가 생기면 커밋 메시지에 이유를 남긴다.

1. **장소** — 영상이 머무르거나 소개하는 가게·명소. 걷는 영상에서 스치는 간판은, **정지 화면에서
   상호를 읽을 수 있고 3초 이상 보이는 경우만** 장소로 센다. 인트로·이동 구간 챕터는 지운다.
2. **`texts`** — 화면에 **실제로 보이는** 문자열만. 챕터 제목·영상 설명의 이름을 옮겨 적지 않는다
   (`name` 에만 둔다). 간판에 영문과 한글이 함께 있으면 둘 다 적는다.
   - `sign` — 간판·상호 표지·입구 현판
   - `caption` — 제작자가 편집으로 넣은 자막
   - `other` — 메뉴판·영수증·포장지
3. **`segments`** — 그 **문자열이 읽을 수 있게 보이는 구간**. 방문 구간 전체가 아니다. 챕터 초안의
   구간은 넓으니 좁힌다. 같은 장소를 다시 보여주면 구간을 여러 개 적는다. 시각은 원본 영상 기준
   (`"5:12"` 형식 가능) — 클립을 잘라도 그대로 둔다.
4. **`dark`** — 기본 `"auto"`(구간 휘도 중앙값 < 0.18). 판단이 갈리면 명시한다.
5. **평가 구간(`clip`)** — 1fps 기준 **15분(900장) 이하**여야 한다(디코딩 상한). 장소가 키프레임
   예산(기본 16장)보다 많으면 보존률 상한이 예산/장소 수로 묶이므로 5~10분, 장소 10곳 안팎을 권장.

## 실행 시간 (대략)

영상 한 편에 파이프라인 4회(ablation) + 선별 프레임 OCR + 오라클(구간 안 샘플 프레임 OCR)이 돈다.
EasyOCR 은 CPU에서 프레임당 1~5초라 **10~15분 구간 한 편에 수십 분**이 걸린다. OCR 캐시가 있어
두 번째 실행부터는 파이프라인 시간만 든다. 먼저 `--only <key> --ablations full` 로 한 편을 확인하고
전체를 돌리는 것을 권한다.

## 보고서 읽는 법

| 열 | 뜻 |
|---|---|
| 장소 보존 | 선별 프레임에서 정답 문자열(간판·자막·기타 무엇이든)이 읽힌 장소 |
| 간판으로 | 그중 **간판으로** 읽힌 장소 — 화질 처리의 기여는 이 열로 본다 |
| 오라클 | 선별 없이 모든 샘플 프레임을 OCR한 상한. 오라클도 못 읽은 장소는 파이프라인 탓이 아니다 |
| 잃은 장소 → 원인 | 게이트 탈락(임계·구제) / 선별 제외(샷 분할·점수·예산) / 읽기 실패(보정) |

영상이 자막으로 상호를 보여주면 `장소 보존`은 높고 `간판으로`는 낮게 나올 수 있다. 그 차이가 곧
"이 영상에서는 보정이 없어도 장소를 찾는다"는 뜻이므로 둘을 섞어 하나의 숫자로 보고하지 않는다.

## 영상 교체

`manifest.json` 의 후보 3편은 역할(야간 저조도 / 편집 브이로그 / 주간 대조군)로 고른 것이고, 간판이
실제로 충분히 보이는지는 콘택트 시트로 확인해야 한다. 조건에 안 맞으면 같은 역할의 다른 영상으로
바꾼다. 한 번 수치를 낸 뒤에는 `sha256` 을 매니페스트에 적어 두면 유튜브 쪽 재인코딩을 감지한다.

## 문제 해결

- `Requested format is not available` / `n challenge` — `pip install -U yt-dlp`, 필요하면 `brew install deno`
- `tesseract 언어 데이터가 없습니다: ['kor']` — `brew install tesseract-lang`
- `샘플 N장이 디코딩 상한을 넘습니다` — `clip` 을 15분 이하로
