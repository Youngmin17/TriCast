# TriCast Studio

가상 텐서코어 누산 알고리즘을 설계하고 모델을 고르면, 같은 입력에 대한 **원본(native) 출력**과 **TriCast
에뮬레이션 출력**을 나란히 보여 주는 웹 앱. LLM 텍스트 생성, 이미지 객체 검출, 이미지 분류를 지원한다.
실제 GPU 의 preset (Hopper 등) 은 출처가 있는 시작점일 뿐이고, 사용자는 F·CS·C 결합 방식 등을 직접 바꾼다.

```
브라우저 (app/web, 정적 SPA)
   │  GET /api/catalog · POST /api/runs · GET /api/runs/{id}
   ▼
FastAPI (app/server.py) ── 작업 큐 (app/jobs.py, GPU 1개를 순차 사용)
   │
   ├─ app/catalog.py     알고리즘 파라미터 공간·preset·형식·모델·작업 (출처·검증 상태 포함)
   ├─ app/runners/llm.py     native vs 에뮬레이션 생성 + 기준 토큰 강제 디코딩 분포 비교
   ├─ app/runners/vision.py  YOLO11n 검출 / ResNet18 분류
   └─ app/metrics.py     비교 지표 (순수 함수)
```

실행 모드 (화면 오른쪽 위에서 바꾼다)

| 모드 | 무엇을 하나 | 필요한 것 |
|---|---|---|
| 예시 데이터 | GPU 에서 미리 실행해 둔 설계점 스윕을 즉시 탐색. 입력은 기록된 것만 | `app/web/demo/` 묶음. GPU·모델 없이 정적 HTTP 서버로 연다 (`file://` 로 직접 열면 브라우저가 모듈 로딩을 막는다) |
| 서버 GPU | 직접 넣은 프롬프트·이미지를 서버(클러스터) GPU 에서 실시간 실행. 원본 결과를 먼저 보여 주고, 같은 요청은 캐시에서 바로 연다 | `python -m app.server --device cuda` 를 GPU 노드에서 실행 (`/api/resources` 로 GPU·대기열·캐시된 모델을 확인) |
| 브라우저 GPU | 이 컴퓨터의 GPU(WebGPU)에서 FP8 행렬곱 하나를 설계한 산술로 실행하고, JS 정확 레퍼런스·서버 TriCast 결과와 비트 대조 | WebGPU 를 지원하는 브라우저. 모델 전체는 아직 서버 GPU 모드 전용 |

1차 대상 모델: Qwen3-0.6B, Llama-3.2-1B, YOLO11n, ResNet18.

실행
- 데모 모드 (GPU·모델 없이, 기록된 실행 + 브라우저 GPU 실험실): 저장소 루트에서 `pip install fastapi uvicorn` 후
  `python -m app.server --demo` → http://127.0.0.1:8765
- 라이브 모드 (CUDA + Triton): `pip install -r app/requirements.txt` 후 `python -m app.server --device cuda`
- 데모 묶음 생성 (GPU): `python -m app.demo --device cuda` (기본 출력 `runs/studio_demo/`, git 이 무시하는 경로) 후
  `catalog.json`·`index.json`·`demo_scores.json`·`runs/`·`media/*.jpg` 를 `app/web/demo/` 로 복사한다 (`*_compare.png` 제외).
  추적되는 경로에 바로 기록하면 첫 run 뒤의 모든 run 이 `env.git_dirty: true` 로 남는다. 커밋된 clean 트리에서 기록한다
- 테스트: `python -m pytest app/tests -q` (저장소 CPU suite 와 분리)

## 화면 구성 (UI/UX)

계측기 콘솔 구조다. 왼쪽 레일에서 실험을 구성하고, 오른쪽에서 두 채널(A = 기준, B = 에뮬레이션)을 비교한다.

| 영역 | 내용 |
|---|---|
| 1 작업 | 텍스트 생성 · 객체 검출 · 이미지 분류 |
| 2 모델 | 작업별 모델과 검증 수준 배지 (전체 평가 / 연산자 검증) |
| 3 누산 알고리즘 | preset 시작점(출처·검증 상태 툴팁) → 알고리즘 종류 → 파라미터. 데이터패스 글리프(남는 가수 비트 F, FP32 경계)와 데이터플로 도식이 파라미터에 따라 바로 바뀐다. preset 과 다르면 "사용자 정의 가상 설계"로 표시 |
| 4 형식과 비교 기준 | 입력 양자화 형식, 기준선 (원본 / 같은 형식 + FP64 누산) |
| 5 입력 | 프롬프트 또는 이미지. 데모는 기록된 입력만, 라이브는 직접 입력·업로드 |
| 결과 · 비교 | 기록된 사례 → 실행 헤더(알고리즘·출처·형식·backend·GPU·소스) → A/B 채널 → 지표 → 설계 공간 차트 → (LLM) 위치별 KL |
| 결과 · 재현 정보 | 환경 기록, 에뮬레이션 실행 증거(패치 수·호출 수), 레시피 YAML, 읽는 법 |

흐름
- 데모 모드: 파라미터를 바꾸면 방금 바꾼 값을 유지하면서 가장 가까운 기록 설계점으로 맞추고 바로 연다. 슬라이더 아래 점이
  기록된 값이다. 설계 공간 차트의 점을 눌러도 그 설계로 바뀐다.
- 라이브 모드: 조합을 MMASpec 규칙 안으로 보정하고(승격 주기는 CS 의 배수, GDFS 타일은 1–8 그룹), "비교 실행" 으로
  작업을 큐에 넣는다. 진행 단계(모델 로드 → 기준 실행 → 에뮬레이션 연결 → 에뮬레이션 실행 → 지표 계산)를 보여 준다.
  실행 중에는 레일이 잠겨 결과가 항상 보낸 설계와 맞는다. 서버가 캐시에서 돌려준 결과에는 "서버 캐시" 표시가 붙는다.
- LLM: 첫 분기 토큰을 주황으로 표시하고, 분기 이후는 위치별 비교를 하지 않는다. 토큰에 마우스를 올리면 상위 5개 후보와 확률.
- 검출: 상자에 마우스를 올리면 다른 채널의 짝이 강조된다. 한쪽에만 있는 상자는 주황 점선, "차이 나는 상자만" 필터.
- 주소 끝 `#llm`, `#detect`, `#classify` 로 작업을, `#webgpu` 로 브라우저 GPU 실험실을 바로 연다.

상태: 기록 없음(가까운 기록 제안), 대기·실행 중(단계 표시), 오류(원인과 코드). 라이트·다크 테마, 휴대폰 폭에서는 레일이 위로 쌓인다.

## 가상 알고리즘과 형식 → 레시피

사용자는 **누산 알고리즘**(MMASpec 필드: `algorithm` 과 그 파라미터)과 **입력 형식**(QuantSpec scheme)을 고른다.
서버는 둘을 레시피 dict 로 합쳐 `tricast.load_recipe` 로 검증한다 (`MMASpec` 규칙: F·F2·G ∈ [1, 48], 승격 주기는
CS 의 배수이며 cofda 전용, GDFS 는 k_tile / group_size ∈ [1, 8]). 번들 레시피와 같은 조합이면 그 이름을 함께 기록한다.

```python
{"name": "studio:<mma 요약>:<format>",
 "defaults": {"weight": <format|None>, "activation": <format|None>, "mma": <mma dict>,
              "transform": "none", "weight_algo": "rtn"},
 "include": ["*"], "exclude": ["lm_head"], "backend": "auto"}
```

**정규형 mma**: 알고리즘의 파라미터 스키마(카탈로그 `algorithms[].params`)에서 `when` 조건이 맞는 키만, 빠진 값은
`default` 로 채운 dict. 서버(`app/catalog.py` 의 `canonical_mma`)는 여기에 더해 null·미지 키·범위 밖 값을 거부하고
정수로 맞춘다. GUI(`app/web/js/algo.js` 의 `canonical`)는 같은 스키마로 기본값만 채운다. 기록된 run 은 정규형의
동등성으로 찾는다.

보정이 필요한 레시피(GPTQ, AWQ, SmoothQuant, 정적 observer)는 v0 에서 제외한다. 비전(Conv2d)은 TriCast 계약상
inference·동적 양자화·RTN 만 허용하므로 transform 이 있는 조합은 비전 작업에서 거부한다.

## API

모든 응답은 JSON. 오류는 `{"error": {"code": str, "message": str}}` 와 4xx/5xx.

데모 모드(`--demo`)는 정적 파일과 `/api/health`·`/api/catalog`·`/api/resources` 만 제공하고, torch·tricast 를
import 하지 않는다. GUI 는 기록된 실행을 정적 파일(`demo/index.json`, `demo/runs/<id>.json`)로 읽는다. 그 밖의
`/api/*` 는 JSON 404 이고, 아래의 실행 API(`/api/runs*`, `/api/media`)는 라이브 모드에서만 있다.

`GET /api/catalog`
```ts
{
  tricast: {version: string, git_sha: string | null},
  mode: "live" | "demo",
  device: {kind: "cuda" | "cpu" | "none", name: string | null, backend: "triton" | "reference" | null},
  tasks: [{id: "llm.generate" | "vision.detect" | "vision.classify", kind: "llm" | "vision",
           label: string, description: string}],
  models: [{id: string, label: string, tasks: string[], revision: string | null,
            support: "full_eval" | "operator" | "experimental", note: string}],
  algorithms: [{id: "cofda" | "gdfs" | "fp32_fma" | "fp64", label: string, description: string,
                formats: (string | null)[],      // 이 알고리즘과 쓸 수 있는 입력 형식
                params: [{key: string, label: string, kind: "int" | "choice",
                          min?: number, max?: number,        // int: MMASpec 이 허용하는 범위
                          ui_min?: number, ui_max?: number,  // int: 슬라이더에 보일 범위
                          options?: (number | string)[], default: number | string,
                          when?: {[key: string]: number | string},   // 이 조건일 때만 의미가 있는 파라미터
                          advanced?: boolean, unit?: string, help: string}]}],
  presets: [{id: string, label: string, source: string,   // 예: "NVIDIA Hopper 모델링 (NADPE)"
             mma: <정규형 mma>, format: string | null,      // preset 이 전제하는 입력 형식
             status: "reference" | "modeled" | "partial_mismatch" | "design_point",
             provenance: string, status_note: string}],
  formats: [{id: string | null, label: string, bits: number | null, note: string}],
  baselines: [{id: "native" | "same_quant_fp64", label: string, description: string}]
}
```

`GET /api/resources` → 데모: `{mode: "demo", server: null}` / 라이브: `{mode: "live", server: {hostname, device, gpus: [{index,
name, memory_total_mb, memory_used_mb, utilization_pct}], torch, triton, cuda, queue: {queued, running}, models_cached: string[],
runnable_tasks: string[]}}`

`POST /api/runs` → `202 {"id": string}`. 같은 정규 요청이 같은 서버 식별자(장치, app·TriCast 소스 해시)로 대기·실행
중이면 그 id 로 202, 이미 끝나 있으면 `200 {"id", "cached": true}`. 장치나 코드가 바뀌면 새로 실행한다. 오류로 끝난
요청은 다시 실행한다. 라이브 서버는 시작할 때 seed 42·결정론 커널·TF32 끄기를 적용한다 (데모 묶음과 같은 설정). 결과는 `runs/studio/<id>.json` 에 원자적으로 저장되고,
서버를 다시 띄우면 최근 50건을 읽어 목록·캐시가 이어진다 (업로드 이미지는 `runs/studio/uploads/`).
```ts
{task: string, model: string, mma: <mma dict>, format: string | null, baseline?: "native" | "same_quant_fp64",
 input: {prompt?: string, max_new_tokens?: number,            // llm.generate (max_new_tokens ≤ 64)
         image?: string /* demo media id 또는 업로드 data URL */}}
```

`GET /api/runs` → `{runs: [{id, status, task, model, mma, format, baseline, input, mma_label, created_utc, summary,
preview}]}` (최신순; `summary`·`preview` 는 index 항목과 같은 형식). `GET /api/media/{name}` → 이미지 파일.

`GET /api/runs/{id}`
```ts
{
  id: string, status: "queued" | "running" | "done" | "error",
  stage: "load" | "baseline" | "patch" | "emulated" | "metrics" | null, progress: number /* 0..1 */,
  cached_baseline?: boolean,          // 원본 결과를 runner 캐시에서 가져왔는지
  // status 가 running 이어도 원본 출력이 나오면 baseline 이 먼저 채워진다 (runner 의 progress(stage, fraction, partial))
  request: <정규화한 POST body: mma 는 정규형, 업로드 data URL 은 저장 파일 이름, max_new_tokens 는 기본값 32 로 채움>,
  server: {device: string, app_sha256?: string, git_sha?: string | null, src_sha256?: string | null},
  mma_label: string /* 예: "CoFDA · F7 · CS32 · fused" */,
  preset: string | null /* 정규형 mma 와 형식이 같은 preset id */,
  recipe: {name: string, bundled: string | null, yaml: string},
  env: object,                        // app.runners.common.run_env: capture_env(Hub 조회 없음) + app_sha256,
                                      // 고정 모델 revision 또는 체크포인트 경로·SHA256, 레시피 해시, seed·결정론 설정
  timing: {baseline_s: number,       // 기준 실행 시간. cached_baseline 이면 처음 실행했을 때의 값
           emulated_s: number},     // LLM 은 자유 생성 + 강제 디코딩, 두 번 디코딩한 시간. 한 번 잰 값이라 성능 수치가 아니다
  baseline: Output, emulated: Output, metrics: Metrics, error?: {code: string, message: string},
  evidence: {patched_linear: number, patched_conv2d: number, emulated_calls: number, backend: string}
}
```

Output (작업별)
```ts
// llm.generate
{label: string, text: string,      // text == tokens 의 piece 를 이어 붙인 것
 tokens: [{id: number, piece: string, logprob: number, top: [{piece: string, p: number}] /* 상위 5 */}]}
// vision.detect — 좌표는 원본 이미지 픽셀, image.media 는 media 파일 이름
{label: string, image: {media: string, width: number, height: number},
 boxes: [{cls: string, cls_id: number, conf: number, xyxy: [number, number, number, number]}]}
// vision.classify
{label: string, image: {media: string, width: number, height: number},
 top: [{label: string, class_id: number, p: number}] /* 상위 5 */}
```

Metrics (작업별, 정의는 `app/metrics.py`)
```ts
// llm.generate
{first_divergence: number | null,   // 두 생성 토큰열이 처음 달라지는 위치 (같으면 null)
 prefix_match: number,              // 그 위치까지 같은 토큰 수
 teacher_forced: {                  // baseline 생성열을 따라 위치별로 비교. 생성과 같은 방식(프롬프트 prefill
                                    // 뒤 KV 캐시로 한 토큰씩)이라 위치 t 는 앞의 t 토큰만 본다. baseline 은 자기
                                    // 생성 단계의 logits, 에뮬레이션은 baseline 토큰을 강제한 디코딩의 logits
   positions: number, top1_agreement: number, kl_mean: number, kl_max: number,
   kl: number[],                    // 위치별 KL(baseline ‖ emulated), nats
   top1: boolean[]}}                // 위치별 top-1 일치
// vision.detect — 같은 클래스, IoU ≥ 0.5, 신뢰도 내림차순 greedy 매칭
{matched: number, baseline_only: number, emulated_only: number, mean_iou: number | null,
 mean_abs_conf_delta: number | null, pairs: [{baseline: number, emulated: number, iou: number}]}
// vision.classify
{top1_same: boolean, top5_overlap: number, kl: number, baseline_top1_p: number, emulated_top1_p: number}
```

## 데모 묶음 (`app/web/demo/`)

- `catalog.json` — `GET /api/catalog` 와 같은 형식 (`mode: "demo"`)
- `index.json` — `{runs: [{id, task, model, mma, format, baseline, input: {prompt} | {image}, title, selection,
  role, summary, preview}]}`; `mma` 는 정규형, `selection` 은 그 사례를 고른 기준, `input.image` 는 media 파일 이름,
  `role` 은 이미지를 고른 이유를 짧게 (예: "차이 점수 1위 (21.0)", LLM 은 null),
  `preview` 는 결과 데이터 표용 출력 요약 (llm: baseline·emulated 텍스트 / detect: 상자 수 / classify: top-1),
  `summary` 는 설계 공간 차트용 지표 (llm: first_divergence, prefix_match, top1_agreement, kl_mean /
  detect: matched, baseline_only, emulated_only, mean_iou / classify: top1_same, top5_overlap, kl)
- 데모는 **설계점 스윕**이다: 같은 입력에서 F (fused·decoupled) 를 여러 값으로 바꾼 run 과 preset 점들
- `runs/<id>.json` — `GET /api/runs/{id}` 와 같은 형식, `status: "done"`
- `media/<id>.jpg` — 입력 이미지 (긴 변 ≤ 1024 px), 출처·라이선스는 `media/SOURCES.md`

데모 수치는 기록된 실행이다. 각 run 의 `env` 가 git SHA, 모델 revision, GPU, 라이브러리 버전을 담는다.

## 해석 규칙 (UI 문구에도 그대로 적용)

- 사용자가 만든 알고리즘은 **가상 설계**다. preset 이름(Hopper 등)은 출처가 있는 모델링 산술일 뿐 해당 GPU 측정이
  아니며, native 실리콘 비트 일치는 Hopper WGMMA 에서 140/141 로 실패 상태다 (`support_matrix.yaml` 의 `integration.silicon_probe.wgmma`).
- native 와 에뮬레이션의 차이는 **양자화와 누산을 함께** 바꾼 결과다. 누산만 보려면 기준선을
  `same_quant_fp64` (같은 형식 + FP64 누산)로 둔다.
- 생성 텍스트가 갈라진 뒤의 토큰 비교는 의미가 없으므로, 분포 차이는 강제 디코딩 지표로 본다. 기준(A)이 생성한
  토큰을 에뮬레이션 모델에 생성과 같은 방식(KV 캐시, 한 토큰씩)으로 차례로 넣고 위치마다 다음 토큰 분포를 비교한다.
  전체 시퀀스를 한 번에 넣으면 텐서 단위 동적 양자화의 스케일이 뒤쪽 토큰에 의존해 인과성이 깨지기 때문이다.
  이 지표는 기준이 고른 경로 위의 비교다. 에뮬레이션 모델이 스스로 생성했다면 다른 경로로 갔을 수 있다.
- 한 장·한 문장의 차이는 사례일 뿐 품질 순위가 아니다. 품질 판단은 전체 평가 (`support_matrix.yaml` 의 `model_families.*.pretrained_quality`) 로 한다.
