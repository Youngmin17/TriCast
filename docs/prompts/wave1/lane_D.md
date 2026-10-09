<!-- 2026-09-30 실제 사용 원본. 경로·번호는 당시 기준 -->

[Lane D — Qwen/Qwen3-0.6B 시연 데모]
소유 파일 (이것만 생성/수정):
  examples/demo_qwen3.py, examples/README.md, docs/demo/DEMO.md

목표: 발표용 간단 데모. GPU 1장, 수 분 안에 한 명령으로 TriCast 의 가치를 보여준다.
API 는 ENGINE §6 계약만 사용 (동시 구현 중): tricast.load_recipe, tricast.patch_model,
tricast.nn.patch.unpatch_model, tricast.eval.ppl.perplexity, tricast.quantize (→ QTensor.dequantize),
tricast.gemm, tricast.get_scheme, tricast.get_preset, tricast.eval.envinfo.capture_env.
레시피 이름은 Lane I 가 만드는 configs/recipes/ 목록을 쓴다: bf16_passthrough, hopper_fp8_w8a8,
blackwell_fp8_w8a8, fp8_f7_lowacc, fp8_f7_decoupled, mxfp8_w_a, mxfp4_w_a, nvfp4_w_a, nvfp4_4o6,
w4a16_g128_zp_gptq, fp8_ema_static, mxfp4_rht.

examples/demo_qwen3.py (argparse; transformers 는 main 안에서 import 해 --help 가 가볍게 동작):
  --model Qwen/Qwen3-0.6B --device cuda --dtype bfloat16 --quick (PPL 윈도우 수 제한) --max-windows N
  --recipes a,b,c (기본 세트) --skip-gptq --out runs/demo_<UTC>
  1) "형식 한눈에": 모델 한 레이어의 q_proj 가중치를 스킴 (bf16, fp8_tensor, mxfp8_e4m3, mxfp6_e3m2,
     mxfp4, nvfp4, nvfp4_4o6, msfp12, int4_g128_zp) 로 quantize → SQNR(dB) · 최대 절대 오차 · 비트/원소
     (스케일 오버헤드 포함 계산) 표.
  2) "누산 알고리즘이 결과를 바꾼다": 짧은 프롬프트 forward 에서 hook 으로 한 레이어(예: layers[10].mlp.down_proj)
     입력 활성 X 를 캡처 → X 와 가중치를 fp8_tensor 로 양자화 → gemm 을 preset (fp64, nvidia_blackwell_fp8,
     nvidia_hopper_fp8, nvidia_ada_fp8, hopper 기반 f_bits=7 fused, f_bits=7 decoupled) 로 실행 → fp64 대비
     상대 오차(Frobenius)·bf16 출력 비트 불일치 비율 표.
  3) "모델 품질": 레시피별 WikiText-2 PPL (quick: 앞 16 윈도우 × 2048 토큰) + 레시피별 경과 시간 표.
     기본 세트: bf16_passthrough, hopper_fp8_w8a8, fp8_f7_lowacc, fp8_f7_decoupled, mxfp8_w_a, mxfp4_w_a,
     nvfp4_w_a, nvfp4_4o6 (+ --skip-gptq 가 아니면 w4a16_g128_zp_gptq). 레시피 전환은 unpatch → patch
     (가중치 원본 보존 확인).
  4) "생성 비교": 같은 프롬프트 2개 (한국어 1, 영어 1; tokenizer.apply_chat_template(...,
     enable_thinking=False), greedy, max_new_tokens 48) 를 bf16 vs hopper fp8 vs fp8 F=7 vs mxfp4 로 생성해
     나란히 출력.
  5) 결과를 runs/demo_<UTC>/results.json 과 report.md (표 + env 요약) 로 저장.
  진행 로그는 단계별 1줄. 예외가 나면 그 단계만 실패로 기록하고 다음 단계로 진행.
examples/README.md: 설치/실행법 (클러스터 env 예: /scratch/uceeeee/conda_envs/tricast/bin/python), 단계별
  예상 소요 시간 (추정치라고 표시), 출력 파일 설명.
docs/demo/DEMO.md: 발표용 3분 시나리오 (각 단계에서 무엇을 보여주고 어떤 메시지를 전달하는지) + 표 읽는 법.
  수치는 아직 없다 — "실측 후 채움" 으로 표시하고 절대 지어내지 말 것.

완료 기준: python -m py_compile examples/demo_qwen3.py, ruff 0, `.venv/bin/python examples/demo_qwen3.py --help`
동작. (실제 실행은 Claude 가 GPU 에서.)
