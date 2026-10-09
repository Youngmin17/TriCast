# CLAUDE.md — TriCast

> Claude Code 가 이 저장소에서 세션을 시작할 때 자동으로 읽는 파일이다. 규칙·용어집·운영 정보는 전부
> `AGENTS.md` 에 있고, 이 파일은 그것을 끌어올리기만 한다. 여기에 규칙을 복제하지 않는다.

@AGENTS.md

## Claude Code 에서만 해당하는 것
- 로드 확인: 새 세션에서 `/context` — 목록에 `CLAUDE.md` 와 `AGENTS.md` 가 함께 있어야 정상. `docs/SPEC.md`,
  `docs/ontology.yaml` 은 요청 시 읽는 파일이라 목록에 없는 것이 정상.
- 슬래시 명령: `/check` (린트 + 테스트), `/golden` (골든 케이스 진단 — 수정 없음). 정의는 `.claude/commands/`.
- 훅 (`.claude/settings.json` → `.claude/hooks/guard.py`): Edit/Write 로 AGENTS.md §4 의 보호 파일이나 게이트
  파일(Makefile, pyproject.toml, CI 워크플로, `.claude/`) 을 고치기 전에 사람에게 묻고, 편집한 `.py` 에 `ruff check`
  를 돌려 경고를 바로 돌려준다. Bash 로 바꾸는 경우는 대상이 아니다. `claude` 실행 전 `TRICAST_HOOKS=off` 로 끈다.
