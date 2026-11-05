// 중간평가 2 도입 슬라이드 3장 (docs/DEMO.md §1–3): 대상 사용자 · 문제 / 입력 → 결과 / 최종 목표.
// 수치는 deck_data.json (make_data.py 가 번들 · runs/ 기록에서 생성) 에서 읽는다. 손으로 옮겨 적지 않는다.
//   python3 docs/slides/midterm2/make_data.py
//   NODE_PATH=~/.claude/skills/paper-slides/node_modules node docs/slides/midterm2/build_deck.js <out.pptx>
const fs = require('fs');
const path = require('path');
const JSZip = require('jszip');
const { Deck, BODY } = require(path.join(process.env.HOME, '.claude/skills/paper-slides/assets/academic_theme.js'));

const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'deck_data.json'), 'utf8'));
const deck = new Deck({ title: 'TriCast — 층별 양자화 Auto-tune', shortTitle: 'TriCast', sections: ['문제', '시나리오', '목표'] });
const F = deck.font;
const t = (extra) => ({ fontFace: F, color: '111111', margin: 0, ...extra });

// ---------- 1. 대상 사용자와 문제 ----------
let s = deck.slide({ title: '누가, 어디서 막히나', section: '문제', notes: [
  '출처: docs/PROBLEM.md §1 (한 문장 문제), §5 (성공의 정의)',
  'docs/research/interviews.md: 로그 07 (30개 이상 중 6개, 실험당 4일), 로그 10 (200곳 전수 탐색 3개월),',
  '로그 16 ("누구 책임인지 가릴 기준이 없다"), 기간 통계 (원인을 못 찾고 오래 튜닝한 5명, 3주~3개월), B-6 사분면 ① 12명',
].join('\n') });
s.addText([
  { text: '대상  ', options: { bold: true, color: 'A31F34' } },
  { text: '칩에 올릴 모델의 층별 양자화 방법 · 정밀도를 정하는 엔지니어 · 연구자 (인터뷰 사분면 ① 12명)' },
], t({ x: BODY.x + 0.2, y: 1.3, w: BODY.w - 0.4, h: 0.45, fontSize: 20 }));
const pains = [
  ['Challenge 1:', '조합마다 코드 재작성', '4일', '실험 1개 = 커널 수정 + 재검증 (로그 07)', '→ 원하던 조합 30개 이상 중 6개만 평가'],
  ['Challenge 2:', '층별 배정을 손으로 전수 탐색', '3주–3개월', '원인을 못 찾고 오래 튜닝한 5명의 기간', '→ 196층 × 16방법은 사람이 못 고른다'],
  ['Challenge 3:', '결과에 실행 환경이 없다', '“누구 책임인지\n가릴 기준이 없다”', '공유된 점수에 SHA · 레시피가 없음 (로그 16)', '→ 같은 조건 비교 · 재현이 안 된다'],
];
pains.forEach(([head, key, big, cap, verdict], i) => {
  const x = BODY.x + 0.2 + i * 4.1, w = 3.85;
  s.addText(head, t({ x, y: 1.98, w, h: 0.38, fontSize: 18, bold: true, align: 'center' }));
  s.addText(key, t({ x, y: 2.36, w, h: 0.42, fontSize: 20, bold: true, align: 'center' }));
  deck.block(s, '', { x: x + 0.15, y: 2.92, w: w - 0.3, h: 1.62, tone: 'pink' });
  s.addText(big, t({ x: x + 0.15, y: 2.98, w: w - 0.3, h: 0.98, fontSize: i === 2 ? 20 : 34, bold: true, align: 'center', valign: 'middle' }));
  s.addText(cap, t({ x: x + 0.25, y: 3.96, w: w - 0.5, h: 0.5, fontSize: 13, color: '404040', align: 'center', valign: 'middle' }));
  s.addShape(deck.pres.shapes.DOWN_ARROW, { x: x + w / 2 - 0.22, y: 4.66, w: 0.44, h: 0.4, fill: { color: 'A040A0' }, line: { color: 'A040A0' } });
  deck.verdict(s, verdict, { x, y: 5.16, w, h: 0.62 });
});
deck.message(s, 'TriCast: 크기 예산 하나로 층별 배정을 고르고, 모든 측정값에 [[GPU · 트리 SHA · 레시피 해시]]를 붙인다',
  { y: 6.08, h: 0.45 }, { bullet: false, size: 20, align: 'center' });

// ---------- 2. 입력 → 결과 ----------
s = deck.slide({ title: '문장 한 줄에서 레시피 한 장까지', section: '시나리오', notes: [
  '출처: docs/DEMO.md §2, docs/ARCHITECTURE.md. 표: runs/20261104_lowbit/qz_trackm_qwen/table.json (196층 × 16방법).',
  `세션 수치: ${data.sources.bundle} (session.best, solutions iso_3.15), seed PPL: ${data.sources.seeds}`,
].join('\n') });
deck.message(s, '휴리스틱(배낭)이 표 위에서 고르고, AI 가 표에 없는 옵션을 제안하고, [[GPU 측정]]이 판정한다',
  { y: 1.3, h: 0.45 }, { bullet: false, size: 20 });
const top = 2.05, bot = 3.75, bh = 1.2;
// ① 요청
deck.block(s, '요청 문장\n“403/128 비트 안에서,\n마지막 블록은 BF16 으로”', { x: 0.6, y: top, w: 2.65, h: bh, tone: 'blue', size: 13 });
deck.block(s, 'Claude 해석 + 코드 검사\n표에 없는 방법 · 모델은 거부', { x: 0.6, y: bot, w: 2.65, h: bh, tone: 'dark', size: 13 });
deck.arrow(s, 1.92, top + bh, 1.92, bot);
// ② 표 + 배낭
deck.block(s, '층별 KL 표\n196층 × 16방법\n(층 하나만 양자화)', { x: 3.6, y: top, w: 2.5, h: bh, tone: 'blue', size: 13 });
deck.block(s, '정확 배낭 DP (휴리스틱)\n예산 안 KL 합 최소', { x: 3.6, y: bot, w: 2.5, h: bh, tone: 'teal', size: 13 });
deck.arrow(s, 4.85, top + bh, 4.85, bot);
deck.arrow(s, 3.25, bot + bh / 2, 3.6, bot + bh / 2);
// ③ AI 라운드 (제안 → 검사 → 측정 → 원장 → 제안)
const lx = 6.6, cw = 1.75, gap = 0.3;
deck.block(s, 'Claude 제안\n레이어 카드 RAG', { x: lx, y: top, w: cw, h: bh, tone: 'gold', size: 13 });
deck.block(s, '결정론 검사\n예산 · 인용 id · 숫자', { x: lx + cw + gap, y: top, w: cw, h: bh, tone: 'teal', size: 13 });
deck.block(s, 'GPU 측정\n검증 KL', { x: lx + cw + gap, y: bot, w: cw, h: bh, tone: 'green', size: 13 });
deck.block(s, '원장 (ledger)\nGPU · SHA · 해시', { x: lx, y: bot, w: cw, h: bh, tone: 'blue', size: 13 });
deck.arrow(s, lx + cw, top + bh / 2, lx + cw + gap, top + bh / 2);
deck.arrow(s, lx + cw + gap + cw / 2, top + bh, lx + cw + gap + cw / 2, bot);
deck.arrow(s, lx + cw + gap, bot + bh / 2, lx + cw, bot + bh / 2);
deck.arrow(s, lx + cw / 2, bot, lx + cw / 2, top + bh);
deck.arrow(s, 6.1, bot + bh / 2, lx, bot + bh / 2);
s.addText('시작\n배정', t({ x: 6.1, y: bot + bh / 2 + 0.06, w: 0.5, h: 0.5, fontSize: 12, color: '404040', align: 'center' }));
s.addText('거부 사유 · 측정값을 다음 라운드에', t({ x: lx - 0.1, y: bot + bh + 0.08, w: 2 * cw + gap + 0.2, h: 0.3, fontSize: 12, color: '404040', align: 'center' }));
// ④ 결과
const rx = lx + 2 * cw + gap + 0.45;
deck.block(s, '레시피 YAML 한 장\n\n검증 KL · test PPL\n+ GPU · 트리 SHA\n+ 레시피 해시', { x: rx, y: top, w: 13.333 - rx - 0.45, h: bot + bh - top, tone: 'gray', size: 14 });
deck.arrow(s, lx + 2 * cw + gap, top + bh / 2, rx, top + bh / 2);
[[1, 0.6], [2, 3.6], [3, lx], [4, rx]].forEach(([n, x]) => deck.marker(s, n, x - 0.18, top - 0.2));
const sess = data.session;
deck.message(s, `데모 세션 (${data.model}, ${sess.bits_label} 비트): 검증 KL ${sess.start_kl} → ${sess.best_kl} (${sess.kl_change}), ` +
  `고르는 데 쓰지 않은 test PPL [[${sess.ppl_pct}]] (보정 seed ${sess.wins} 에서 낮음, p = ${sess.p}), LLM $${sess.llm_cost}`,
  { y: 5.35, h: 0.75 }, { bullet: false, size: 17 });
if (data.transfer_line) {
  deck.message(s, `같은 옵션을 다른 예산의 휴리스틱 해에 칸은 그대로 두고 적용: ${data.transfer_line} (3.5비트 이상은 잡음 수준)`,
    { y: 6.12, h: 0.7 }, { bullet: false, size: 17 });
}

// ---------- 3. 최종 목표 ----------
s = deck.slide({ title: '최종 목표와 지금', section: '목표', notes: [
  '출처: docs/DEMO.md §3 (수정안 2026-11-05, 팀 확정 필요), docs/evidence/2026-11-04_lowbit_track_m.md §2.6–2.8,',
  `${data.sources.bundle}, ${data.sources.seeds}. 판정은 고르는 데 쓰지 않은 WikiText-2 test PPL (seqlen 2048).`,
].join('\n') });
const head = ['구분', '지표', '목표 (최종 평가까지)', '지금 (Qwen3-0.6B)'];
const hcell = (x) => ({ text: x, options: { bold: true, color: 'FFFFFF', fill: { color: 'A31F34' } } });
const rows = [head.map(hcell)].concat(data.goals.map((g) => [g.kind, g.metric, g.target, g.now].map((x, j) => ({
  text: x, options: j === 3 && g.key ? { color: 'D0021B', bold: true } : {},
}))));
s.addTable(rows, {
  x: BODY.x + 0.1, y: 1.35, w: BODY.w - 0.2, colW: [0.8, 2.6, 4.45, 4.4],
  fontFace: F, fontSize: 13, color: '111111', valign: 'middle', margin: [3, 6, 3, 6],
  border: { type: 'solid', pt: 0.75, color: 'BFBFBF' }, rowH: 0.5, autoPage: false,
});
s.addText('목표 수치는 수정안 — 팀 확정 필요. 측정값은 모두 A100 기록값 (GPU · 트리 SHA · 레시피 해시 동반).',
  t({ x: BODY.x + 0.1, y: 6.5, w: BODY.w - 0.2, h: 0.3, fontSize: 12, color: '595959' }));

// PowerPoint for Mac 은 lang 이 en-US 인 한글 run 을 단어 중간에서 줄바꿈한다 → 모든 run 을 ko-KR 로
const out = process.argv[2] || 'tricast_midterm2_intro.pptx';
deck.pres.write({ outputType: 'nodebuffer' }).then(async (buf) => {
  const zip = await JSZip.loadAsync(buf);
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/(slides|notesSlides)\/.*\.xml$/.test(n))) {
    const xml = await zip.file(name).async('string');
    zip.file(name, xml.replace(/lang="en-US"/g, 'lang="ko-KR"'));
  }
  fs.writeFileSync(out, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
  console.log('DECK_WRITTEN', out);
});
