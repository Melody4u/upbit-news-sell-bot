const fs = require('fs');

const journalPath = process.argv[2];
const statePath = process.argv[3];

let lastSent = 0;
try {
  lastSent = JSON.parse(fs.readFileSync(statePath, 'utf8')).last_sent_ts || 0;
} catch {}

let text;
try {
  text = fs.readFileSync(journalPath, 'utf8');
} catch (e) {
  // 정상 케이스: 아직 거래가 없어서 journal 파일이 없을 수 있음(초기 구간)
  if (e && (e.code === 'ENOENT' || String(e).includes('ENOENT'))) {
    console.log(JSON.stringify({ lastSent, maxTs: lastSent, eventsCount: 0, message: '' }));
    process.exit(0);
  }
  console.log(JSON.stringify({ error: 'journal_read_failed', detail: String(e), lastSent }));
  process.exit(0);
}

const lines = text.trim().split(/\r?\n/);

const events = [];
for (let i = lines.length - 1; i >= 0; i--) {
  const line = (lines[i] || '').trim();
  if (!line) continue;
  let o;
  try {
    o = JSON.parse(line);
  } catch {
    continue;
  }

  const ts = Number(o.ts || 0);
  if (ts && ts <= lastSent) break; // stop scanning older entries

  const side = o.side;
  const stop = !!o.stop_event;

  const reasons = Array.isArray(o.reasons)
    ? o.reasons
    : (Array.isArray(o.reason) ? o.reason : (typeof o.reason === 'string' ? [o.reason] : []));

  const hasHard = reasons.some(r => typeof r === 'string' && r.startsWith('hard_stop'));

  if (side === 'sell' && stop && hasHard) {
    events.push({
      ts,
      market: o.market || o.symbol || o.code || '',
      price: o.price ?? o.avg_price ?? o.fill_price ?? o.executed_price ?? o.last_price ?? null,
      reasons,
    });
  }
}

// oldest -> newest
events.sort((a, b) => (a.ts || 0) - (b.ts || 0));

const maxTs = events.reduce((m, e) => Math.max(m, e.ts || 0), lastSent);

let message = '';
if (events.length) {
  const parts = [];
  parts.push('🚨 하드 스탑(강제 손절) 이벤트 감지');
  parts.push('');

  for (const e of events) {
    const dt = new Date(e.ts).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul' });
    parts.push('- 시간: ' + dt);
    parts.push('  마켓: ' + (e.market || '(unknown)'));
    if (e.price != null) parts.push('  가격: ' + e.price);
    parts.push('  사유: ' + (e.reasons || []).join(', '));
    parts.push('');
  }

  parts.push('복구 규칙: 24시간 이내에 "분봉 컨센서스 + score >= scout" 조건이 만족되면 매매 재개');
  message = parts.join('\n');
}

console.log(JSON.stringify({ lastSent, maxTs, eventsCount: events.length, message }));
