
// spine-body.js -- the body, in the real game.
//
// It owns no policy, no physics and no world model of its own:
//   * it presses keys when Python tells it to,
//   * it reports the state the SERVER gives it, once per physics tick,
//   * it answers block-column queries out of the chunk cache the server sent it.
// Every judgement is made in Python, by the gate in spine/embodied.py, against
// world data that came from the game and not from a file this repo wrote.
//
// usage: node spine-body.js <obs.jsonl> <tcp_port> <max_ticks>
const mineflayer = require('mineflayer');
const { Vec3 } = require('vec3');
const fs = require('fs');
const net = require('net');

const outPath = process.argv[2];
const port = parseInt(process.argv[3], 10);
const maxTicks = parseInt(process.argv[4], 10);
const tpTo = process.argv[5] ? process.argv[5].split(',').map(Number) : null;

const bot = mineflayer.createBot({
  host: '127.0.0.1', port: 25565, username: 'spinebot',
  version: '1.21.1', auth: 'offline', physicsEnabled: true
});
const fd = fs.openSync(outPath, 'w');
let sock = null, buf = '', tick = 0, lastAct = null, stopped = false, ready = false;

const AIR = new Set(['air', 'cave_air', 'void_air']);

function nameAt(x, y, z) {
  // blockAt wants a Vec3; a plain object falls through to pos.floored() and throws.
  const b = bot.blockAt(new Vec3(Math.floor(x), Math.floor(y), Math.floor(z)));
  return b ? b.name : null;          // null means "chunk not loaded": say so, do not guess
}

const KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak'];
function apply(act) {
  for (const k of KEYS) { try { bot.setControlState(k, !!(act.keys && act.keys[k])); } catch (e) {} }
  if (act.yaw !== undefined && act.yaw !== null) { try { bot.look(act.yaw, 0, true); } catch (e) {} }
  if (act.chat) { try { bot.chat(act.chat); } catch (e) {} }
  lastAct = act;
}

function state() {
  const e = bot.entity; if (!e) return null;
  const p = e.position, v = e.velocity;
  return {
    tick,
    pos: [+p.x.toFixed(5), +p.y.toFixed(5), +p.z.toFixed(5)],
    vel: [+v.x.toFixed(5), +v.y.toFixed(5), +v.z.toFixed(5)],
    yaw: +e.yaw.toFixed(5), on_ground: e.onGround, health: bot.health, food: bot.food,
    below: nameAt(p.x, p.y - 0.5, p.z),
    feet: nameAt(p.x, p.y + 0.01, p.z),
    head: nameAt(p.x, p.y + 1.5, p.z)
  };
}

// One column of the REAL world, out of the chunk cache. Cells are only the
// non-air ones: the gate treats absent as air anyway, and it keeps the line short.
function column(q) {
  const cells = [];
  let unloaded = 0, seen = 0;
  for (let y = Math.floor(q.y0); y >= Math.floor(q.y1); y--) {
    const n = nameAt(q.x, y, q.z);
    if (n === null) { unloaded++; continue; }
    seen++;
    if (!AIR.has(n)) cells.push([y, n]);
  }
  return { id: q.id, x: q.x, z: q.z, cells, unloaded, seen };
}

function handle(obj) {
  if (obj.q) {
    let ans;
    try { ans = column(obj.q); } catch (e) { ans = { id: obj.q.id, x: obj.q.x, z: obj.q.z, cells: [], error: String(e) }; }
    try { sock.write('C' + JSON.stringify(ans) + '\n'); } catch (e) {}
  } else if (obj.act) {
    apply(obj.act);
  }
  if (obj.quit) stopped = true;
}

const sockReady = new Promise(res => {
  const go = () => {
    const s = net.connect(port, '127.0.0.1', () => { sock = s; console.log('SOCKOK'); res(); });
    s.on('error', () => setTimeout(go, 200));
    s.on('data', d => {
      buf += d.toString('utf8');
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        let m; try { m = JSON.parse(line); } catch (e) { continue; }
        handle(m);
      }
    });
  };
  go();
});

process.on('uncaughtException', e => console.log('UNCAUGHT', e && e.message));

bot.once('spawn', async () => {
  console.log('SPAWN', bot.entity.position.toString());
  if (tpTo) {                       // the bench places the body; the body does not choose
    bot.chat('/tp ' + tpTo.join(' '));
    await new Promise(r => setTimeout(r, 900));
    console.log('PLACED', bot.entity.position.toString());
  }
  await new Promise(r => setTimeout(r, 2500));   // let the chunk cache fill
  await sockReady;
  // one physics tick of settling so blockAt is answering before the gate asks
  await new Promise(r => setTimeout(r, 400));
  console.log('READY', bot.entity.position.toString(), 'below', nameAt(bot.entity.position.x, bot.entity.position.y - 0.5, bot.entity.position.z));
  ready = true;
  fs.writeSync(fd, JSON.stringify({ event: 'spawn', pos: state().pos, hp: bot.health }) + '\n');
  bot.on('physicsTick', () => {
    tick++;
    const s = state(); if (!s) return;
    if (sock && sock.writable) { try { sock.write('S' + JSON.stringify(s) + '\n'); } catch (e) {} }
    try { fs.writeSync(fd, JSON.stringify({ s, act: lastAct }) + '\n'); } catch (e) {}
    if (tick >= maxTicks || stopped) {
      console.log('STOP', tick);
      try { fs.closeSync(fd); } catch (e) {}
      try { sock.destroy(); } catch (e) {}
      bot.quit();
    }
  });
});

bot.on('death', () => {
  // the game's own verdict on the episode; recorded, not interpreted
  console.log('DEATH at', bot.entity.position.toString(), 'health', bot.health);
  try { fs.writeSync(fd, JSON.stringify({ event: 'death', pos: state().pos, tick }) + '\n'); } catch (e) {}
});
bot.on('kicked', r => console.log('KICKED', JSON.stringify(r)));
bot.on('error', e => console.log('ERR', e && e.message));
bot.on('end', () => { try { fs.closeSync(fd); } catch (e) {} process.exit(0); });
