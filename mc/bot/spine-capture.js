
// spine-capture.js -- drive the real game and record what the SERVER says happens.
// Every row is one physics tick of the real client, so a predictor can be checked
// against it tick by tick instead of at the ends.
const mineflayer = require('mineflayer');
const fs = require('fs');
const outPath = process.argv[2] || 'capture.jsonl';
const cfg = JSON.parse(process.argv[3] || '{"ticks":0,"plan":[]}');
const plan = cfg.plan;
const bot = mineflayer.createBot({ host:'127.0.0.1', port:25565, username:'spinebot',
  version:'1.21.1', auth:'offline', physicsEnabled:true });
const fd = fs.openSync(outPath, 'w');
let tick = 0;
const planAt = {};
for (const p of plan) { (planAt[p.tick] = planAt[p.tick] || []).push(p); }
function row(tag) {
  const e = bot.entity; if (!e) return;
  const p = e.position, v = e.velocity;
  const b = bot.blockAt(p.offset(0, -0.5, 0));
  fs.writeSync(fd, JSON.stringify({
    tick, tag, pos:[+p.x.toFixed(5), +p.y.toFixed(5), +p.z.toFixed(5)],
    vel:[+v.x.toFixed(5), +v.y.toFixed(5), +v.z.toFixed(5)],
    yaw:+e.yaw.toFixed(5), on_ground: e.onGround,
    health: bot.health, food: bot.food,
    below: b ? b.name : 'void',
    keys: bot.controlState ? Object.fromEntries(Object.entries(bot.controlState).filter(([k,v])=>v)) : {}
  }) + '\n');
}
bot.once('spawn', () => {
  row('spawn');
  bot.physicsTick = 0;
  bot.on('physicsTick', () => {
    tick++;
    const acts = planAt[tick] || [];
    for (const a of acts) {
      for (const [k, v] of Object.entries(a.keys || {})) {
        try { bot.setControlState(k, v); } catch (e) {}
      }
      if (a.yaw !== undefined) bot.look(a.yaw, 0, true);
      if (a.chat) bot.chat(a.chat);
    }
    row('tick');
    if (tick >= (cfg.ticks || 120)) {
      fs.closeSync(fd); console.log('captured ticks', tick); bot.quit();
    }
  });
});
bot.on('error', e => console.log('ERR', e.message));
bot.on('kicked', r => console.log('KICKED', JSON.stringify(r)));
bot.on('end', () => { try { fs.closeSync(fd); } catch (e) {} console.log('end'); });
