
const mineflayer = require('mineflayer');
const bot = mineflayer.createBot({host:'127.0.0.1',port:25565,username:'builder',
  version:'1.21.1',auth:'offline'});
bot.once('spawn', () => { console.log('READY', bot.entity.position); });
setTimeout(() => { bot.quit(); process.exit(0); }, parseInt(process.argv[2]||'15000',10));
