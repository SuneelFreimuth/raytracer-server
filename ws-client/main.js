const WebSocket = require('ws');
const readline = require('readline/promises');
const { exit } = require('process');

function main() {
  let connected;

  const hostname = process.argv[2];

  const ws = new WebSocket(`ws://${hostname}`)

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  ws.on('error', console.error);

  ws.on('open', async () => {
    connected = true;
    while (connected) {
      const msg = await rl.question('> ');
      if (msg.length > 0)
        await ws.send(msg);
    }
  });

  ws.on('message', (msg, isBinary) => {
    const now = new Date().toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
    
    if (isBinary) {
      console.log(`[${now}] ${formatBytes(bytes)}`);
    } else
      console.log(`[${now}] ${msg}`);
  });

  ws.on('close', () => {
    console.log('Disconnected.');
    exit(0);
  });
}

const formatBytes = bs => {
  if (bs.length === 0)
    return '{}';

  let str = '{' + formatByte(bs[0]);
  for (let i = 1; i < bs.length; i++)
    str += ' ' + formatByte(bs[i]);
  str += '}';
  return str;
}

const formatByte = b => (
  ((b >> 4) & 0xF).toString(16) +
  (b & 0xF).toString(16)
);


main();