import { spawn } from 'child_process';
import { setTimeout } from 'timers/promises';

// Start server
const server = spawn('node', ['node_modules/ts-node/dist/bin.js', 'src/server/index.ts'], {
  stdio: ['ignore', 'pipe', 'pipe'],
  shell: true
});

let serverOutput = '';
server.stdout.on('data', (data) => {
  serverOutput += data.toString();
  console.log('[SERVER]', data.toString().trim());
});
server.stderr.on('data', (data) => {
  serverOutput += data.toString();
  console.error('[SERVER ERR]', data.toString().trim());
});

await setTimeout(3000);

// Make request
console.log('Making request...');
try {
  const response = await fetch('http://localhost:3002/mcp', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream'
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'test', version: '1.0.0' }
      }
    })
  });
  
  console.log('Response status:', response.status);
  console.log('Response headers:', Object.fromEntries(response.headers.entries()));
  const text = await response.text();
  console.log('Response body:', text || '(empty)');
} catch (e) {
  console.error('Fetch error:', e);
}

server.kill();
console.log('\nFull server output:\n', serverOutput);
