import json
import sys
from pathlib import Path

import paramiko

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

request = json.loads(sys.stdin.read())
client = paramiko.SSHClient()
client.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
client.connect('connect.bjb2.seetacloud.com', port=26703, username='root',
               password=request.pop('password'), look_for_keys=False,
               allow_agent=False, timeout=20, auth_timeout=20)
try:
    action = request['action']
    if action == 'exec':
        _, stdout, stderr = client.exec_command(request['command'], timeout=request.get('timeout', 120))
        print(stdout.read().decode('utf-8', errors='replace'))
        print(stderr.read().decode('utf-8', errors='replace'), file=sys.stderr)
        sys.exit(stdout.channel.recv_exit_status())
    with client.open_sftp() as sftp:
        if action == 'put':
            sftp.put(request['local'], request['remote'], confirm=True)
        elif action == 'get':
            sftp.get(request['remote'], request['local'])
        else:
            raise ValueError(action)
    print(json.dumps({'action': action, 'complete': True}))
finally:
    client.close()
