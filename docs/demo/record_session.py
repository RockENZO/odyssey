import os,pty,select,time,json,fcntl,termios,struct
from pathlib import Path
root=Path(__file__).resolve().parents[2]; out=root/'docs/demo'; out.mkdir(exist_ok=True,parents=True)
for name in ('demo-workspace','demo-config','demo-data'):(root/name).mkdir(exist_ok=True)
(root/'demo-config/config.toml').write_text('ollama_host = "http://localhost:11434"\nfast_model = "qwen3.8:27b-mlx"\ndeep_model = "qwen3.8:27b-mlx"\nembedding_model = "qwen3.8:27b-mlx"\n')
pid,fd=pty.fork()
if pid==0:
 os.chdir(root/'demo-workspace')
 os.environ.update(TERM='xterm-256color',ODYSSEY_CONFIG_DIR=str(root/'demo-config'),ODYSSEY_DATA_DIR=str(root/'demo-data'),PYTHONWARNINGS='ignore')
 os.execv(str(root/'.venv/bin/odyssey'),['odyssey'])
fcntl.ioctl(fd,termios.TIOCSWINSZ,struct.pack('HHHH',32,110,0,0))
start=time.monotonic(); sent=False; exited=False; last=time.monotonic(); transcript='';events=[]
with (out/'session.cast').open('w') as f:
 f.write(json.dumps({'version':2,'width':110,'height':32,'title':'Odyssey — local tool-use demo','env':{'TERM':'xterm-256color'}})+'\n');f.flush()
 while time.monotonic()-start<240:
  if select.select([fd],[],[],.2)[0]:
   try: b=os.read(fd,65536)
   except OSError:break
   if not b:break
   s=b.decode('utf-8','replace');transcript+=s;last=time.monotonic()
   f.write(json.dumps([round(last-start,4),'o',s])+'\n');f.flush()
   if '\x1b[6n' in s:os.write(fd,b'\x1b[1;1R')
  if not sent and 'Type /help' in transcript and time.monotonic()-last>.8:
   os.write(fd,b"Use write to create greet.py with greet(name) returning 'Hello, ' + name + '!'. Then use bash to run python3 -c \"from greet import greet; assert greet('world') == 'Hello, world!'; print('PASS: greeting verified')\". Do not use sub-agents or memory. Summarize briefly.\r")
   sent=True;transcript='';print('Demo prompt submitted',flush=True)
  if sent and '→ bash' in transcript and time.monotonic()-last>4 and not exited:
   os.write(fd,b'!python3 -c "from greet import greet; assert greet(\'world\') == \'Hello, world!\'; print(\'PASS: greeting verified\')"\r/save\r/exit\r');exited=True
  if exited and 'Odyssey session ended' in transcript:break
 try:os.kill(pid,15)
 except ProcessLookupError:pass
os.waitpid(pid,0)
(out/'capture-summary.json').write_text(json.dumps({'model':'qwen3.8:27b-mlx','elapsed_seconds':round(time.monotonic()-start,2),'real_cli':True,'prompt_submitted':sent,'graceful_exit':exited,'source_commit':os.popen('git -C '+str(root)+' rev-parse HEAD').read().strip()},indent=2)+'\n')
print('Recording complete',flush=True)
