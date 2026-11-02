import json,copy,math
from pathlib import Path
import pyte
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[2];out=root/'docs/demo'
lines=(out/'session.cast').read_text().splitlines();header=json.loads(lines[0]);events=[json.loads(l) for l in lines[1:]]
screen=pyte.Screen(header['width'],header['height']);stream=pyte.Stream(screen)
font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',19)
titlefont=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',47)
subfont=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
W,H=1600,1000
palette={'default':'#dce8f7','black':'#111827','red':'#ff8794','green':'#74e0ae','brown':'#e5c478','blue':'#7faeff','magenta':'#d9a6ff','cyan':'#6ce5cf','white':'#eaf2ff'}
def render():
 im=Image.new('RGB',(W,H),'#0b1424');d=ImageDraw.Draw(im)
 d.text((70,30),'Odyssey / terminal AI coding agent',font=titlefont,fill='#eef5ff')
 d.text((70,94),'Real local-model session • create a Python function → run an assertion → verify output',font=subfont,fill='#b5c8e1')
 d.rounded_rectangle((60,157,1540,920),18,fill='#0e192a',outline='#2a405b',width=2)
 for x,c in [(87,'#ff6a6a'),(114,'#ffd06b'),(141,'#75d6a2')]:d.ellipse((x,176,x+14,190),fill=c)
 d.text((190,170),'odyssey  /  qwen3.8:27b-mlx  /  disposable workspace',font=subfont,fill='#9db1cc')
 for y in range(screen.lines):
  for x in range(screen.columns):
   ch=screen.buffer[y][x]
   if not ch.data.strip():continue
   c=palette.get(ch.fg,'#dce8f7')
   if len(ch.fg)==6 and ch.fg not in palette:c='#'+ch.fg
   d.text((82+x*12.8,222+y*21),ch.data,font=font,fill=c)
 d.text((70,947),'Recorded CLI output • idle gaps shortened • tool trace and original timing included in the repo',font=subfont,fill='#8ca3be')
 return im
frames=[];durations=[];prev=0
for t,kind,data in events:
 if t>33:break
 if kind!='o':continue
 dt=min(max(t-prev,.04),2.2);prev=t
 if dt>.15 and frames:durations[-1]+=round(dt*1000)
 stream.feed(data)
 # Capture stable chunks rather than each cursor movement.
 if '\r\n' in data and '\x1b[6n' not in data:
  frames.append(render());durations.append(180)
# Final complete state before session-save path is displayed is the thumbnail.
# Render the final model response from genuine terminal output, omitting later administrative commands.
s=pyte.Screen(header['width'],header['height']);ss=pyte.Stream(s)
for t,k,data in events:
 if t>32.733 or ('\x1b[6n' in data and t>20):break
 ss.feed(data)
screen=s
poster=render();poster.save(out/'preview.png');poster.save(out/'preview.png')
frames.append(poster);durations.append(4000)
# Avoid excessively long contact sheets/large gifs, keep visual states and timing.
frames[0].save(out/'odyssey-demo.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True)
(out/'render-manifest.json').write_text(json.dumps({'frames':len(frames),'duration_ms':sum(durations),'editing':'ANSI replay; idle gaps capped at 2.2s; viewing holds added. Original timing in session.cast.','model':'qwen3.8:27b-mlx'},indent=2)+'\n')
print('Rendered',len(frames),'frames;',sum(durations),'ms')
