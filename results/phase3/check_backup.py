import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from dotenv import dotenv_values
from necro.engine import prompt_fingerprint
from necro.experiment_guard import verify

root=Path.cwd()
paths=[Path(x) for x in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0') if x]
paths=sorted({p for p in paths if p.is_file()})
bad=[]
values=dotenv_values('.env')
secrets=[v for k,v in values.items() if v and len(v)>=16 and ('API_KEY' in k or k.endswith('_TOKEN'))]
for path in paths:
    content=path.read_bytes()
    if any(value.encode() in content for value in secrets):
        bad.append(str(path))
if bad:
    raise ValueError('Credential value found in Git scope: '+', '.join(bad))
documents=[Path('README.md'),Path('THIRD_PARTY_NOTICES.md'),*Path('docs').rglob('*.md')]
link_errors=[]
checked=0
for document in documents:
    text=document.read_text(encoding='utf-8')
    text=re.sub(r'```.*?```','',text,flags=re.S)
    for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        link=link.strip('<>')
        if re.match(r'^[a-zA-Z]+:',link): continue
        filename,_,anchor=unquote(link).partition('#')
        target=(document.parent/filename).resolve() if filename else document.resolve()
        checked+=1
        if not target.exists():
            link_errors.append(f'{document}: {link}')
        elif anchor and target.suffix=='.md':
            content=target.read_text(encoding='utf-8')
            anchors=set(re.findall(r'<a\s+(?:id|name)=[\"\']([^\"\']+)',content))
            anchors|={re.sub(r'[^\w -]','',x.lower()).replace(' ','-') for x in re.findall(r'^#+\s+(.+?)\s*$',content,re.M)}
            if anchor not in anchors: link_errors.append(f'{document}: {link}')
if link_errors:
    raise ValueError('Broken documentation links: '+json.dumps(link_errors,ensure_ascii=False))
verify(Path('data/phase3/coverage-v1'),'test',Path('results/phase3/selected/adapter'))
verify(Path('data/phase3/robustness-v1'),'test',Path('results/phase3/selected/adapter'))
expected=json.loads(Path('results/phase3/selected/selection.json').read_text())['prompt_sha256']
assert prompt_fingerprint()==expected
snapshot={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
Path('results/phase3/backup-checked-files.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print(json.dumps({'git_files':len(paths),'local_links_checked':checked,'credential_matches':0,'frozen_phase3_integrity':'passed','prompt_fingerprint':expected}),flush=True)
