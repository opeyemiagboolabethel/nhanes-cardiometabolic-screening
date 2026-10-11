"""Download official public CDC files and record checksums."""
from pathlib import Path
import json,hashlib,urllib.request
root=Path(__file__).parent/'data';raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
manifest=[]
for role,year,prefix in [('development',2015,'I'),('temporal_test',2017,'P')]:
    for component in ['DEMO','BMX','GHB','DIQ']:
        name=component+'_I' if prefix=='I' else 'P_'+component
        url=f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt'
        b=urllib.request.urlopen(url,timeout=60).read()
        if not b.startswith(b'HEADER RECORD'):raise ValueError('Expected SAS transport file: '+name)
        (raw/(name+'.xpt')).write_bytes(b)
        manifest.append(dict(role=role,file=name+'.xpt',url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
