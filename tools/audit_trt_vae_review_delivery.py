"""Actual loopback Range delivery plus saved workflow schema checks; no browser."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT/'artifacts/acceleration-research-20260909'
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file,write_new_json  # noqa: E402
from tools.audit_progressive_workflows import audit_candidate  # noqa: E402
from tools.build_trt_vae_workflows import recipe  # noqa: E402


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos,self.sections,self.canvases = [],[],0
    def handle_starttag(self,tag,attrs):
        attrs = dict(attrs)
        if tag == 'video':
            if 'autoplay' in attrs or 'controls' not in attrs:
                raise ValueError('Review must be manually playable')
            self.videos.append(attrs['src'])
        if tag == 'section':
            self.sections.append(attrs['id'])
        if tag == 'canvas':
            self.canvases += 1


def audit(root,port):
    root = Path(root).resolve(strict=True)
    receipt = json.loads((root/'private-receipt.json').read_text(encoding='utf8'))
    public = root/'public'
    augmented = receipt['status'] == 'eight_short_pairs_and_static_text_human_pending'
    if receipt['status'] not in ('seven_short_pairs_built_human_pending','eight_short_pairs_and_static_text_human_pending') or digest_file(public/'review.html') != receipt['review_html_sha256']:
        raise ValueError('Review receipt changed')
    url = f'http://127.0.0.1:{port}'
    with urllib.request.urlopen(url+'/review.html',timeout=10) as response:
        body = response.read()
        if response.status != 200 or body != (public/'review.html').read_bytes():
            raise ValueError('Live review HTML differs')
    parser = Page()
    parser.feed(body.decode('utf8'))
    count = 8 if augmented else 7
    expected = [f'pair-{i}-{side}.mp4' for i in range(1,count+1) for side in ('A','B')]
    sections = [f'pair-{i}' for i in range(1,count+1)] + (['text-check'] if augmented else [])
    if parser.videos != expected or parser.sections != sections or parser.canvases != 2*count:
        raise ValueError('Not all14 videos and native-pixel detail panels are inline')
    if augmented:
        from tools.extend_trt_short_review import text_section
        if text_section(Path(receipt['text_root'])) not in body.decode('utf8'):
            raise ValueError('Actual inline text chart bytes or controls changed')
    checks = []
    for item in receipt['sections']:
        for side,identity in item['mapping'].items():
            name = item['id']+'-'+side+'.mp4'
            path = public/name
            if digest_file(path) != identity['sha256'] or digest_file(identity['path']) != identity['sha256']:
                raise ValueError('Original/public media changed')
            size = path.stat().st_size
            for start,end in ((0,127),(size-256,size-1)):
                request = urllib.request.Request(url+'/'+name,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=10) as response, path.open('rb') as file:
                    file.seek(start)
                    if (response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}' or
                        response.read() != file.read(end-start+1) or response.headers.get_content_type() != 'video/mp4'):
                        raise ValueError('Live video Range response differs')
            checks.append({'file':name,'sha256':identity['sha256'],'head_and_tail_range_pass':True})
    for private in ('private-receipt.json','../private-receipt.json','request.json'):
        try:
            urllib.request.urlopen(url+'/'+private,timeout=5)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        else:
            raise ValueError('Private file must not be served')
    info = json.loads((RESEARCH/'trt-public-api-cpu-v2/object-info.json').read_text(encoding='utf8'))
    workflows = {}
    for mode,task,name in (('check',None,'Runtime_Check'),('compile',None,'Compile'),('decoder','T2VA','T2VA_Decoder'),('full','I2VA','I2VA_Full'),('compare','T2VA','Same_Latent_Compare')):
        source = json.loads((RESEARCH/'pilot-api-drafts'/f'{task}_native8.prompt.json').read_text(encoding='utf8')) if task else None
        path = PROJECT/f'examples/workflows/30-trt-vae/2026-09-10_H3_TRT_VAE_{name}_EXP.json'
        workflows[name] = {'audit':audit_candidate(recipe(source,mode),json.loads(path.read_text(encoding='utf8')),info),'sha256':digest_file(path)}
    return {'status':f'all{count*2}_live_ranges_private404_and_five_workflows_pass','url':url+'/review.html',
        'static_text_inline_verified':augmented,
        'videos':checks,'workflows':workflows,'no_browser_action':True,
        'limits':'Actual HTTP bytes/schema checks,not browser playback or human acceptance. Source MP4s were independently fully decoded before this copy; no frontend export or upload.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    args = parser.parse_args()
    report = audit(args.root,args.port)
    write_new_json(args.root/'delivery-audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('videos','workflows')},indent=2))
