"""Fetch pinned WebShop full data and build its original Lucene index."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'external/webshop'
REV='64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd'

def main():
    if not os.environ.get('JAVA_HOME'):
        raise SystemExit('Set JAVA_HOME to a Java 11 JDK before setup.')
    if not REPO.exists():
        subprocess.run(['git','clone','--filter=blob:none','https://github.com/princeton-nlp/WebShop.git',str(REPO)],check=True)
        subprocess.run(['git','checkout',REV],cwd=REPO,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert actual==REV,(actual,REV)
    snapshot_download('HongbangYuan/webshop',repo_type='dataset',
        revision='0129d4a81dbdb827e76afd20a1e2c38b61098613',
        allow_patterns=['items_shuffle.json','items_ins_v2.json','items_human_ins.json'],local_dir=REPO/'data')
    checks={'items_shuffle.json':'2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8',
            'items_ins_v2.json':'1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e'}
    for name,expected in checks.items():
        with (REPO/'data'/name).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
    utils=REPO/'web_agent_site/utils.py'
    utils.write_text(utils.read_text().replace('items_shuffle_1000.json','items_shuffle.json').replace('items_ins_v2_1000.json','items_ins_v2.json'))
    search=REPO/'search_engine'
    if (search/'indexes/.complete').exists():
        print('Verified data; index already built.')
        return
    for name in ('resources','resources_100','resources_1k','resources_100k'):
        (search/name).mkdir(exist_ok=True)
    subprocess.run([sys.executable,'convert_product_file_format.py'],cwd=search,check=True)
    subprocess.run([sys.executable,'-m','pyserini.index.lucene','--collection','JsonCollection',
        '--input','resources','--index','indexes','--generator','DefaultLuceneDocumentGenerator',
        '--threads','4','--storePositions','--storeDocvectors','--storeRaw'],cwd=search,check=True)
    (search/'indexes/.complete').write_text(REV+'\n')

if __name__=='__main__':
    main()
