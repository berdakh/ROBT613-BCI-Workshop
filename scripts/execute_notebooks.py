"""Execute real-data notebooks, retain outputs, and record failures honestly."""
from pathlib import Path
import argparse,os,json,sys,tempfile,time
import nbformat
from nbclient import NotebookClient
parser=argparse.ArgumentParser()
parser.add_argument('--only',nargs='+',help='Two-digit lesson numbers, e.g. 00 06')
parser.add_argument('--timeout',type=int,default=1200)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
results=[]
with tempfile.TemporaryDirectory(prefix='bci-kernel-') as temp:
    kernel=Path(temp)/'kernels'/'bci-course'; kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'BCI course','language':'python'}))
    os.environ['JUPYTER_PATH']=temp+os.pathsep+os.environ.get('JUPYTER_PATH','')
    for path in sorted((root/'notebooks').glob('*.ipynb')):
        if args.only and path.stem[:2] not in args.only: continue
        nb=nbformat.read(path,as_version=4); start=time.time()
        print('Executing',path.name,flush=True)
        try:
            NotebookClient(nb,timeout=args.timeout,kernel_name='bci-course',resources={'metadata':{'path':str(root)}}).execute()
            nbformat.write(nb,path)
            result={'notebook':path.name,'status':'passed','seconds':round(time.time()-start,1)}
        except Exception as exc:
            result={'notebook':path.name,'status':'failed','error':str(exc)[-2000:]}
        results.append(result); print(result,flush=True)
(root/'execution_report.json').write_text(json.dumps(results,indent=2))
if any(r['status']=='failed' for r in results): raise SystemExit(1)
