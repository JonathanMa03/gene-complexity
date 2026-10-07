"""Bounded parallel HTTP ranges, appended atomically per batch to a resumable prefix."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,time
import requests
BLOCK=16*1024*1024

def fetch(url,start,end):
 for attempt in range(6):
  try:
   with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,90)) as response:
    response.raise_for_status()
    if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise RuntimeError('Server did not honor exact byte range')
    data=response.content
    if len(data)!=end-start+1:raise RuntimeError('Incomplete byte range')
    return data
  except (requests.RequestException,RuntimeError) as error:
   print(json.dumps(dict(stage='range_retry',start=start,attempt=attempt+1,error=str(error))),flush=True)
   if attempt==5:raise
   time.sleep(min(2**attempt,20))

def download(url,path,size,connections=4):
 path=Path(path); start_time=time.monotonic(); offset=path.stat().st_size if path.exists() else 0
 if offset>size:raise RuntimeError('Partial file exceeds expected size')
 print(json.dumps(dict(stage='transfer_start',file=path.name,resume_bytes=offset,expected_bytes=size)),flush=True)
 with ThreadPoolExecutor(max_workers=connections) as pool, path.open('ab') as handle:
  while offset<size:
   starts=list(range(offset,min(offset+connections*BLOCK,size),BLOCK))
   futures=[pool.submit(fetch,url,s,min(s+BLOCK,size)-1) for s in starts]
   # Each completed block is durable; retries reuse this exact contiguous prefix.
   for s,future in zip(starts,futures):
    data=future.result();assert s==offset
    handle.write(data);handle.flush();os.fsync(handle.fileno());offset+=len(data)
   print(json.dumps(dict(stage='transfer_progress',file=path.name,bytes=offset,total=size,seconds=round(time.monotonic()-start_time,2))),flush=True)
 if path.stat().st_size!=size:raise RuntimeError('Downloaded file size mismatch')
