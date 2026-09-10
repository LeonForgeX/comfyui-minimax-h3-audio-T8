"""Deterministic CPU-only FI boundary sources; these are NOT model outputs.

Generate 30fps moving texture/HUD, 30fps low texture with a brief flash, and a
24fps hard cut with an explicit right-frame marker. No GPU/worker, audio, upscale
or acceptance claim. Files are separate from every H3 source/candidate.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.dlss_fi_media import inspect_source  # noqa: E402

CASES = {'hud30':(30,60,()),'flash_low_texture30':(30,60,()),'hard_cut24':(24,48,(24,))}
WIDTH,HEIGHT = 512,256


def render(case,index):
    import cv2
    if case not in CASES or type(index) is not int or not 0 <= index < CASES[case][1]:
        raise ValueError('Unknown fixed boundary frame')
    y,x = np.mgrid[:HEIGHT,:WIDTH]
    if case == 'flash_low_texture30':
        level = np.clip(100+x/20+y/40,0,255).astype(np.uint8)
        rgb = np.repeat(level[...,None],3,axis=2)
        cv2.circle(rgb,(70+index*5,140),23,(160,170,180),-1)
        if index in (20,21):
            rgb[:] = 245
    else:
        phase = index*5
        grid = (((x+phase)//24+y//24)%2).astype(np.uint8)
        rgb = np.empty((HEIGHT,WIDTH,3),np.uint8)
        rgb[:]=np.where(grid[...,None]>0,np.array([38,92,139]),np.array([133,170,194]))
        if case=='hard_cut24' and index>=24:
            rgb=rgb[...,::-1].copy()
            rgb=np.flip(rgb,axis=0).copy()
        cv2.rectangle(rgb,(60+phase%300,90),(125+phase%300,190),(205,175,70),-1)
        cv2.rectangle(rgb,(0,0),(WIDTH-1,37),(12,16,24),-1)
        cv2.putText(rgb,'SCORE 012345   HEALTH 100',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.62,(230,230,230),1,cv2.LINE_AA)
    return np.ascontiguousarray(rgb)


def build(root):
    import av
    root=Path(root).resolve()
    research=Path(__file__).resolve().parents[1]/'artifacts/acceleration-research-20260909'
    if root.exists() or root==research or not root.is_relative_to(research):
        raise ValueError('New exclusive research directory required')
    root.mkdir(parents=True)
    manifest={'status':'CPU_synthetic_sources_only_no_FI_execution','cases':{}}
    for name,(rate,count,cuts) in CASES.items():
        path=root/(name+'.mp4')
        with path.open('xb') as file,av.open(file,'w',format='mp4',options={'movflags':'faststart'}) as container:
            stream=container.add_stream('libx264',rate=rate)
            stream.width,stream.height,stream.pix_fmt=WIDTH,HEIGHT,'yuv420p'
            stream.codec_context.time_base=Fraction(1,rate)
            stream.codec_context.max_b_frames=0
            stream.codec_context.thread_count=1
            stream.options={'crf':'18','preset':'medium','threads':'1'}
            for field in ('color_primaries','color_trc','colorspace','color_range'):
                setattr(stream.codec_context,field,1)
            for index in range(count):
                frame=av.VideoFrame.from_ndarray(render(name,index),format='rgb24')
                frame.pts,frame.time_base=index,Fraction(1,rate)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        source=inspect_source(path,cuts=cuts)
        if source['plan'].source_count!=count or source['plan'].source_rate!=rate or source['audio_packets']:
            raise ValueError('Synthetic encoded file differs from its declared source')
        manifest['cases'][name]={'file':source['file'],'width':WIDTH,'height':HEIGHT,'source_frames':count,
                                 'source_fps':rate,'cuts':list(cuts),'has_audio':False}
    with (root/'manifest.json').open('x',encoding='utf8') as output:
        json.dump(manifest,output,ensure_ascii=False,indent=2)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().run_root),ensure_ascii=False))
