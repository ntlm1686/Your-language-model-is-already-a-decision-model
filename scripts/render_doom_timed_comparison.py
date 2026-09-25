"""Render two Doom runs on a shared real-time clock and evaluation window.

TIMEOUT labels the end of the evaluation window. An episode that finishes
sooner holds its final score; its actual termination remains in the summary.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import subprocess
import tempfile

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_doom_gifs import render_frame

class Replay:
    def __init__(self, video, trace, summary, name):
        self.rows=[r for r in map(json.loads,trace.read_text().splitlines()) if r['episode']==0 and r['policy']=='api_position_none']
        self.summary=json.loads(summary.read_text())
        self.episode=self.summary['results']['api_position_none']['episodes'][0]
        self.reader=imageio.get_reader(video)
        assert self.reader.count_frames()==len(self.rows)==self.episode['steps']
        self.times=[r['elapsed_after_s'] for r in self.rows]
        assert self.times==sorted(self.times)
        self.name=name;self.cache={}
    def at(self, seconds):
        index=bisect_right(self.times,seconds)-1
        finished=seconds>=self.episode['wall_seconds']
        if index not in self.cache:
            row=self.rows[max(0,index)]
            panel=render_frame(self.reader.get_data(max(0,index)),row,self.name)
            if index<0:
                d=ImageDraw.Draw(panel)
                d.rectangle((690,82,1260,265),fill=(11,20,27))
                d.text((700,100),'Waiting for first decision...',font=FONT_SMALL,fill='white')
                d.rectangle((20,605,680,640),fill=(11,20,27))
            self.cache[index]=panel
        kills=self.episode['kills'] if finished else (self.rows[index]['kills_before'] if index>=0 else 0)
        return self.cache[index],kills,finished

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans'
FONT_BIG=ImageFont.truetype(FONT+'-Bold.ttf',40)
FONT_SMALL=ImageFont.truetype(FONT+'.ttf',22)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for model in ['qwen','jev']:
        for kind in ['video','trace','summary']:
            p.add_argument(f'--{model}-{kind}',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=15)
    a=p.parse_args()
    q=Replay(a.qwen_video,a.qwen_trace,a.qwen_summary,'Qwen3.5-9B')
    j=Replay(a.jev_video,a.jev_trace,a.jev_summary,'Jev 1.13.0')
    assert q.summary['max_wall_seconds']==j.summary['max_wall_seconds']==a.seconds
    assert q.summary['hard_deadline'] and j.summary['hard_deadline']
    fps=8;budget_frames=round(a.seconds*fps)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='doom-wall-time-') as temp:
            video=Path(temp)/'combined.mp4'
            writer=imageio.get_writer(video,fps=fps,codec='libx264',quality=7,macro_block_size=16)
            try:
                for frame_index in range(budget_frames+16):
                    seconds=min(frame_index/fps,a.seconds)
                    expired=frame_index>=budget_frames
                    canvas=Image.new('RGB',(2560,832),(11,20,27))
                    scores=[]
                    for x,replay,color in [(0,q,(107,207,255)),(1280,j,(255,199,102))]:
                        panel,kills,finished=replay.at(seconds)
                        if expired:kills=replay.episode['kills']
                        scores.append(kills)
                        canvas.paste(panel,(x,112))
                        d=ImageDraw.Draw(canvas)
                        header_x=x+(32 if x==0 else 150)
                        d.text((header_x,14),f'{replay.name}  |  Kills: {kills}',font=FONT_BIG,fill=color)
                        status='TIMEOUT' if expired else ('FINAL SCORE HELD' if finished else 'PLAYING')
                        d.text((header_x,72),status,font=FONT_SMALL,fill=(190,205,215))
                        if expired:
                            d.rounded_rectangle((x+48,360,x+640,506),radius=12,fill=(11,20,27),outline=color,width=3)
                            d.text((x+344,397),'TIMEOUT',font=FONT_BIG,anchor='mm',fill=color)
                            d.text((x+344,446),f'{a.seconds:g} s evaluation window ended',font=FONT_SMALL,anchor='mm',fill='white')
                            d.text((x+344,478),f'Final kills: {kills}',font=FONT_SMALL,anchor='mm',fill='white')
                    d=ImageDraw.Draw(canvas)
                    d.text((1280,34),f'{seconds:04.1f} / {a.seconds:g} s',font=FONT_BIG,anchor='mm',fill='white')
                    d.text((1280,83),f'{scores[0]} : {scores[1]}',font=FONT_SMALL,anchor='mm',fill='white')
                    writer.append_data(np.asarray(canvas))
            finally:writer.close()
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(video),
                '-filter_complex','[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5',
                '-loop','0',str(a.output)],check=True)
    finally:
        q.reader.close();j.reader.close()
    print(a.output)

if __name__=='__main__':main()
