#!/usr/bin/env bash
# Times each audio filter used by the studio on a 3 s clip (CI diagnosis).
out=${1:-diag.txt}; d=$(mktemp -d)
{ ffmpeg -version | head -1; lsb_release -ds 2>/dev/null; nproc; } > "$out"
ffmpeg -loglevel error -y -f lavfi -i color=c=red:s=720x1280:d=3 -f lavfi -i sine=frequency=300:duration=2.4 -af adelay=300\|300,apad -t 3 -c:v libx264 -preset ultrafast -c:a aac "$d/in.mp4"
M="$PWD/live_ai/assets/rnnoise-std.rnnn"
run() { local name="$1"; shift; local t0=$(date +%s.%N); timeout 60 ffmpeg -loglevel error -nostdin -y -i "$d/in.mp4" -map 0:v:0 -map 0:a:0 -vf "scale=720:1280,fps=30,format=yuv420p" -af "$1" -t 3 -c:v libx264 -preset veryfast -c:a aac "$d/o.mp4" 2> "$d/err.txt"; local rc=$?; echo "$name rc=$rc secs=$(echo "$(date +%s.%N) - $t0" | bc) $(head -c 200 "$d/err.txt")" >> "$out"; }
run plain "aresample=48000,aformat=channel_layouts=stereo,apad=whole_dur=3"
run apad_bare "aresample=48000,aformat=channel_layouts=stereo,apad"
run arnndn "aresample=48000,highpass=f=70,arnndn=m=$M,aformat=channel_layouts=stereo,apad=whole_dur=3"
run arnndn_mono_first "aresample=48000,aformat=channel_layouts=mono,arnndn=m=$M,aformat=channel_layouts=stereo"
run afftdn "aresample=48000,highpass=f=70,afftdn=nr=20:nf=-35:tn=1,aformat=channel_layouts=stereo,apad=whole_dur=3"
run volume_expr "aresample=48000,volume='if(gt(between(t,0.30,2.70),0),1,0.1)':eval=frame,aformat=channel_layouts=stereo,apad=whole_dur=3"
run full "aresample=48000,highpass=f=70,arnndn=m=$M,volume='if(gt(between(t,0.30,2.70),0),1,0.1)':eval=frame,aformat=channel_layouts=stereo,apad=whole_dur=3"
run highpass_only "aresample=48000,highpass=f=70,aformat=channel_layouts=stereo,apad=whole_dur=3"
run used_now "aresample=48000,aformat=channel_layouts=mono,arnndn=m=$M,volume='if(gt(between(t,0.30,2.70),0),1,0.1)':eval=frame,aformat=channel_layouts=stereo,apad=whole_dur=3"
run afftdn_now "aresample=48000,aformat=channel_layouts=mono,afftdn=nr=20:nf=-35,aformat=channel_layouts=stereo,apad=whole_dur=3"
cat "$out"
