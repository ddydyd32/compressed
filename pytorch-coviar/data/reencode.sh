#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./reencode.sh [input dir] [output dir]"
fi

indir=$1
outdir=$2

mkdir outdir
if [[ ! -d "${outdir}" ]]; then
  echo "${outdir} doesn't exist. Creating it.";
  mkdir -p ${outdir}
fi

for c in $(ls ${indir})
do
	for inname in $(ls ${indir}/${c}/*avi)
	do
		class_path="$(dirname "$inname")"
		class_name="${class_path##*/}"

		outname="${outdir}/${class_name}/${inname##*/}"
		outname="${outname%.*}.mp4"

		mkdir -p "$(dirname "$outname")"
		ffmpeg -i ${inname} -vf scale=336:256,setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo ${outname}
		# ffmpeg -i ${inname} -vf pad="ih*16/9:ih:(ow-iw)/2:(oh-ih)/2" -q:v 1 -c:v mpeg4 -f rawvideo ${outname}

	done
done