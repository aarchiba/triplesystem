#!/bin/sh
#PBS -l nodes=1:ppn=1:compute:new

echo foo >/tmp/bar
rsync /tmp/bar nimrod:
