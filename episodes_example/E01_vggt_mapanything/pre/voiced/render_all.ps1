# Render the voiced variant (9 shots; long shots --jobs 2 by default; OMP/MKL threads total 6), then concat into one silent video.
# Usage: powershell -File render_all.ps1 [-Jobs 1] [-Shots p1_vggt,...]   (-Jobs 1 to leave room for other work)
#   $env:EXPLAINER_PY  = python of the render env (default: python on PATH)
#   $env:EXPLAINER_CPU = optional hex core mask for a shared machine, e.g. 'F' = cores 0-3 (children inherit it)
# Under `cmd /c start /WAIT` the exit code of render_shot is not reliable: judge by the produced files.
param([int]$Jobs = 2, [string[]]$Shots = @('c0_open', 'c1_vggt', 'p1_vggt', 'c2_mapanything', 'p2_mapanything', 'c3_compare', 'p3_compare', 'c4_test', 'p4_our_test'))
$v = $PSScriptRoot
$root = (Resolve-Path "$v\..\..\..\..").Path
$py = if ($env:EXPLAINER_PY) { $env:EXPLAINER_PY } else { 'python' }
$th = [string][int](6 / $Jobs)
$env:PYTHONUTF8 = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:OMP_NUM_THREADS = $th; $env:MKL_NUM_THREADS = $th
New-Item -ItemType Directory -Force "$v\_logs" | Out-Null
function Run-Py([string[]]$argv, [string]$log) {
    if ($env:EXPLAINER_CPU) {
        cmd /c start `"`" /B /WAIT /BELOWNORMAL /AFFINITY $env:EXPLAINER_CPU $py -I -B -X utf8 @argv > $log 2>&1
    } else {
        & $py -I -B -X utf8 @argv > $log 2>&1
    }
}
foreach ($s in $Shots) {
    $nj = if ($s.StartsWith('p')) { $Jobs } else { 1 }
    $t0 = Get-Date
    Run-Py @("$root\tools\render_shot.py", "$v\shots\$s.py", '--jobs', "$nj") "$v\_logs\render_$s.log"
    $meta = "$v\shots\render\$s\meta.json"          # a crashed render can still leave a new mp4; meta.json is written last
    $ok = (Test-Path $meta) -and ((Get-Item $meta).LastWriteTime -gt $t0)
    "{0} {1} {2:n0}s" -f $s, $(if ($ok) { 'OK' } else { 'MISSING' }), ((Get-Date) - $t0).TotalSeconds
}
Run-Py @("$root\tools\concat.py", "$v", '--out', "$v\render\E01_pre_en_voiced_silent.mp4") "$v\_logs\concat.log"
Get-Content "$v\_logs\concat.log"
