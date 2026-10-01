# run_recap_grid.ps1 -- systematic closed-loop grid of the thesis
# (3 routes x 60/100/140 m, heading rule "compass", simulator world).
#
# Run from the repository root with the virtual environment active and
# Unreal in Play mode:
#
#     .\scripts\run_recap_grid.ps1
#
# The script is idempotent: finished runs (summary.json present) are skipped,
# interrupted runs are continued with --resume, every run gets up to three
# attempts. If Unreal crashes, restart it and start the script again.
#
# Teach libraries are expected under data\teach_<route> (50x50 px / 10 gray
# levels at 60, 100 and 140 m). In the thesis the Innenstadt library was
# stored as data\teach_rgb; the run configuration of every reported run is
# recorded in its params.json.

$simcfg = "sim_config"
$routes = @(
    @{ name = "innenstadt"; route = "routes\route_s.csv";        teach = "data\teach_innenstadt"; gz = 640 },
    @{ name = "nordpark";   route = "routes\route_nordpark.csv"; teach = "data\teach_nordpark";   gz = 656 },
    @{ name = "sudbrack";   route = "routes\route_sudbrack.csv"; teach = "data\teach_sudbrack";   gz = 663 }
)
foreach ($r in $routes) {
    foreach ($alt in 60, 100, 140) {
        $out = "data\recap\{0}_alt{1:d4}m_compass" -f $r.name, $alt
        if (Test-Path "$out\summary.json") { Write-Host "skip $out (done)"; continue }
        for ($try = 1; $try -le 3; $try++) {
            $cli = @("scripts\recapitulate_route.py", "--world", "sim", "--route", $r.route,
                     "--alt", $alt, "--teach-root", $r.teach,
                     "--origin-lat", 52.040111, "--origin-lon", 8.495097, "--origin-h", 800,
                     "--sim-config", $simcfg, "--ground-z0", $r.gz, "--workers", 12,
                     "--heading-mode", "compass", "--out", $out)
            if (Test-Path "$out\steps.csv") { $cli += "--resume" }
            Write-Host "== $out (attempt $try)"
            python $cli
            if (Test-Path "$out\summary.json") { break }
        }
    }
}

# Figures and table for the 3 x 3 grid (rows = routes, columns = altitudes):
$grid = Get-ChildItem data\recap -Directory |
        Where-Object Name -match '_alt0(060|100|140)m_compass$' |
        ForEach-Object FullName
python scripts\aggregate_recap.py --runs $grid --out data\figs_recap_grid --cols 3
