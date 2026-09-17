# Syncing results between PC and laptop

Two channels, deliberately disjoint, so they never fight each other:

| Channel | Carries | Size |
|---|---|---|
| **git** (GitHub) | code, figures, CSVs, notebooks, `run_config.json`, summaries | ~25 MB |
| **Syncthing** (peer-to-peer) | raw `*.parquet` / `*.pkl` trajectories under `results/` | ~4.9 GB, 2,536 files |

The split is enforced by [`results/.stignore`](../results/.stignore): Syncthing
ignores every extension git tracks, plus run logs and `progress.jsonl` (the
sweep resume journal, which must stay machine-local or a resumed run on one
machine will think the other machine's cells are already done).

GitHub is not an option for the raw data: free Git LFS is 1 GB storage and
1 GB/month bandwidth, and 2,536 objects would be slow even if it fit.

## PC (`neotheone`) — already configured

- Syncthing 2.1.5 installed via winget.
- Config home: `%LOCALAPPDATA%\Syncthing`
- Folder ID `aa-results` -> `D:\Research\argumentative-awareness\results`, type `sendreceive`.
- Runs at logon via the Scheduled Task named **Syncthing**.
- Web UI: <http://127.0.0.1:8384>
- **Device ID:** `LL4J5ZZ-5FC2QRY-LDJ2WO5-377BREF-2V2OAQW-R2LKNKN-TEPFESV-KYE33QY`

## Laptop — one-time setup

1. Clone the repo first, so `results/.stignore` exists before Syncthing scans:

       git clone https://github.com/keremoner/argumentative-awareness.git
       cd argumentative-awareness

2. Install Syncthing:

       winget install --id Syncthing.Syncthing

3. Start it (it opens <http://127.0.0.1:8384>):

       syncthing serve --no-restart

4. **Add Remote Device** -> paste the PC device ID above -> Save.
   On the PC's web UI, accept the incoming device when it appears.

5. **Add Folder**:
   - Folder ID: `aa-results`  (must match exactly)
   - Folder Path: the laptop's `...\argumentative-awareness\results`
   - Sharing tab: tick `neotheone`
   - Save. The 4.9 GB pulls over; LAN transfers are fast, and it falls back to
     relays when the machines are on different networks.

6. Make it start at logon (PowerShell, adjust the exe path if winget's differs):

       $exe = (Get-Command syncthing).Source
       Register-ScheduledTask -TaskName Syncthing -Force `
         -Action  (New-ScheduledTaskAction -Execute $exe -Argument 'serve --no-restart --no-browser') `
         -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME) `
         -Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero))

## Day to day

- `git pull` / `git push` as usual for code and analysis outputs.
- Raw data syncs on its own whenever both machines are powered on and online.
  Nothing to run.
- Both machines are `sendreceive`, so a new sweep on either side propagates to
  the other. If the same file is edited on both while disconnected, Syncthing
  keeps the loser as `*.sync-conflict-*` rather than discarding it.
