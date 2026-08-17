# SIH Demo Checklist

## Before the demo

- [ ] Python environment working: `.venv\Scripts\python.exe --version`
- [ ] Dependencies installed: `python -m pytest -q` passes
- [ ] Backend starts: `python -m uvicorn api.main:app --reload` → `http://127.0.0.1:8000/health` returns ok
- [ ] Frontend starts: `cd dashboard && npm run dev` → `http://localhost:5173/`
- [ ] Browser opens the dashboard
- [ ] Internet available for basemap tiles (map layers still render if offline)
- [ ] Demo date available: `http://127.0.0.1:8000/available-dates` contains `2025-01-01`
- [ ] PM2.5 layer loads
- [ ] AQI layer loads
- [ ] Hotspots load
- [ ] CPCB stations load
- [ ] Location click works
- [ ] No console errors

## Demo story

- [ ] Show the problem: sparse CPCB stations, coarse satellite data
- [ ] Show sparse stations (toggle CPCB stations)
- [ ] Show 1 km PM2.5 map
- [ ] Switch to 500 m — explain: *"spatial detail increases; this does not automatically imply higher accuracy"*
- [ ] Explain spatial refinement (downscaling ≠ image resizing)
- [ ] Show PM2.5-derived AQI
- [ ] Show predicted high-pollution zones (not confirmed emission sources)
- [ ] Click a location — show PM2.5, AQI, model, mode, resolution, limitations
- [ ] Show model information honestly: XGBoost, fallback, AOD unavailable, 16 rows, provisional
- [ ] Show feature importance — *"model contributions, not causal proof"*
- [ ] Show limitations panel (uncertainty deferred)

## Backup

- [ ] Keep terminal commands ready
- [ ] Keep Swagger available: `http://127.0.0.1:8000/docs`
- [ ] Keep screenshot/video fallback if internet fails
- [ ] Do not modify code immediately before demo
