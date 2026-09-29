//! npu-watt core: per-rail power/energy measurement for NPU-pinned LLM inference on Snapdragon X
//! (Windows on ARM), built on the Windows Energy Meter.
//!
//! Task Manager's NPU graph can read 0% while the Hexagon cDSP is saturated, because work reaching
//! the DSP over FastRPC (llama.cpp's HTP backend, QNN, ...) is not a WDDM engine. The Energy Meter
//! exposes cumulative energy per rail (`npu`, `gpu`, `cpu_cluster_*`, `memory`, `soc`, `system`, ...),
//! which is enough to tell whether the NPU is working, how hard, and what it costs per token.
//!
//! Typical use from Rust:
//!
//! ```no_run
//! use npu_watt::*;
//! use std::time::Duration;
//! let idle = measure_idle(Duration::from_secs(3)).unwrap();
//! let rec = Recorder::start(Duration::from_millis(250), CpuSource::None).unwrap();
//! // ... run the workload ...
//! let samples = rec.stop().unwrap();
//! let report = analyze(&samples, &idle, &AnalyzeOpts { label: "baseline".into(), tokens: Some(250.0), tok_s: Some(28.3), ..Default::default() });
//! println!("{}", report.render_text());
//! ```

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use windows::Win32::Foundation::{CloseHandle, FILETIME, HANDLE};
use windows::Win32::System::JobObjects::*;
use windows::Win32::System::Performance::*;
use windows::Win32::System::Power::*;
use windows::Win32::System::Threading::*;
use windows::core::{PCWSTR, w};

/// The Energy counter is cumulative picowatt-hours.
const PWH_TO_J: f64 = 3.6e-9;
/// PDH_FMT_DOUBLE | PDH_FMT_NOSCALE (0x1000): raw counter units, no PDH scaling (pWh and mW here).
const RAW_DOUBLE: PDH_FMT = PDH_FMT(PDH_FMT_DOUBLE.0 | 0x1000);

/// Rail name (lowercase, e.g. `npu`, `cpu_cluster_0`) to a value.
pub type Rails = BTreeMap<String, f64>;

fn get(r: &Rails, k: &str) -> f64 {
    r.get(k).copied().unwrap_or(0.0)
}

fn cpu_sum(r: &Rails) -> f64 {
    r.iter().filter(|(k, _)| k.starts_with("cpu_cluster")).map(|(_, v)| v).sum()
}

fn check(status: u32, what: &str) -> Result<(), String> {
    if status == 0 { Ok(()) } else { Err(format!("{what} failed: PDH status 0x{status:08X}")) }
}

// ---------------------------------------------------------------------------------------------
// Meter: the Energy Meter counters
// ---------------------------------------------------------------------------------------------

/// A PDH query over `\Energy Meter(*)\Energy` and `\Power`.
pub struct Meter {
    query: PDH_HQUERY,
    energy: PDH_HCOUNTER,
    power: PDH_HCOUNTER,
}

impl Drop for Meter {
    fn drop(&mut self) {
        unsafe {
            PdhCloseQuery(self.query);
        }
    }
}

impl Meter {
    /// Opens the counters and collects once (PDH cooks values from two collections).
    pub fn open() -> Result<Meter, String> {
        unsafe {
            let mut query = PDH_HQUERY::default();
            check(PdhOpenQueryW(PCWSTR::null(), 0, &mut query), "PdhOpenQuery")?;
            let mut energy = PDH_HCOUNTER::default();
            let mut power = PDH_HCOUNTER::default();
            check(
                PdhAddEnglishCounterW(query, w!("\\Energy Meter(*)\\Energy"), 0, &mut energy),
                "add Energy counter (no Energy Meter on this machine?)",
            )?;
            check(PdhAddEnglishCounterW(query, w!("\\Energy Meter(*)\\Power"), 0, &mut power), "add Power counter")?;
            check(PdhCollectQueryData(query), "PdhCollectQueryData")?;
            std::thread::sleep(Duration::from_millis(100));
            Ok(Meter { query, energy, power })
        }
    }

    /// Returns (cumulative energy per rail in joules, instantaneous power per rail in watts).
    pub fn sample(&self) -> Result<(Rails, Rails), String> {
        unsafe {
            check(PdhCollectQueryData(self.query), "PdhCollectQueryData")?;
        }
        let e = self.read(self.energy)?;
        let p = self.read(self.power)?;
        Ok((
            e.into_iter().filter(|(k, _)| k != "_total").map(|(k, v)| (k, v * PWH_TO_J)).collect(),
            p.into_iter().filter(|(k, _)| k != "_total").map(|(k, v)| (k, v / 1000.0)).collect(),
        ))
    }

    fn read(&self, counter: PDH_HCOUNTER) -> Result<Rails, String> {
        unsafe {
            let mut bytes = 0u32;
            let mut count = 0u32;
            PdhGetFormattedCounterArrayW(counter, RAW_DOUBLE, &mut bytes, &mut count, None);
            let mut buf = vec![0u64; (bytes as usize).div_ceil(8) + 1];
            let ptr = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
            check(
                PdhGetFormattedCounterArrayW(counter, RAW_DOUBLE, &mut bytes, &mut count, Some(ptr)),
                "PdhGetFormattedCounterArray",
            )?;
            let mut out = Rails::new();
            for i in 0..count as usize {
                let item = &*ptr.add(i);
                out.insert(item.szName.to_string().unwrap_or_default().to_lowercase(), item.FmtValue.Anonymous.doubleValue);
            }
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Battery
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Battery {
    pub on_ac: bool,
    /// Present-state draw in W; positive while discharging, negative while charging.
    pub discharge_w: f64,
    pub remaining_wh: f64,
    pub full_wh: f64,
}

pub fn battery() -> Option<Battery> {
    unsafe {
        let mut s: SYSTEM_BATTERY_STATE = std::mem::zeroed();
        let st = CallNtPowerInformation(
            SystemBatteryState,
            None,
            0,
            Some(&mut s as *mut _ as *mut _),
            std::mem::size_of::<SYSTEM_BATTERY_STATE>() as u32,
        );
        if st.0 != 0 || !s.BatteryPresent {
            return None;
        }
        // Rate is a signed mW value that the struct declares unsigned; negative means discharging.
        let rate = s.Rate as i32 as f64 / 1000.0;
        Some(Battery {
            on_ac: s.AcOnLine,
            discharge_w: -rate,
            remaining_wh: s.RemainingCapacity as f64 / 1000.0,
            full_wh: s.MaxCapacity as f64 / 1000.0,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// CPU time attribution
// ---------------------------------------------------------------------------------------------

/// A Windows job object; every process (and descendant) assigned to it is counted.
pub struct Job(HANDLE);
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    pub fn new() -> Result<Job, String> {
        unsafe { CreateJobObjectW(None, PCWSTR::null()).map(Job).map_err(|e| e.to_string()) }
    }

    /// Assigns a process by raw handle (e.g. `child.as_raw_handle()`).
    pub fn assign_raw(&self, process: *mut std::ffi::c_void) -> Result<(), String> {
        unsafe { AssignProcessToJobObject(self.0, HANDLE(process)).map_err(|e| e.to_string()) }
    }

    pub fn cpu_seconds(&self) -> f64 {
        unsafe {
            let mut info: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = std::mem::zeroed();
            let ok = QueryInformationJobObject(
                Some(self.0),
                JobObjectBasicAccountingInformation,
                &mut info as *mut _ as *mut _,
                std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                None,
            );
            if ok.is_err() { 0.0 } else { (info.TotalUserTime + info.TotalKernelTime) as f64 / 1.0e7 }
        }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// Where per-process CPU time comes from during a recording.
pub enum CpuSource {
    /// No process attribution; only the cpu rails are reported.
    None,
    /// All processes in a job object (includes descendants).
    Job(Arc<Job>),
    /// Specific process ids (their own CPU time only, not descendants).
    Pids(Vec<u32>),
}

impl CpuSource {
    fn seconds(&self, last: &mut BTreeMap<u32, f64>) -> f64 {
        match self {
            CpuSource::None => 0.0,
            CpuSource::Job(j) => j.cpu_seconds(),
            CpuSource::Pids(pids) => {
                for &pid in pids {
                    unsafe {
                        if let Ok(h) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) {
                            let (mut c, mut e, mut k, mut u) = (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
                            if GetProcessTimes(h, &mut c, &mut e, &mut k, &mut u).is_ok() {
                                let ft = |f: FILETIME| ((f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64) as f64 / 1.0e7;
                                last.insert(pid, ft(k) + ft(u));
                            }
                            let _ = CloseHandle(h);
                        }
                    }
                }
                last.values().sum()
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Sample {
    /// Seconds since the recorder started.
    pub t: f64,
    /// Cumulative energy per rail, joules.
    pub energy_j: Rails,
    /// Instantaneous power per rail, watts.
    pub power_w: Rails,
    /// Cumulative CPU seconds of the attributed processes.
    pub cpu_s: f64,
    /// Battery discharge in W (positive = discharging) when a battery exists.
    pub batt_w: Option<f64>,
    pub batt_on_ac: Option<bool>,
    pub batt_remaining_wh: Option<f64>,
}

/// Averages per-rail power over `secs` with the machine otherwise as-is. Run this before the workload.
pub fn measure_idle(secs: Duration) -> Result<Rails, String> {
    let m = Meter::open()?;
    let a = m.sample()?.0;
    let t = Instant::now();
    std::thread::sleep(secs);
    let b = m.sample()?.0;
    let dt = t.elapsed().as_secs_f64();
    Ok(b.iter().map(|(k, v)| (k.clone(), (v - get(&a, k)) / dt)).collect())
}

/// Background sampler. `start` returns once the meter is open and the first sample is taken.
pub struct Recorder {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Result<Vec<Sample>, String>>>,
}

impl Recorder {
    pub fn start(interval: Duration, cpu: CpuSource) -> Result<Recorder, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let (tx, rx) = mpsc::channel::<Result<(), String>>();
        let thread = std::thread::spawn(move || -> Result<Vec<Sample>, String> {
            let meter = match Meter::open() {
                Ok(m) => {
                    let _ = tx.send(Ok(()));
                    m
                }
                Err(e) => {
                    let _ = tx.send(Err(e.clone()));
                    return Err(e);
                }
            };
            let start = Instant::now();
            let mut last_cpu = BTreeMap::new();
            let take = |last_cpu: &mut BTreeMap<u32, f64>| -> Result<Sample, String> {
                let (energy_j, power_w) = meter.sample()?;
                let b = battery();
                Ok(Sample {
                    t: start.elapsed().as_secs_f64(),
                    energy_j,
                    power_w,
                    cpu_s: cpu.seconds(last_cpu),
                    batt_w: b.map(|b| b.discharge_w),
                    batt_on_ac: b.map(|b| b.on_ac),
                    batt_remaining_wh: b.map(|b| b.remaining_wh),
                })
            };
            let mut out = vec![take(&mut last_cpu)?];
            while !stop2.load(Ordering::Relaxed) {
                std::thread::sleep(interval);
                out.push(take(&mut last_cpu)?);
            }
            Ok(out)
        });
        rx.recv().map_err(|e| e.to_string())??;
        Ok(Recorder { stop, thread: Some(thread) })
    }

    /// Stops sampling (taking one final sample) and returns everything recorded.
    pub fn stop(mut self) -> Result<Vec<Sample>, String> {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().map_err(|_| "sampler thread panicked".to_string())?
    }
}

// ---------------------------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AnalyzeOpts {
    pub label: String,
    pub tokens: Option<f64>,
    pub tok_s: Option<f64>,
    /// npu draw above idle that counts as "NPU active", watts.
    pub npu_active_w: f64,
    /// Extra GPU draw above idle tolerated in an NPU-only verdict, watts.
    pub gpu_tolerance_w: f64,
    /// Average cores of attributed CPU time tolerated in an NPU-only verdict.
    pub cpu_tolerance_cores: f64,
}

impl Default for AnalyzeOpts {
    fn default() -> Self {
        AnalyzeOpts { label: "run".into(), tokens: None, tok_s: None, npu_active_w: 0.75, gpu_tolerance_w: 0.3, cpu_tolerance_cores: 0.35 }
    }
}

/// Per-rail numbers over the NPU-active window.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RailStat {
    pub avg_w: f64,
    pub vs_idle_w: f64,
    pub energy_j: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Efficiency {
    pub j_per_token: f64,
    pub tok_s_per_w: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BatteryReport {
    pub on_ac_at_end: bool,
    pub full_wh: f64,
    pub start_pct: f64,
    pub end_pct: f64,
    /// Mean battery-reported draw over the NPU-active window while unplugged.
    pub discharge_w: Option<f64>,
    /// Hours of continuous decode from a full battery at the measured battery draw.
    pub hours_at_battery_w: Option<f64>,
    pub tokens_per_charge_at_battery_w: Option<f64>,
    /// The same from the `system` rail (valid on AC too, includes display and platform overhead).
    pub hours_at_system_w: Option<f64>,
    pub tokens_per_charge_at_system_w: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Report {
    pub label: String,
    pub wall_s: f64,
    /// Total time of intervals where the npu was clearly above idle.
    pub window_s: f64,
    pub idle_w: Rails,
    /// Keys: npu, gpu, cpu (all clusters summed), memory, soc, system, plus any other rail.
    pub rails: BTreeMap<String, RailStat>,
    /// Average cores of CPU time of the attributed processes over the window (None if not attributed).
    pub process_cores: Option<f64>,
    /// `npu-only`, `npu-mixed` or `npu-inactive`.
    pub verdict: String,
    pub verdict_notes: Vec<String>,
    pub tokens: Option<f64>,
    pub tok_s: Option<f64>,
    /// Keys: npu, soc, system.
    pub efficiency: BTreeMap<String, Efficiency>,
    pub battery: Option<BatteryReport>,
}

pub fn analyze(samples: &[Sample], idle_w: &Rails, o: &AnalyzeOpts) -> Report {
    let mut r = Report { label: o.label.clone(), idle_w: idle_w.clone(), tokens: o.tokens, tok_s: o.tok_s, ..Default::default() };
    if samples.len() < 2 {
        r.verdict = "npu-inactive".into();
        r.verdict_notes.push("fewer than two samples".into());
        return r;
    }
    r.wall_s = samples.last().unwrap().t - samples[0].t;
    let idle_npu = get(idle_w, "npu");
    let win: Vec<(&Sample, &Sample)> = samples
        .windows(2)
        .filter(|w| {
            let dt = w[1].t - w[0].t;
            dt > 0.0 && (get(&w[1].energy_j, "npu") - get(&w[0].energy_j, "npu")) / dt > idle_npu + o.npu_active_w
        })
        .map(|w| (&w[0], &w[1]))
        .collect();
    let has_batt = samples[0].batt_w.is_some();
    r.battery = has_batt.then(|| BatteryReport {
        on_ac_at_end: samples.last().unwrap().batt_on_ac.unwrap_or(true),
        ..Default::default()
    });
    if win.is_empty() {
        r.verdict = "npu-inactive".into();
        r.verdict_notes.push(format!("no interval had npu draw more than {}W above idle", o.npu_active_w));
        return r;
    }
    let dur: f64 = win.iter().map(|(a, b)| b.t - a.t).sum();
    r.window_s = dur;
    let de = |rail: &str| -> f64 { win.iter().map(|(a, b)| get(&b.energy_j, rail) - get(&a.energy_j, rail)).sum() };
    let rail_names: Vec<String> = samples[0].energy_j.keys().filter(|k| !k.starts_with("cpu_cluster")).cloned().collect();
    for name in &rail_names {
        let j = de(name);
        r.rails.insert(name.clone(), RailStat { avg_w: j / dur, vs_idle_w: j / dur - get(idle_w, name), energy_j: j });
    }
    let cpu_j: f64 = samples[0].energy_j.keys().filter(|k| k.starts_with("cpu_cluster")).map(|k| de(k)).sum();
    r.rails.insert("cpu".into(), RailStat { avg_w: cpu_j / dur, vs_idle_w: cpu_j / dur - cpu_sum(idle_w), energy_j: cpu_j });

    let total_cpu = win.iter().map(|(a, b)| b.cpu_s - a.cpu_s).sum::<f64>();
    let attributed = samples.last().unwrap().cpu_s > 0.0;
    r.process_cores = attributed.then(|| total_cpu / dur);

    let gpu_d = r.rails.get("gpu").map(|s| s.vs_idle_w).unwrap_or(0.0);
    if gpu_d > o.gpu_tolerance_w {
        r.verdict_notes.push(format!("gpu +{gpu_d:.2}W over idle"));
    }
    if let Some(c) = r.process_cores {
        if c > o.cpu_tolerance_cores {
            r.verdict_notes.push(format!("attributed processes used {c:.2} CPU cores"));
        }
    }
    r.verdict = if r.verdict_notes.is_empty() { "npu-only" } else { "npu-mixed" }.into();

    if let (Some(n), Some(rate)) = (o.tokens, o.tok_s) {
        let _ = n;
        for k in ["npu", "soc", "system"] {
            if let Some(s) = r.rails.get(k) {
                r.efficiency.insert(k.into(), Efficiency { j_per_token: s.avg_w / rate, tok_s_per_w: rate / s.avg_w });
            }
        }
    }

    if let Some(b) = r.battery.as_mut() {
        let first = samples[0].batt_remaining_wh.unwrap_or(0.0);
        let last = samples.last().unwrap().batt_remaining_wh.unwrap_or(0.0);
        let full = battery().map(|b| b.full_wh).unwrap_or(0.0);
        b.full_wh = full;
        if full > 0.0 {
            b.start_pct = 100.0 * first / full;
            b.end_pct = 100.0 * last / full;
        }
        let drain: Vec<f64> = win.iter().filter(|(a, b2)| a.batt_on_ac == Some(false) && b2.batt_on_ac == Some(false)).filter_map(|(_, b2)| b2.batt_w).collect();
        if !drain.is_empty() {
            b.discharge_w = Some(drain.iter().sum::<f64>() / drain.len() as f64);
        }
        let per_charge = |w: f64| (full / w, o.tok_s.map(|t| full / w * 3600.0 * t));
        if let Some(w) = b.discharge_w.filter(|w| *w > 0.5) {
            let (h, t) = per_charge(w);
            b.hours_at_battery_w = Some(h);
            b.tokens_per_charge_at_battery_w = t;
        }
        let sys = r.rails["system"].avg_w;
        if sys > 0.5 && full > 0.0 {
            let (h, t) = per_charge(sys);
            b.hours_at_system_w = Some(h);
            b.tokens_per_charge_at_system_w = t;
        }
    }
    r
}

impl Report {
    pub fn render_text(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "================ npu-watt: {} ================", self.label);
        let g = |k: &str| get(&self.idle_w, k);
        let _ = writeln!(s, "wall {:.1}s; idle baseline: npu {:.2}W gpu {:.2}W soc {:.2}W system {:.2}W", self.wall_s, g("npu"), g("gpu"), g("soc"), g("system"));
        if self.verdict == "npu-inactive" {
            let _ = writeln!(s, "VERDICT: NPU NEVER ACTIVE ({})", self.verdict_notes.join("; "));
            return s;
        }
        let _ = writeln!(s, "NPU-active window: {:.1}s", self.window_s);
        let _ = writeln!(s, "{:<10} {:>9} {:>11} {:>10}", "rail", "avg W", "vs idle", "energy J");
        for k in ["npu", "gpu", "cpu", "memory", "soc", "system"] {
            if let Some(st) = self.rails.get(k) {
                let _ = writeln!(s, "{k:<10} {:9.2} {:+11.2} {:10.1}", st.avg_w, st.vs_idle_w, st.energy_j);
            }
        }
        if let Some(c) = self.process_cores {
            let _ = writeln!(s, "attributed process CPU: {c:.2} cores average during the window");
        }
        match self.verdict.as_str() {
            "npu-only" => {
                let _ = writeln!(s, "VERDICT: NPU-ONLY. npu drew {:.2}W ({:+.2}W vs idle); GPU idle; CPU work minimal.", self.rails["npu"].avg_w, self.rails["npu"].vs_idle_w);
            }
            _ => {
                let _ = writeln!(s, "VERDICT: NPU ACTIVE BUT NOT NPU-ONLY: {}.", self.verdict_notes.join("; "));
            }
        }
        if let (Some(n), Some(t)) = (self.tokens, self.tok_s) {
            let _ = writeln!(s, "\ntokens {n:.0} at {t:.2} tok/s");
            for (k, note) in [("npu", ""), ("soc", ""), ("system", "  (includes display and platform overhead)")] {
                if let Some(e) = self.efficiency.get(k) {
                    let _ = writeln!(s, "  {k:<7}: {:7.3} J/token  {:6.2} tok/s/W{note}", e.j_per_token, e.tok_s_per_w);
                }
            }
        }
        if let Some(b) = &self.battery {
            let _ = writeln!(s, "battery: {:.1}% -> {:.1}% of {:.1} Wh ({})", b.start_pct, b.end_pct, b.full_wh, if b.on_ac_at_end { "on AC at end" } else { "on battery" });
            if let (Some(w), Some(h)) = (b.discharge_w, b.hours_at_battery_w) {
                let _ = write!(s, "  battery-reported draw {w:.1}W -> {h:.1} h continuous decode");
                if let Some(t) = b.tokens_per_charge_at_battery_w {
                    let _ = write!(s, ", about {:.0}k tokens per charge", t / 1000.0);
                }
                let _ = writeln!(s);
            } else {
                let _ = writeln!(s, "  no battery draw (on AC); unplug for a battery-side number");
            }
            if let Some(h) = b.hours_at_system_w {
                let _ = writeln!(s, "  system-rail draw would give {h:.1} h");
            }
        }
        s
    }
}

/// A/B comparison of two reports (B relative to A).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub a: String,
    pub b: String,
    /// Percent change of B vs A; None when a side lacks the number.
    pub tok_s_pct: Option<f64>,
    pub npu_w_pct: Option<f64>,
    pub system_w_pct: Option<f64>,
    pub npu_j_per_token_pct: Option<f64>,
    pub system_j_per_token_pct: Option<f64>,
    pub npu_tok_s_per_w_pct: Option<f64>,
    pub summary: String,
}

fn pct(a: f64, b: f64) -> Option<f64> {
    (a.abs() > 1e-12).then(|| 100.0 * (b - a) / a)
}

pub fn compare(a: &Report, b: &Report) -> Comparison {
    let rail = |r: &Report, k: &str| r.rails.get(k).map(|s| s.avg_w);
    let eff = |r: &Report, k: &str| r.efficiency.get(k).cloned();
    let opt2 = |x: Option<f64>, y: Option<f64>| x.zip(y).and_then(|(x, y)| pct(x, y));
    let tok_s_pct = opt2(a.tok_s, b.tok_s);
    let npu_w_pct = opt2(rail(a, "npu"), rail(b, "npu"));
    let system_w_pct = opt2(rail(a, "system"), rail(b, "system"));
    let npu_j = opt2(eff(a, "npu").map(|e| e.j_per_token), eff(b, "npu").map(|e| e.j_per_token));
    let sys_j = opt2(eff(a, "system").map(|e| e.j_per_token), eff(b, "system").map(|e| e.j_per_token));
    let npu_tpw = opt2(eff(a, "npu").map(|e| e.tok_s_per_w), eff(b, "npu").map(|e| e.tok_s_per_w));
    let f = |x: Option<f64>| x.map(|v| format!("{v:+.1}%")).unwrap_or_else(|| "n/a".into());
    let summary = format!(
        "{} vs {}: decode {} ({} -> {} tok/s), NPU power {} ({} -> {} W), NPU J/token {}, NPU tok/s/W {}, system power {}, system J/token {}",
        b.label,
        a.label,
        f(tok_s_pct),
        a.tok_s.map(|v| format!("{v:.2}")).unwrap_or("?".into()),
        b.tok_s.map(|v| format!("{v:.2}")).unwrap_or("?".into()),
        f(npu_w_pct),
        rail(a, "npu").map(|v| format!("{v:.2}")).unwrap_or("?".into()),
        rail(b, "npu").map(|v| format!("{v:.2}")).unwrap_or("?".into()),
        f(npu_j),
        f(npu_tpw),
        f(system_w_pct),
        f(sys_j),
    );
    Comparison {
        a: a.label.clone(),
        b: b.label.clone(),
        tok_s_pct,
        npu_w_pct,
        system_w_pct,
        npu_j_per_token_pct: npu_j,
        system_j_per_token_pct: sys_j,
        npu_tok_s_per_w_pct: npu_tpw,
        summary,
    }
}

/// Pulls (tokens, tok/s) from the last llama.cpp `eval time = ... / N runs (..., X tokens per second)` line
/// (the prompt-eval line is ignored).
pub fn parse_eval(lines: &[String]) -> Option<(f64, f64)> {
    let mut found = None;
    for l in lines {
        if !l.contains("eval time") || l.contains("prompt eval") || !l.contains("tokens per second") {
            continue;
        }
        let runs = l.split(" runs").next().and_then(|s| s.rsplit(|c: char| c == '/' || c == ' ').next()).and_then(|s| s.parse::<f64>().ok());
        let rate = l.split(" tokens per second").next().and_then(|s| s.rsplit(' ').next()).and_then(|s| s.parse::<f64>().ok());
        if let (Some(n), Some(r)) = (runs, rate) {
            found = Some((n, r));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rails(pairs: &[(&str, f64)]) -> Rails {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    /// Synthetic run: 4 intervals of 0.5 s idle-ish, then 4 with npu at 4 W.
    fn synth(npu_w: f64, gpu_w: f64) -> Vec<Sample> {
        let mut out = Vec::new();
        let (mut en, mut eg, mut es, mut ec) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..9 {
            let active = i > 4;
            if i > 0 {
                en += if active { npu_w } else { 0.0 } * 0.5;
                eg += gpu_w * 0.5;
                es += if active { 10.0 } else { 5.0 } * 0.5;
                ec += 1.0 * 0.5;
            }
            out.push(Sample {
                t: i as f64 * 0.5,
                energy_j: rails(&[("npu", en), ("gpu", eg), ("soc", es), ("system", es * 2.0), ("cpu_cluster_0", ec)]),
                power_w: Rails::new(),
                cpu_s: i as f64 * 0.01,
                batt_w: None,
                batt_on_ac: None,
                batt_remaining_wh: None,
            });
        }
        out
    }

    #[test]
    fn npu_only_verdict_and_efficiency() {
        let idle = rails(&[("npu", 0.0), ("gpu", 0.02), ("soc", 5.0), ("system", 10.0), ("cpu_cluster_0", 1.0)]);
        let r = analyze(&synth(4.0, 0.02), &idle, &AnalyzeOpts { tokens: Some(100.0), tok_s: Some(20.0), ..Default::default() });
        assert_eq!(r.verdict, "npu-only");
        assert!((r.window_s - 2.0).abs() < 1e-9);
        assert!((r.rails["npu"].avg_w - 4.0).abs() < 1e-9);
        assert!((r.efficiency["npu"].j_per_token - 0.2).abs() < 1e-9);
        assert!((r.efficiency["npu"].tok_s_per_w - 5.0).abs() < 1e-9);
    }

    #[test]
    fn gpu_use_makes_it_mixed() {
        let idle = rails(&[("npu", 0.0), ("gpu", 0.02), ("system", 10.0)]);
        let r = analyze(&synth(4.0, 3.0), &idle, &AnalyzeOpts::default());
        assert_eq!(r.verdict, "npu-mixed");
    }

    #[test]
    fn idle_npu_is_inactive() {
        let idle = rails(&[("npu", 0.0)]);
        let r = analyze(&synth(0.0, 0.0), &idle, &AnalyzeOpts::default());
        assert_eq!(r.verdict, "npu-inactive");
    }

    #[test]
    fn compare_reports_relative_change() {
        let idle = rails(&[("npu", 0.0), ("gpu", 0.0), ("system", 10.0)]);
        let mk = |label: &str, npu: f64, tok: f64| {
            analyze(&synth(npu, 0.0), &idle, &AnalyzeOpts { label: label.into(), tokens: Some(100.0), tok_s: Some(tok), ..Default::default() })
        };
        let c = compare(&mk("base", 4.0, 25.0), &mk("tile16", 2.56, 23.0));
        assert!((c.tok_s_pct.unwrap() + 8.0).abs() < 1e-6);
        assert!((c.npu_w_pct.unwrap() + 36.0).abs() < 1e-6);
        assert!(c.summary.contains("tile16 vs base"));
    }

    #[test]
    fn parses_llama_eval_line() {
        let lines = vec![
            "common_perf_print: prompt eval time =       0.00 ms /     1 tokens (    0.00 ms per token,      inf tokens per second)".to_string(),
            "common_perf_print:        eval time =    8831.66 ms /   250 runs   (   35.33 ms per token,    28.31 tokens per second)".to_string(),
        ];
        assert_eq!(parse_eval(&lines), Some((250.0, 28.31)));
    }
}
