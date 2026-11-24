// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Longbow patch (JT 10): when a join table asks the kernel for 2 MiB pages.
//!
//! Written by Leto Labs for Longbow; see `LONGBOW.md` (family JT).
//!
//! An advised mapping gets a huge page at the first write to each 2 MiB
//! region. When the kernel has no free huge page, that write compacts memory
//! in the writing thread (THP `defrag` `madvise` or `always`), and most of
//! those compactions fail. On worker-01 a region that the kernel backs from a
//! free huge page costs one thread about 0.1 ms, and 1.5 ms (4.4 ms at most)
//! when 24 threads of one NUMA node back regions together: zeroing 2 MiB is
//! bound by memory bandwidth. A region that compacts costs 2 to 5 ms at the
//! median and up to 340 ms, and nine of ten still get base pages.
//!
//! The default mode therefore backs an advised mapping when it is made, one
//! 2 MiB region at a time (`MADV_POPULATE_WRITE`), and measures each region
//! in the calling thread. The first region that the kernel backs with base
//! pages, or that takes more than [`STALL_CPU`] of thread CPU time, closes
//! the process-wide [`StallGate`]: the rest of that mapping loses its advice,
//! and later tables use the heap until one table tries again.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// The environment variable that chooses the [`HugePageMode`]. Read once per
/// process.
pub const HUGE_PAGES_ENV: &str = "LONGBOW_JOIN_TABLE_HUGE_PAGES";

/// The size of the pages that an advised mapping asks for.
pub(crate) const HUGE_PAGE: usize = 2 << 20;

/// A whole 2 MiB region that takes more thread CPU time than this to back
/// has stalled, even when it gets a huge page: the kernel compacted for it.
/// Zeroing a free huge page takes 0.1 ms alone and up to 4.4 ms when 24
/// threads of one node do it together (worker-01); the bound leaves room
/// for a host with a quarter of that memory bandwidth per thread.
const STALL_CPU: Duration = Duration::from_millis(16);

/// The first pause after a stall; each stall without a wholly backed table
/// in between doubles the pause, up to [`MAX_PAUSE`].
const FIRST_PAUSE: Duration = Duration::from_secs(1);

/// The longest pause between two attempts.
const MAX_PAUSE: Duration = Duration::from_secs(64);

/// How join-table arrays from 4 MiB on are backed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageMode {
    /// On the heap, without advice.
    Off,
    /// In their own mapping, advised before the first write (JT 6). Each
    /// 2 MiB region is allocated by its first write, with the host's
    /// compaction in the writing thread.
    FirstTouch,
    /// In their own mapping, advised and backed when it is made, whatever
    /// that costs. A diagnostic: it measures the advice without the gate.
    Populate,
    /// As [`Self::Populate`] while huge pages are free, and on the heap
    /// while they are not ([`StallGate`]). The default.
    Auto,
}

impl HugePageMode {
    /// The mode of a value of [`HUGE_PAGES_ENV`]: unset, empty or `auto` is
    /// [`Self::Auto`]; `on`, `1`, `true`, `yes` or `enabled` is
    /// [`Self::FirstTouch`]; `off`, `0`, `false`, `no` or `disabled` is
    /// [`Self::Off`]; `populate` is [`Self::Populate`] (any case). Any other
    /// value is [`Self::Auto`], with a warning.
    pub(crate) fn from_env_value(value: Option<&str>) -> Self {
        let Some(value) = value.map(str::trim) else {
            return Self::Auto;
        };
        let is = |words: &[&str]| words.iter().any(|w| value.eq_ignore_ascii_case(w));
        if is(&["off", "0", "false", "no", "disabled"]) {
            Self::Off
        } else if is(&["on", "1", "true", "yes", "enabled"]) {
            Self::FirstTouch
        } else if is(&["populate"]) {
            Self::Populate
        } else if value.is_empty() || is(&["auto"]) {
            Self::Auto
        } else {
            log::warn!(
                "{HUGE_PAGES_ENV}={value:?} is not recognized; join tables use \
                 the default (use auto, on, off or populate)"
            );
            Self::Auto
        }
    }
}

/// Whether the kernel can give an advised mapping huge pages: transparent
/// huge pages exist and are not `never` (`enabled` reads `[always]` or
/// `[madvise]`).
pub(crate) fn transparent_huge_pages_available(enabled: Option<&str>) -> bool {
    enabled.is_some_and(|mode| !mode.contains("[never]"))
}

/// How join tables of this process ask for 2 MiB pages: the mode of
/// [`HUGE_PAGES_ENV`], or [`HugePageMode::Off`] where the kernel has no
/// usable transparent huge pages. Read once per process.
pub fn huge_page_mode() -> HugePageMode {
    static MODE: std::sync::OnceLock<HugePageMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode =
            HugePageMode::from_env_value(std::env::var(HUGE_PAGES_ENV).ok().as_deref());
        let available = transparent_huge_pages_available(
            std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
                .ok()
                .as_deref(),
        );
        if available { mode } else { HugePageMode::Off }
    })
}

/// The gate of the process's join tables.
pub(crate) static STALL_GATE: StallGate = StallGate::new();

/// What backing one 2 MiB region cost the calling thread.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RegionCost {
    /// Thread CPU time, user and system.
    pub(crate) cpu: Duration,
    /// Minor faults of the thread: one for a huge page, one per base page
    /// otherwise.
    pub(crate) faults: u64,
}

impl RegionCost {
    /// Whether a whole 2 MiB region with this cost stalled: the kernel gave
    /// base pages (at least half of the region's `base_pages` faulted one by
    /// one), or the huge page took more than [`STALL_CPU`].
    pub(crate) fn stalled(&self, base_pages: usize) -> bool {
        self.faults.saturating_mul(2) >= base_pages as u64 || self.cpu > STALL_CPU
    }
}

#[derive(Debug)]
struct GateState {
    /// `None` while the gate is open; otherwise when a table may try again.
    retry_at: Option<Instant>,
    /// The pause of the last stall; zero once a table was wholly backed.
    pause: Duration,
    /// A table is trying again and has not reported yet.
    probing: bool,
}

/// Decides whether a new join table may ask for huge pages.
///
/// Open: every table asks. A stalled region closes the gate for a pause;
/// tables made meanwhile use the heap. After the pause one table (the probe)
/// asks again: its first region reopens the gate, or closes it for twice the
/// pause. A stall therefore costs each thread that is backing a table at
/// that moment one region, and then one region per pause.
#[derive(Debug)]
pub(crate) struct StallGate {
    state: Mutex<GateState>,
    /// Mirrors `retry_at.is_none()`, for the check before each region.
    open: AtomicBool,
}

impl StallGate {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(GateState {
                retry_at: None,
                pause: Duration::ZERO,
                probing: false,
            }),
            open: AtomicBool::new(true),
        }
    }

    /// A ticket for a table made at `now`, or `None` while the gate is
    /// closed: the table then uses the heap.
    pub(crate) fn admit(&self, now: Instant) -> Option<Ticket<'_>> {
        let mut state = self.state.lock();
        let probe = match state.retry_at {
            None => false,
            Some(retry_at) if now < retry_at || state.probing => return None,
            Some(_) => {
                state.probing = true;
                true
            }
        };
        Some(Ticket {
            gate: self,
            probe,
            reported: false,
        })
    }

    /// Whether tables may ask for huge pages now.
    pub(crate) fn is_open(&self) -> bool {
        self.open.load(Ordering::Relaxed)
    }

    fn close(&self, probe: bool, now: Instant, cost: RegionCost) {
        let mut state = self.state.lock();
        if probe {
            state.probing = false;
        } else if state.retry_at.is_some() {
            // Another thread closed the gate first: one stall, one pause.
            return;
        }
        state.pause = if state.pause.is_zero() {
            FIRST_PAUSE
        } else {
            state.pause.saturating_mul(2).min(MAX_PAUSE)
        };
        state.retry_at = Some(now + state.pause);
        self.open.store(false, Ordering::Relaxed);
        log::info!(
            "join tables stop asking for 2 MiB pages for {:?}: backing one took \
             {:?} of CPU time and {} page faults",
            state.pause,
            cost.cpu,
            cost.faults
        );
    }

    fn reopen(&self) {
        let mut state = self.state.lock();
        state.probing = false;
        state.retry_at = None;
        self.open.store(true, Ordering::Relaxed);
        log::debug!("join tables ask for 2 MiB pages again");
    }

    fn release_probe(&self) {
        self.state.lock().probing = false;
    }

    fn wholly_backed(&self) {
        let mut state = self.state.lock();
        if state.retry_at.is_none() {
            state.pause = Duration::ZERO;
        }
    }
}

/// The right of one table to ask for huge pages, from [`StallGate::admit`].
/// A probe that is dropped without a report lets the next table probe.
#[derive(Debug)]
pub(crate) struct Ticket<'a> {
    gate: &'a StallGate,
    probe: bool,
    reported: bool,
}

impl Ticket<'_> {
    /// Whether the table may ask for its next region: the gate is open, or
    /// this table is the probe and has not reported yet.
    pub(crate) fn may_ask(&self) -> bool {
        (self.probe && !self.reported) || self.gate.is_open()
    }

    /// Reports the cost of a whole 2 MiB region of `base_pages` base pages,
    /// measured at `now`. Returns whether it stalled; the table then stops
    /// asking.
    pub(crate) fn region(
        &mut self,
        cost: RegionCost,
        base_pages: usize,
        now: Instant,
    ) -> bool {
        let first_of_probe = self.probe && !self.reported;
        let stalled = cost.stalled(base_pages);
        if stalled {
            self.gate.close(first_of_probe, now, cost);
        } else if first_of_probe {
            self.gate.reopen();
        }
        self.reported = true;
        stalled
    }

    /// Reports that every region of the table was backed without a stall.
    pub(crate) fn wholly_backed(self) {
        self.gate.wholly_backed();
    }
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        if self.probe && !self.reported {
            self.gate.release_probe();
        }
    }
}

/// How [`back_mapping`] left a mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Backed {
    /// Every page is resident.
    Whole,
    /// The pages from this byte on are not resident and have no advice: the
    /// gate closed, or a region stalled.
    Until(usize),
    /// The kernel refused to back pages (it has no `MADV_POPULATE_WRITE`
    /// before Linux 5.14, or it is out of memory); the pages from this byte
    /// on keep their advice and are allocated by their first write.
    Refused(usize),
}

/// The calling thread's CPU time and minor faults so far.
#[cfg(target_os = "linux")]
fn thread_cost() -> Option<RegionCost> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` writes one `rusage` to the given pointer, which is
    // valid for it.
    if unsafe { libc::getrusage(libc::RUSAGE_THREAD, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the call succeeded, so it initialized the structure (which is
    // also valid when zeroed).
    let usage = unsafe { usage.assume_init() };
    let time = |value: libc::timeval| {
        Duration::new(
            u64::try_from(value.tv_sec).unwrap_or(0),
            u32::try_from(value.tv_usec)
                .unwrap_or(0)
                .saturating_mul(1000),
        )
    };
    Some(RegionCost {
        cpu: time(usage.ru_utime) + time(usage.ru_stime),
        faults: u64::try_from(usage.ru_minflt).unwrap_or(0),
    })
}

#[cfg(test)]
thread_local! {
    /// Test hook: [`back_mapping`] on this thread reports the whole region
    /// of this index as stalled and every other region as backed by a free
    /// huge page, whatever the host did.
    pub(crate) static STALL_AT_REGION: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Set once the kernel refuses `MADV_POPULATE_WRITE` as unknown.
#[cfg(target_os = "linux")]
static POPULATE_UNSUPPORTED: AtomicBool = AtomicBool::new(false);

/// Backs the advised mapping `start..start + bytes` now, one 2 MiB region at
/// a time, so that no later write to it allocates a page. With a `ticket`,
/// each whole region is measured and reported to its gate; at the first
/// stall, or once the gate is closed, the remaining regions lose their
/// advice (`MADV_NOHUGEPAGE`) and stay unbacked: their first writes get base
/// pages without compaction.
///
/// # Safety
///
/// `start..start + bytes` must be a private anonymous mapping that the caller
/// owns, with `start` 2 MiB aligned and `bytes` a multiple of `page`, the
/// base page size.
#[cfg(target_os = "linux")]
pub(crate) unsafe fn back_mapping(
    start: *mut u8,
    bytes: usize,
    page: usize,
    mut ticket: Option<Ticket<'_>>,
) -> Backed {
    if POPULATE_UNSUPPORTED.load(Ordering::Relaxed) {
        return Backed::Refused(0);
    }
    let mut offset = 0;
    while offset < bytes {
        let len = HUGE_PAGE.min(bytes - offset);
        // SAFETY: by the caller's contract `offset..bytes` is inside its
        // mapping and page aligned. Neither advice changes the contents or
        // the permissions of the pages: one backs them, the other clears the
        // huge-page flag of the range.
        let (refused, cost) = unsafe {
            if ticket.as_ref().is_some_and(|ticket| !ticket.may_ask()) {
                libc::madvise(
                    start.add(offset).cast(),
                    bytes - offset,
                    libc::MADV_NOHUGEPAGE,
                );
                return Backed::Until(offset);
            }
            let before = thread_cost();
            let refused =
                (libc::madvise(start.add(offset).cast(), len, libc::MADV_POPULATE_WRITE)
                    != 0)
                    .then(std::io::Error::last_os_error);
            let cost = before.zip(thread_cost()).map(|(before, after)| RegionCost {
                cpu: after.cpu.saturating_sub(before.cpu),
                faults: after.faults.saturating_sub(before.faults),
            });
            (refused, cost)
        };
        if let Some(error) = refused {
            if error.raw_os_error() == Some(libc::EINVAL) {
                POPULATE_UNSUPPORTED.store(true, Ordering::Relaxed);
            }
            return Backed::Refused(offset);
        }
        #[cfg(test)]
        let cost = match STALL_AT_REGION.get() {
            Some(region) => Some(RegionCost {
                cpu: Duration::ZERO,
                faults: if region == offset / HUGE_PAGE {
                    u64::MAX
                } else {
                    1
                },
            }),
            None => cost,
        };
        offset += len;
        if len == HUGE_PAGE
            && let (Some(ticket), Some(cost)) = (ticket.as_mut(), cost)
            && ticket.region(cost, HUGE_PAGE / page, Instant::now())
        {
            if offset < bytes {
                // SAFETY: as above; `offset..bytes` is the rest of the
                // mapping.
                unsafe {
                    libc::madvise(
                        start.add(offset).cast(),
                        bytes - offset,
                        libc::MADV_NOHUGEPAGE,
                    );
                }
            }
            return Backed::Until(offset);
        }
    }
    if let Some(ticket) = ticket {
        ticket.wholly_backed();
    }
    Backed::Whole
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUGE: RegionCost = RegionCost {
        cpu: Duration::from_micros(100),
        faults: 1,
    };
    const BASE: RegionCost = RegionCost {
        cpu: Duration::from_micros(600),
        faults: 512,
    };
    const COMPACTED: RegionCost = RegionCost {
        cpu: Duration::from_millis(30),
        faults: 1,
    };

    #[test]
    fn the_switch_names_four_modes() {
        use HugePageMode::*;
        for (values, mode) in [
            (
                &[
                    "off", "0", "false", " off ", "OFF", "Off", "no", "NO", "disabled",
                ][..],
                Off,
            ),
            (&["on", "ON", "1", "true", "yes", "enabled"][..], FirstTouch),
            (&["populate", " Populate "][..], Populate),
            // Unrecognized values warn and use the default.
            (&["", "auto", "AUTO", "maybe"][..], Auto),
        ] {
            for value in values {
                assert_eq!(HugePageMode::from_env_value(Some(value)), mode, "{value:?}");
            }
        }
        assert_eq!(HugePageMode::from_env_value(None), Auto);
        assert!(transparent_huge_pages_available(Some(
            "always [madvise] never\n"
        )));
        assert!(transparent_huge_pages_available(Some(
            "[always] madvise never\n"
        )));
        assert!(!transparent_huge_pages_available(Some(
            "always madvise [never]\n"
        )));
        assert!(!transparent_huge_pages_available(None));
        // The process's mode: the variable's, or off without usable THP.
        let thp =
            std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled").ok();
        let expected = if transparent_huge_pages_available(thp.as_deref()) {
            HugePageMode::from_env_value(std::env::var(HUGE_PAGES_ENV).ok().as_deref())
        } else {
            Off
        };
        assert_eq!(huge_page_mode(), expected);
    }

    #[test]
    fn a_region_stalls_on_base_pages_or_a_slow_huge_page() {
        assert!(!HUGE.stalled(512));
        assert!(BASE.stalled(512));
        assert!(COMPACTED.stalled(512));
        // Half of the base pages faulting one by one is a fallback.
        let half = RegionCost {
            faults: 256,
            ..HUGE
        };
        assert!(half.stalled(512));
        assert!(
            !RegionCost {
                faults: 255,
                ..HUGE
            }
            .stalled(512)
        );
        // 64 KiB base pages: 32 to a region.
        assert!(RegionCost { faults: 32, ..HUGE }.stalled(32));
        // The CPU bound is exclusive.
        assert!(
            !RegionCost {
                cpu: STALL_CPU,
                ..HUGE
            }
            .stalled(512)
        );
    }

    #[test]
    fn a_stall_closes_the_gate_until_one_probe_reopens_it() {
        let gate = StallGate::new();
        let start = Instant::now();
        let mut first = gate.admit(start).expect("an open gate admits");
        let mut second = gate.admit(start).expect("an open gate admits");
        assert!(first.may_ask() && second.may_ask());
        assert!(!first.region(HUGE, 512, start));
        assert!(first.region(BASE, 512, start));
        // The other table stops at its next region; new tables use the heap.
        assert!(!gate.is_open());
        assert!(!first.may_ask() && !second.may_ask());
        assert!(gate.admit(start).is_none());
        assert!(gate.admit(start + FIRST_PAUSE / 2).is_none());
        // A second stall of the same moment does not lengthen the pause.
        assert!(second.region(COMPACTED, 512, start + FIRST_PAUSE / 2));
        assert!(
            gate.admit(start + FIRST_PAUSE - Duration::from_nanos(1))
                .is_none()
        );

        // After the pause exactly one table probes.
        let after = start + FIRST_PAUSE;
        let mut probe = gate.admit(after).expect("the pause is over");
        assert!(probe.may_ask());
        assert!(gate.admit(after).is_none());
        assert!(!gate.is_open());
        // Its first region reopens the gate for every table.
        assert!(!probe.region(HUGE, 512, after));
        assert!(gate.is_open() && probe.may_ask());
        let other = gate.admit(after).expect("reopened");
        assert!(other.may_ask());
    }

    #[test]
    fn stalled_probes_double_the_pause_up_to_the_longest() {
        let gate = StallGate::new();
        let mut now = Instant::now();
        assert!(gate.admit(now).unwrap().region(BASE, 512, now));
        let mut pause = FIRST_PAUSE;
        for _ in 0..10 {
            assert!(gate.admit(now + pause - Duration::from_nanos(1)).is_none());
            now += pause;
            let mut probe = gate.admit(now).expect("the pause is over");
            assert!(probe.region(BASE, 512, now));
            assert!(!probe.may_ask());
            pause = (pause * 2).min(MAX_PAUSE);
        }
        assert_eq!(pause, MAX_PAUSE);

        // A probe whose first region is huge reopens the gate, but the
        // pause stays long until a table is wholly backed.
        now += pause;
        let mut probe = gate.admit(now).expect("the pause is over");
        assert!(!probe.region(HUGE, 512, now));
        assert!(probe.region(BASE, 512, now));
        assert!(
            gate.admit(now + MAX_PAUSE - Duration::from_nanos(1))
                .is_none()
        );
        now += MAX_PAUSE;
        let mut probe = gate.admit(now).expect("the pause is over");
        assert!(!probe.region(HUGE, 512, now));
        probe.wholly_backed();
        assert!(gate.admit(now).unwrap().region(BASE, 512, now));
        assert!(
            gate.admit(now + FIRST_PAUSE - Duration::from_nanos(1))
                .is_none()
        );
        assert!(gate.admit(now + FIRST_PAUSE).is_some());
    }

    #[test]
    fn a_probe_without_a_report_lets_the_next_table_probe() {
        let gate = StallGate::new();
        let now = Instant::now();
        assert!(gate.admit(now).unwrap().region(BASE, 512, now));
        let after = now + FIRST_PAUSE;
        // The probe's mapping was refused: it is dropped without a region.
        drop(gate.admit(after).expect("the pause is over"));
        assert!(!gate.is_open());
        let mut probe = gate.admit(after).expect("the probe was released");
        assert!(gate.admit(after).is_none());
        assert!(!probe.region(HUGE, 512, after));
        assert!(gate.is_open());
        // A table that was wholly backed while the gate is closed by another
        // stall does not shorten the pause.
        let mut open = gate.admit(after).unwrap();
        let mut late = gate.admit(after).unwrap();
        assert!(open.region(BASE, 512, after));
        assert!(!late.region(HUGE, 512, after));
        late.wholly_backed();
        assert!(gate.admit(after + FIRST_PAUSE).is_none());
        assert!(gate.admit(after + 2 * FIRST_PAUSE).is_some());
    }
}
