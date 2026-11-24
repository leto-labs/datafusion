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
//! The automatic mode therefore backs an advised mapping when it is made, one
//! 2 MiB region at a time, and measures each region in the calling thread. A
//! region is backed by one write to its first byte: the kernel then allocates
//! the huge page, or one base page when it has none. A second write, to the
//! middle of the region, faults only in the second case. The first region
//! that the kernel backs with base pages, or that takes more than
//! [`STALL_CPU`] of thread CPU time, closes the process-wide [`StallGate`]:
//! the rest of that mapping loses its advice, and later tables use the heap
//! until one table tries again.
//!
//! An earlier version backed each region with `MADV_POPULATE_WRITE`. With
//! free huge pages, TPC-H Q21 at SF100 was then 3% slower than with the
//! advice alone (1.035 and 1.028 in one ABBA, no compaction stall, equal
//! huge pages); with the two writes it was 1.010 and 1.004, and 0.996 and
//! 0.981 in a second ABBA. The cause is not established: a probe of 24
//! threads that map, back and fill tables did not show the call slower than
//! the writes. The writes need no kernel support beyond the advice, and a
//! stalled region keeps two base pages instead of 512.

// Only the mode and the gate's types are used where there is no `madvise`.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

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

/// The first pause after a stall. A stalled probe doubles the pause, and so
/// does a stall that follows a reopening by less than the last pause, up to
/// [`MAX_PAUSE`].
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
    /// compaction in the writing thread. The default.
    FirstTouch,
    /// In their own mapping, advised and backed when it is made, whatever
    /// that costs. A diagnostic: it measures the advice without the gate.
    Populate,
    /// As [`Self::Populate`] while huge pages are free, and on the heap
    /// while they are not ([`StallGate`]). Opt-in.
    Auto,
}

impl HugePageMode {
    /// The mode of a value of [`HUGE_PAGES_ENV`]: unset, empty, `on`, `1`,
    /// `true`, `yes` or `enabled` is [`Self::FirstTouch`]; `off`, `0`,
    /// `false`, `no` or `disabled` is [`Self::Off`]; `auto` is
    /// [`Self::Auto`]; `populate` is [`Self::Populate`] (any case). Any
    /// other value is [`Self::FirstTouch`], with a warning.
    pub(crate) fn from_env_value(value: Option<&str>) -> Self {
        let Some(value) = value.map(str::trim) else {
            return Self::FirstTouch;
        };
        let is = |words: &[&str]| words.iter().any(|w| value.eq_ignore_ascii_case(w));
        if is(&["off", "0", "false", "no", "disabled"]) {
            Self::Off
        } else if value.is_empty() || is(&["on", "1", "true", "yes", "enabled"]) {
            Self::FirstTouch
        } else if is(&["populate"]) {
            Self::Populate
        } else if is(&["auto"]) {
            Self::Auto
        } else {
            log::warn!(
                "{HUGE_PAGES_ENV}={value:?} is not recognized; join tables use \
                 the default (use on, off, auto or populate)"
            );
            Self::FirstTouch
        }
    }
}

/// Whether the kernel can give an advised mapping huge pages: transparent
/// huge pages exist and are not `never` (`enabled` reads `[always]` or
/// `[madvise]`).
pub(crate) fn transparent_huge_pages_available(enabled: Option<&str>) -> bool {
    enabled.is_some_and(|mode| !mode.contains("[never]"))
}

/// Whether the kernel's transparent huge pages are 2 MiB (`hpage_pmd_size`).
/// They are 512 MiB with 64 KiB base pages; an unreadable size counts as
/// 2 MiB.
pub(crate) fn huge_pages_are_2_mib(pmd_size: Option<&str>) -> bool {
    pmd_size
        .and_then(|size| size.trim().parse::<usize>().ok())
        .is_none_or(|size| size == HUGE_PAGE)
}

/// The mode of a process whose variable asks for `mode`:
/// [`HugePageMode::Off`] where the kernel has no usable transparent huge
/// pages (`enabled`), and for the automatic mode also where they are not
/// 2 MiB (`pmd_size`): every 2 MiB region would then count as stalled.
pub(crate) fn mode_on_host(
    mode: HugePageMode,
    enabled: Option<&str>,
    pmd_size: Option<&str>,
) -> HugePageMode {
    if !transparent_huge_pages_available(enabled)
        || (mode == HugePageMode::Auto && !huge_pages_are_2_mib(pmd_size))
    {
        HugePageMode::Off
    } else {
        mode
    }
}

/// How join tables of this process ask for 2 MiB pages: the mode of
/// [`HUGE_PAGES_ENV`] on this host ([`mode_on_host`]). Read once per process.
pub fn huge_page_mode() -> HugePageMode {
    static MODE: std::sync::OnceLock<HugePageMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let read = |name: &str| {
            std::fs::read_to_string(format!("/sys/kernel/mm/transparent_hugepage/{name}"))
                .ok()
        };
        mode_on_host(
            HugePageMode::from_env_value(std::env::var(HUGE_PAGES_ENV).ok().as_deref()),
            read("enabled").as_deref(),
            read("hpage_pmd_size").as_deref(),
        )
    })
}

/// The gate of the process's join tables.
pub(crate) static STALL_GATE: StallGate = StallGate::new();

/// What backing one 2 MiB region cost the calling thread.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RegionCost {
    /// Thread CPU time, user and system.
    pub(crate) cpu: Duration,
    /// The kernel gave base pages: after the write to the region's first
    /// byte, the write to its middle faulted too.
    pub(crate) base_pages: bool,
}

impl RegionCost {
    /// Whether a region with this cost stalled: the kernel gave base pages,
    /// or the huge page took more than [`STALL_CPU`].
    pub(crate) fn stalled(&self) -> bool {
        self.base_pages || self.cpu > STALL_CPU
    }
}

#[derive(Debug)]
struct GateState {
    /// `None` while the gate is open; otherwise when a table may try again.
    retry_at: Option<Instant>,
    /// The pause of the last stall; zero before the first one.
    pause: Duration,
    /// When the last probe reopened the gate.
    reopened_at: Option<Instant>,
    /// A table is trying again and has not reported yet.
    probing: bool,
}

/// Decides whether a new join table may ask for huge pages.
///
/// Open: every table asks. A stalled region closes the gate for a pause;
/// tables made meanwhile use the heap. After the pause one table (the probe)
/// asks again: its first region reopens the gate, or closes it for twice the
/// pause. A stall therefore costs one region to each thread that is backing
/// a table at that moment, and then one region per pause while huge pages
/// stay scarce. A gate that stalls again sooner after a reopening than its
/// last pause lasted also doubles the pause, so a host that frees a few
/// huge pages now and then is asked less and less often.
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
                reopened_at: None,
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
        let pause = {
            let mut state = self.state.lock();
            let lasted = if probe {
                state.probing = false;
                false
            } else if state.retry_at.is_some() {
                // Another thread closed the gate first: one stall, one pause.
                return;
            } else {
                // The gate stayed open for at least its last pause.
                state.reopened_at.is_none_or(|reopened| {
                    now.saturating_duration_since(reopened) >= state.pause
                })
            };
            state.pause = if state.pause.is_zero() || lasted {
                FIRST_PAUSE
            } else {
                state.pause.saturating_mul(2).min(MAX_PAUSE)
            };
            state.retry_at = Some(now + state.pause);
            self.open.store(false, Ordering::Relaxed);
            state.pause
        };
        log::info!(
            "join tables stop asking for 2 MiB pages for {pause:?}: one took \
             {:?} of CPU time and got {} pages",
            cost.cpu,
            if cost.base_pages { "base" } else { "huge" }
        );
    }

    fn reopen(&self, now: Instant) {
        {
            let mut state = self.state.lock();
            state.probing = false;
            state.retry_at = None;
            state.reopened_at = Some(now);
            self.open.store(true, Ordering::Relaxed);
        }
        log::debug!("join tables ask for 2 MiB pages again");
    }

    fn release_probe(&self) {
        self.state.lock().probing = false;
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

    /// Reports the cost of a whole 2 MiB region, measured at `now`. Returns
    /// whether it stalled; the table then stops asking.
    pub(crate) fn region(&mut self, cost: RegionCost, now: Instant) -> bool {
        let first_of_probe = self.probe && !self.reported;
        let stalled = cost.stalled();
        if stalled {
            self.gate.close(first_of_probe, now, cost);
        } else if first_of_probe {
            self.gate.reopen(now);
        }
        self.reported = true;
        stalled
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
    /// Every whole 2 MiB region is backed. (A last part of less than 2 MiB
    /// gets base pages at its first writes; the kernel never compacts for
    /// it.)
    Whole,
    /// The regions from this byte on are not backed and have no advice: the
    /// gate closed, or a region stalled.
    Until(usize),
    /// The regions from this byte on are not backed and keep their advice:
    /// the kernel refused to end it.
    Refused(usize),
}

/// The calling thread's CPU time (user and system) and minor faults so far.
#[cfg(target_os = "linux")]
fn thread_usage() -> Option<(Duration, libc::c_long)> {
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
    Some((time(usage.ru_utime) + time(usage.ru_stime), usage.ru_minflt))
}

#[cfg(test)]
thread_local! {
    /// Test hook: [`back_mapping`] on this thread reports the whole region
    /// of this index as stalled and every other region as backed by a free
    /// huge page, whatever the host did.
    pub(crate) static STALL_AT_REGION: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Backs the 2 MiB region at `region` now and returns what that cost the
/// calling thread (`None` when the thread's usage cannot be read).
///
/// The write to the first byte makes the kernel allocate the region: a huge
/// page where the region is advised and one is free or can be compacted,
/// else one base page. The write to the middle then faults only if the
/// region did not get a huge page.
///
/// # Safety
///
/// `region..region + HUGE_PAGE` must be inside a private anonymous mapping
/// that the caller owns, writable and zeroed, with no reference to its
/// bytes alive.
#[cfg(target_os = "linux")]
pub(crate) unsafe fn back_region(region: *mut u8) -> Option<RegionCost> {
    let before = thread_usage();
    // SAFETY: by the caller's contract both bytes are inside its writable
    // mapping and nothing else refers to them. They are zero, so storing
    // zero changes no contents.
    let middle = unsafe {
        region.write_volatile(0);
        let middle = thread_usage();
        region.add(HUGE_PAGE / 2).write_volatile(0);
        middle
    };
    let after = thread_usage();
    let ((start, _), (_, faults), (end, faults_after)) = (before?, middle?, after?);
    Some(RegionCost {
        cpu: end.saturating_sub(start),
        base_pages: faults_after > faults,
    })
}

/// Ends the advice of `start + offset..start + bytes`, which stays unbacked.
///
/// # Safety
///
/// As [`back_mapping`], with `offset < bytes` page aligned.
#[cfg(target_os = "linux")]
unsafe fn end_advice(start: *mut u8, offset: usize, bytes: usize) -> Backed {
    // SAFETY: by the caller's contract the range is inside its mapping and
    // page aligned. The advice clears the huge-page flag of the range (and
    // splits the mapping there); it changes no contents and no permission.
    let ended = unsafe {
        libc::madvise(
            start.add(offset).cast(),
            bytes - offset,
            libc::MADV_NOHUGEPAGE,
        )
    } == 0;
    if ended {
        Backed::Until(offset)
    } else {
        Backed::Refused(offset)
    }
}

/// Backs the whole 2 MiB regions of the advised mapping
/// `start..start + bytes` now, one at a time ([`back_region`]), so that no
/// later write to them waits for a huge page. With a `ticket`, each region
/// is measured and reported to its gate; at the first stall, or once the
/// gate is closed, the remaining regions lose their advice
/// (`MADV_NOHUGEPAGE`) and stay unbacked: their first writes get base pages
/// without compaction.
///
/// # Safety
///
/// `start..start + bytes` must be a private anonymous mapping that the caller
/// owns, writable and zeroed, with `start` 2 MiB aligned, `bytes` a multiple
/// of the base page size, and no reference to its bytes alive.
#[cfg(target_os = "linux")]
pub(crate) unsafe fn back_mapping(
    start: *mut u8,
    bytes: usize,
    mut ticket: Option<Ticket<'_>>,
) -> Backed {
    let mut offset = 0;
    while offset + HUGE_PAGE <= bytes {
        if ticket.as_ref().is_some_and(|ticket| !ticket.may_ask()) {
            // SAFETY: `offset < bytes`, both page aligned, in the caller's
            // mapping.
            return unsafe { end_advice(start, offset, bytes) };
        }
        // SAFETY: `offset..offset + HUGE_PAGE` is inside the caller's
        // mapping, to which nothing else refers.
        let cost = unsafe { back_region(start.add(offset)) };
        #[cfg(test)]
        let cost = match STALL_AT_REGION.get() {
            Some(region) => Some(RegionCost {
                cpu: Duration::ZERO,
                base_pages: region == offset / HUGE_PAGE,
            }),
            None => cost,
        };
        offset += HUGE_PAGE;
        if let (Some(ticket), Some(cost)) = (ticket.as_mut(), cost)
            && ticket.region(cost, Instant::now())
        {
            if offset == bytes {
                return Backed::Whole;
            }
            // SAFETY: `offset < bytes`, both page aligned, in the caller's
            // mapping.
            return unsafe { end_advice(start, offset, bytes) };
        }
    }
    Backed::Whole
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUGE: RegionCost = RegionCost {
        cpu: Duration::from_micros(100),
        base_pages: false,
    };
    const BASE: RegionCost = RegionCost {
        cpu: Duration::from_micros(600),
        base_pages: true,
    };
    const COMPACTED: RegionCost = RegionCost {
        cpu: Duration::from_millis(30),
        base_pages: false,
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
            // Unrecognized values warn and use the default.
            (
                &["", "on", "ON", "1", "true", "yes", "enabled", "maybe"][..],
                FirstTouch,
            ),
            (&["populate", " Populate "][..], Populate),
            (&["auto", "AUTO", " auto "][..], Auto),
        ] {
            for value in values {
                assert_eq!(HugePageMode::from_env_value(Some(value)), mode, "{value:?}");
            }
        }
        assert_eq!(HugePageMode::from_env_value(None), FirstTouch);
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
        // The automatic mode needs 2 MiB huge pages; the others only THP.
        let (madvise, never) = (
            Some("always [madvise] never\n"),
            Some("always madvise [never]\n"),
        );
        assert!(huge_pages_are_2_mib(Some("2097152\n")));
        assert!(huge_pages_are_2_mib(None));
        assert!(!huge_pages_are_2_mib(Some("536870912\n")));
        for mode in [Off, FirstTouch, Populate, Auto] {
            assert_eq!(mode_on_host(mode, madvise, Some("2097152\n")), mode);
            assert_eq!(mode_on_host(mode, madvise, None), mode);
            assert_eq!(mode_on_host(mode, never, Some("2097152\n")), Off);
            assert_eq!(mode_on_host(mode, None, None), Off);
            assert_eq!(
                mode_on_host(mode, madvise, Some("536870912\n")),
                if mode == Auto { Off } else { mode }
            );
        }
    }

    #[test]
    fn a_region_stalls_on_base_pages_or_a_slow_huge_page() {
        assert!(!HUGE.stalled());
        assert!(BASE.stalled());
        assert!(COMPACTED.stalled());
        // The CPU bound is exclusive.
        let at_the_bound = RegionCost {
            cpu: STALL_CPU,
            ..HUGE
        };
        assert!(!at_the_bound.stalled());
        let over_the_bound = RegionCost {
            cpu: STALL_CPU + Duration::from_nanos(1),
            ..HUGE
        };
        assert!(over_the_bound.stalled());
    }

    #[test]
    fn a_stall_closes_the_gate_until_one_probe_reopens_it() {
        let gate = StallGate::new();
        let start = Instant::now();
        let mut first = gate.admit(start).expect("an open gate admits");
        let mut second = gate.admit(start).expect("an open gate admits");
        assert!(first.may_ask() && second.may_ask());
        assert!(!first.region(HUGE, start));
        assert!(first.region(BASE, start));
        // The other table stops at its next region; new tables use the heap.
        assert!(!gate.is_open());
        assert!(!first.may_ask() && !second.may_ask());
        assert!(gate.admit(start).is_none());
        assert!(gate.admit(start + FIRST_PAUSE / 2).is_none());
        // A second stall of the same moment does not lengthen the pause.
        assert!(second.region(COMPACTED, start + FIRST_PAUSE / 2));
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
        assert!(!probe.region(HUGE, after));
        assert!(gate.is_open() && probe.may_ask());
        let other = gate.admit(after).expect("reopened");
        assert!(other.may_ask());
    }

    #[test]
    fn stalled_probes_double_the_pause_up_to_the_longest() {
        let gate = StallGate::new();
        let mut now = Instant::now();
        assert!(gate.admit(now).unwrap().region(BASE, now));
        let mut pause = FIRST_PAUSE;
        for _ in 0..10 {
            assert!(gate.admit(now + pause - Duration::from_nanos(1)).is_none());
            now += pause;
            let mut probe = gate.admit(now).expect("the pause is over");
            assert!(probe.region(BASE, now));
            assert!(!probe.may_ask());
            pause = (pause * 2).min(MAX_PAUSE);
        }
        assert_eq!(pause, MAX_PAUSE);
    }

    /// A gate that stalls again soon after a probe reopened it doubles its
    /// pause; one that stayed open for at least its last pause starts over.
    #[test]
    fn a_stall_soon_after_a_reopening_doubles_the_pause() {
        let gate = StallGate::new();
        let mut now = Instant::now();
        let step = Duration::from_nanos(1);
        assert!(gate.admit(now).unwrap().region(BASE, now));
        let mut pause = FIRST_PAUSE;
        // Each probe gets one huge page, and the next region stalls.
        for _ in 0..3 {
            now += pause;
            let mut probe = gate.admit(now).expect("the pause is over");
            assert!(!probe.region(HUGE, now));
            assert!(gate.is_open());
            now += pause - step;
            assert!(probe.region(BASE, now));
            pause *= 2;
            assert!(gate.admit(now + pause - step).is_none());
        }
        assert_eq!(pause, 8 * FIRST_PAUSE);
        // The gate reopens and stays open for the whole last pause.
        now += pause;
        let mut probe = gate.admit(now).expect("the pause is over");
        assert!(!probe.region(HUGE, now));
        now += pause;
        assert!(gate.admit(now).unwrap().region(BASE, now));
        assert!(gate.admit(now + FIRST_PAUSE - step).is_none());
        assert!(gate.admit(now + FIRST_PAUSE).is_some());
    }

    #[test]
    fn a_probe_without_a_report_lets_the_next_table_probe() {
        let gate = StallGate::new();
        let now = Instant::now();
        assert!(gate.admit(now).unwrap().region(BASE, now));
        let after = now + FIRST_PAUSE;
        // The probe's mapping was refused: it is dropped without a region.
        drop(gate.admit(after).expect("the pause is over"));
        assert!(!gate.is_open());
        let mut probe = gate.admit(after).expect("the probe was released");
        assert!(gate.admit(after).is_none());
        assert!(!probe.region(HUGE, after));
        assert!(gate.is_open());
    }
}
