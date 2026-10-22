// Ported from KataGo cpp/search/timecontrols.cpp, cpp/search/timecontrols.h @ ba938676d7f42d70950b3a535af2466fb642008c

use std::fmt::Write as _;

use crate::core::global::KataError;
use crate::game::board::Board;
use crate::game::boardhistory::BoardHistory;
use crate::port::cpp::cpp_max;
use crate::port::cpp::cpp_min;
use crate::port::fmt::cpp_ostream_f64;
// PORT NOTE: exp goes through port::math (glibc-identical on every target, PORTING.md §6).
use crate::port::math;

/// upstream: TimeControls (timecontrols.h)
/// PORT NOTE: all fields are plain data and the C++ struct is freely copied by value
/// (`TimeControls tc; ... return tc;`), so Clone + Copy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeControls {
  /*
    Supported time controls are Fischer or generalized byo-yomi.
    Nonzero increment together with byo yomi is not supported.

    Fisher: Always in main time. After every move, increment is added.

    Byoyomi: Either in main time, or in overtime. In overtime, we have numPeriodsLeft many periods,
    each one of perPeriodTime long, and does not get used up if we play numStonesPerPeriod stones during
    that period. numPeriodsLeft
  */
  pub original_main_time: f64,
  pub increment: f64,
  pub main_time_limit: f64,
  pub max_time_per_move: f64,
  pub original_num_periods: i32,
  pub num_stones_per_period: i32,
  pub per_period_time: f64,

  pub main_time_left: f64,
  pub in_overtime: bool,
  pub num_periods_left_including_current: i32,
  pub num_stones_left_in_period: i32,
  pub time_left_in_period: f64,
}

impl TimeControls {
  //The threshold at which we consider time allowed to be unlimited
  /// upstream: TimeControls::UNLIMITED_TIME_THRESHOLD (timecontrols.h)
  pub const UNLIMITED_TIME_THRESHOLD: f64 = 1e20;
  //The max time we tolerate a user inputting
  /// upstream: TimeControls::MAX_USER_INPUT_TIME (timecontrols.h)
  pub const MAX_USER_INPUT_TIME: f64 = 1e25;
  //The value that fields default to when unset and need to be unlimited by default
  /// upstream: TimeControls::UNLIMITED_TIME_DEFAULT (timecontrols.h)
  pub const UNLIMITED_TIME_DEFAULT: f64 = 1e30;
  //The value that fields default to when unset and need to be unlimited by default and larger than other things
  /// upstream: TimeControls::UNLIMITED_TIME_DEFAULT_LARGE (timecontrols.h)
  pub const UNLIMITED_TIME_DEFAULT_LARGE: f64 = 1e40;

  //Construct a TimeControls with unlimited main time and otherwise zero initialized.
  /// upstream: TimeControls::TimeControls (timecontrols.cpp)
  pub fn new() -> TimeControls {
    TimeControls {
      original_main_time: TimeControls::UNLIMITED_TIME_DEFAULT,
      increment: 0.0,
      main_time_limit: TimeControls::UNLIMITED_TIME_DEFAULT_LARGE,
      max_time_per_move: TimeControls::UNLIMITED_TIME_DEFAULT_LARGE,
      original_num_periods: 0,
      num_stones_per_period: 0,
      per_period_time: 0.0,

      main_time_left: TimeControls::UNLIMITED_TIME_DEFAULT,
      in_overtime: false,
      num_periods_left_including_current: 0,
      num_stones_left_in_period: 0,
      time_left_in_period: 0.0,
    }
  }

  // upstream: TimeControls::~TimeControls (timecontrols.cpp)
  // PORT NOTE: empty destructor, omitted.

  /// upstream: TimeControls::isEffectivelyUnlimitedTime (timecontrols.cpp)
  pub fn is_effectively_unlimited_time(&self) -> bool {
    (self.main_time_left > TimeControls::UNLIMITED_TIME_THRESHOLD
      || (self.in_overtime && self.time_left_in_period > TimeControls::UNLIMITED_TIME_THRESHOLD))
      && self.max_time_per_move > TimeControls::UNLIMITED_TIME_THRESHOLD
  }

  /// upstream: TimeControls::absoluteTime (timecontrols.cpp)
  pub fn absolute_time(main_time: f64) -> TimeControls {
    let mut tc = TimeControls::new();
    tc.original_main_time = main_time;
    tc.increment = 0.0;
    tc.main_time_limit = TimeControls::UNLIMITED_TIME_DEFAULT_LARGE;
    tc.max_time_per_move = TimeControls::UNLIMITED_TIME_DEFAULT_LARGE;
    tc.original_num_periods = 0;
    tc.num_stones_per_period = 0;
    tc.per_period_time = 0.0;
    tc.main_time_left = main_time;
    tc.in_overtime = false;
    tc.num_periods_left_including_current = 0;
    tc.num_stones_left_in_period = 0;
    tc.time_left_in_period = 0.0;
    tc
  }

  /// upstream: TimeControls::fischerTime (timecontrols.cpp)
  pub fn fischer_time(main_time: f64, increment: f64) -> TimeControls {
    let mut tc = TimeControls::new();
    tc.original_main_time = main_time;
    tc.increment = increment;
    tc.main_time_limit = TimeControls::UNLIMITED_TIME_DEFAULT_LARGE;
    tc.max_time_per_move = TimeControls::UNLIMITED_TIME_DEFAULT_LARGE;
    tc.original_num_periods = 0;
    tc.num_stones_per_period = 0;
    tc.per_period_time = 0.0;
    tc.main_time_left = main_time;
    tc.in_overtime = false;
    tc.num_periods_left_including_current = 0;
    tc.num_stones_left_in_period = 0;
    tc.time_left_in_period = 0.0;
    tc
  }

  /// upstream: TimeControls::fischerCappedTime (timecontrols.cpp)
  pub fn fischer_capped_time(main_time: f64, increment: f64, main_time_limit: f64, max_time_per_move: f64) -> Result<TimeControls, KataError> {
    if main_time_limit < main_time {
      return Err(KataError::StringError(String::from("TimeControls: mainTimeLimit is smaller than mainTime")));
    }
    let mut tc = TimeControls::new();
    tc.original_main_time = main_time;
    tc.increment = increment;
    tc.main_time_limit = main_time_limit;
    tc.max_time_per_move = max_time_per_move;
    tc.original_num_periods = 0;
    tc.num_stones_per_period = 0;
    tc.per_period_time = 0.0;
    tc.main_time_left = main_time;
    tc.in_overtime = false;
    tc.num_periods_left_including_current = 0;
    tc.num_stones_left_in_period = 0;
    tc.time_left_in_period = 0.0;
    Ok(tc)
  }

  /// upstream: TimeControls::canadianOrByoYomiTime (timecontrols.cpp)
  pub fn canadian_or_byo_yomi_time(main_time: f64, per_period_time: f64, num_periods: i32, num_stones_per_period: i32) -> TimeControls {
    let mut tc = TimeControls::new();
    tc.original_main_time = main_time;
    tc.increment = 0.0;
    tc.main_time_limit = TimeControls::UNLIMITED_TIME_DEFAULT_LARGE;
    tc.max_time_per_move = TimeControls::UNLIMITED_TIME_DEFAULT_LARGE;
    tc.original_num_periods = num_periods;
    tc.num_stones_per_period = num_stones_per_period;
    tc.per_period_time = per_period_time;
    tc.main_time_left = main_time;
    tc.in_overtime = false;
    tc.num_periods_left_including_current = num_periods;
    tc.num_stones_left_in_period = 0;
    tc.time_left_in_period = 0.0;
    tc
  }

  /// upstream: TimeControls::toDebugString (timecontrols.cpp)
  /// PORT NOTE: overloads (PORTING.md §1 rule 3): `toDebugString()` has the fewest parameters and
  /// keeps the base name; `(board, hist, lagBuffer)` appends its parameter names. It calls getTime,
  /// which can throw, so it returns `Result`.
  pub fn to_debug_string_board_hist_lag_buffer(&self, board: &Board, hist: &BoardHistory, lag_buffer: f64) -> Result<String, KataError> {
    let mut out = String::new();
    write!(out, "originalMainTime {}", cpp_ostream_f64(self.original_main_time)).unwrap();
    if self.increment != 0.0 {
      write!(out, " increment {}", cpp_ostream_f64(self.increment)).unwrap();
    }
    if self.main_time_limit < TimeControls::UNLIMITED_TIME_THRESHOLD {
      write!(out, " mainTimeLimit {}", cpp_ostream_f64(self.main_time_limit)).unwrap();
    }
    if self.max_time_per_move < TimeControls::UNLIMITED_TIME_THRESHOLD {
      write!(out, " maxTimePerMove {}", cpp_ostream_f64(self.max_time_per_move)).unwrap();
    }

    if self.original_num_periods != 0 {
      write!(out, " originalNumPeriods {}", self.original_num_periods).unwrap();
    }
    if self.num_stones_per_period != 0 {
      write!(out, " numStonesPerPeriod {}", self.num_stones_per_period).unwrap();
    }
    if self.per_period_time != 0.0 {
      write!(out, " perPeriodTime {}", cpp_ostream_f64(self.per_period_time)).unwrap();
    }
    write!(out, " mainTimeLeft {}", cpp_ostream_f64(self.main_time_left)).unwrap();
    // `out << inOvertime`: a bool streams as 0/1
    write!(out, " inOvertime {}", self.in_overtime as i32).unwrap();
    if self.num_periods_left_including_current != 0 {
      write!(out, " numPeriodsLeftIncludingCurrent {}", self.num_periods_left_including_current).unwrap();
    }
    if self.num_stones_left_in_period != 0 {
      write!(out, " numStonesLeftInPeriod {}", self.num_stones_left_in_period).unwrap();
    }
    if self.time_left_in_period != 0.0 {
      write!(out, " timeLeftInPeriod {}", cpp_ostream_f64(self.time_left_in_period)).unwrap();
    }

    let mut min_time: f64 = 0.0;
    let mut recommended_time: f64 = 0.0;
    let mut max_time: f64 = 0.0;
    self.get_time(board, hist, lag_buffer, &mut min_time, &mut recommended_time, &mut max_time)?;
    write!(
      out,
      " minRecMax {} {} {}",
      cpp_ostream_f64(min_time),
      cpp_ostream_f64(recommended_time),
      cpp_ostream_f64(max_time)
    )
    .unwrap();

    //Rounded time limit recommendation at the start of search
    let rrec0: f64 = self.round_up_time_limit_if_needed(lag_buffer, 0.0, recommended_time);
    //Rounded time limit recommendation as we're just about to hit limit
    let rreclimit: f64 = self.round_up_time_limit_if_needed(lag_buffer, recommended_time - 0.000001, recommended_time);
    //Rounded time limit recommendation as we're just about to hit rounded limit
    let rreclimit2: f64 = self.round_up_time_limit_if_needed(lag_buffer, rreclimit - 0.000001, rreclimit);
    write!(
      out,
      " rrec0 {} rreclimit {} rreclimit2 {}",
      cpp_ostream_f64(rrec0),
      cpp_ostream_f64(rreclimit),
      cpp_ostream_f64(rreclimit2)
    )
    .unwrap();

    Ok(out)
  }

  /// upstream: TimeControls::toDebugString (timecontrols.cpp)
  /// PORT NOTE: the missing spaces between fields below are upstream's (kept verbatim).
  pub fn to_debug_string(&self) -> String {
    let mut out = String::new();
    write!(out, "originalMainTime {}", cpp_ostream_f64(self.original_main_time)).unwrap();
    if self.increment != 0.0 {
      write!(out, "increment {}", cpp_ostream_f64(self.increment)).unwrap();
    }
    if self.main_time_limit < TimeControls::UNLIMITED_TIME_THRESHOLD {
      write!(out, " mainTimeLimit {}", cpp_ostream_f64(self.main_time_limit)).unwrap();
    }
    if self.max_time_per_move < TimeControls::UNLIMITED_TIME_THRESHOLD {
      write!(out, " maxTimePerMove {}", cpp_ostream_f64(self.max_time_per_move)).unwrap();
    }
    if self.original_num_periods != 0 {
      write!(out, "originalNumPeriods {}", self.original_num_periods).unwrap();
    }
    if self.num_stones_per_period != 0 {
      write!(out, "numStonesPerPeriod {}", self.num_stones_per_period).unwrap();
    }
    if self.per_period_time != 0.0 {
      write!(out, "perPeriodTime {}", cpp_ostream_f64(self.per_period_time)).unwrap();
    }
    write!(out, "mainTimeLeft {}", cpp_ostream_f64(self.main_time_left)).unwrap();
    write!(out, "inOvertime {}", self.in_overtime as i32).unwrap();
    if self.num_periods_left_including_current != 0 {
      write!(out, "numPeriodsLeftIncludingCurrent {}", self.num_periods_left_including_current).unwrap();
    }
    if self.num_stones_left_in_period != 0 {
      write!(out, "numStonesLeftInPeriod {}", self.num_stones_left_in_period).unwrap();
    }
    if self.time_left_in_period != 0.0 {
      write!(out, "timeLeftInPeriod {}", cpp_ostream_f64(self.time_left_in_period)).unwrap();
    }
    out
  }

  //minTime - if you use less than this, you are wasting time that will not be reclaimed
  //recommendedTime - recommended mean time to search
  //maxTime - very bad to go over this time, possibly immediately losing
  /// upstream: TimeControls::getTime (timecontrols.cpp)
  pub fn get_time(
    &self,
    board: &Board,
    hist: &BoardHistory,
    lag_buffer: f64,
    min_time: &mut f64,
    recommended_time: &mut f64,
    max_time: &mut f64,
  ) -> Result<(), KataError> {
    let _ = hist;

    let board_area: i32 = board.x_size * board.y_size;
    let num_stones_on_board: i32 = board.num_stones_on_board();

    //Very crude way to estimate game progress
    let mut approx_turns_left_absolute: f64;
    let mut approx_turns_left_increment: f64; //Turns left in which we plan to spend our main time
    let mut approx_turns_left_byo_yomi: f64; //Turns left in which we plan to spend our main time
    {
      let typical_game_length_to_allow_for_absolute: f64 = 0.95 * board_area as f64 + 20.0;
      let typical_game_length_to_allow_for_increment: f64 = 0.75 * board_area as f64 + 15.0;
      let typical_game_length_to_allow_for_byo_yomi: f64 = 0.50 * board_area as f64 + 10.0;

      let min_approx_turns_left_absolute: f64 = 0.15 * board_area as f64 + 30.0;
      let min_approx_turns_left_increment: f64 = 0.10 * board_area as f64 + 20.0;
      let min_approx_turns_left_byo_yomi: f64 = 0.02 * board_area as f64 + 4.0;

      approx_turns_left_absolute = cpp_max(typical_game_length_to_allow_for_absolute - num_stones_on_board as f64, min_approx_turns_left_absolute);
      approx_turns_left_increment = cpp_max(typical_game_length_to_allow_for_increment - num_stones_on_board as f64, min_approx_turns_left_increment);
      approx_turns_left_byo_yomi = cpp_max(typical_game_length_to_allow_for_byo_yomi - num_stones_on_board as f64, min_approx_turns_left_byo_yomi);

      //Multiply by 0.5 since we only make half the moves
      approx_turns_left_absolute *= 0.5;
      approx_turns_left_increment *= 0.5;
      approx_turns_left_byo_yomi *= 0.5;
    }

    // PORT NOTE: the lambda captures the three approxTurnsLeft doubles by value and `this`. None of
    // them is written after this point and `self` is `&self`, so the closure just borrows them
    // (no aliasing, no explicit parameters needed).
    let divide_time_evenly_for_game = |time: f64, is_increment_or_abs: bool, is_byo_yomi: bool| -> f64 {
      let main_time_to_use_if_absolute: f64 = time / approx_turns_left_absolute;

      if is_increment_or_abs {
        let mut main_time_to_use: f64;
        if time <= 0.0 {
          main_time_to_use = time;
        } else {
          main_time_to_use = time / approx_turns_left_increment;
          //Make sure that if the increment is really very small, we don't choose a policy that is all that much more extreme than absolute time.
          main_time_to_use = cpp_min(main_time_to_use, main_time_to_use_if_absolute + 2.0 * self.increment);
        }
        return main_time_to_use;
      } else if is_byo_yomi {
        let mut main_time_to_use: f64;
        if self.per_period_time <= 0.0 || self.num_stones_per_period <= 0 {
          main_time_to_use = main_time_to_use_if_absolute;
        } else {
          let byo_yomi_time_per_move: f64 = self.per_period_time / self.num_stones_per_period as f64;

          //Under the assumption that we spend a fixed amount of time per move and then when we run out of main time, we use our byo yomi time, and
          //strength is proportional to log(time spent), then the optimal policy is to use e * byoYomi time per move and running out in 1/e proportion
          //of the turns that we would if we spent only the byo yomi time per move.
          let theoretical_optimal_turns_to_spend_our_time: f64 = (time / byo_yomi_time_per_move) * math::exp(-1.0);
          let mut approx_turns_left_to_use: f64 = theoretical_optimal_turns_to_spend_our_time;

          //If our desired time is longer than optimal (because in reality saving time for deep enough in the midgame is more important)
          //then attempt to stretch it out to some degree.
          if approx_turns_left_byo_yomi > theoretical_optimal_turns_to_spend_our_time {
            approx_turns_left_to_use = cpp_min(approx_turns_left_byo_yomi, theoretical_optimal_turns_to_spend_our_time * 1.75);
          }

          //If we'd be even slower than absolute time, then of course move as if absolute time.
          if approx_turns_left_to_use > approx_turns_left_absolute {
            approx_turns_left_to_use = approx_turns_left_absolute;
          }
          //Make sure that at the very end of our main time, we don't do silly things
          if approx_turns_left_to_use < 1.0 {
            approx_turns_left_to_use = 1.0;
          }

          main_time_to_use = time / approx_turns_left_to_use;
          //Make sure that if the byo yomi is really very small, we don't choose a policy that is all that much more extreme than absolute time.
          main_time_to_use = cpp_min(main_time_to_use, main_time_to_use_if_absolute + 3.0 * byo_yomi_time_per_move);
          //Make sure that we don't use less than the byo yomi time as our "basic" time. This can happen in the transition period
          //when main time left is not large
          if main_time_to_use < byo_yomi_time_per_move {
            main_time_to_use = byo_yomi_time_per_move;
          }
          //If we are using less than 1.5x the byoYomiTimePerMove and doing so would dip us into byo yomi, then go ahead and dip in.
          if main_time_to_use < byo_yomi_time_per_move * 1.5 && time < byo_yomi_time_per_move * 1.5 {
            main_time_to_use = time + byo_yomi_time_per_move;
          }
        }
        return main_time_to_use;
      }

      main_time_to_use_if_absolute
    };

    //Initialize
    *min_time = 0.0;
    *recommended_time = 0.0;
    *max_time = 0.0;

    let mut lag_buffer_to_use: f64 = lag_buffer;

    //Fischer or absolute time handling
    if self.increment > 0.0 || self.num_periods_left_including_current <= 0 {
      if self.in_overtime {
        return Err(KataError::StringError(String::from(
          "TimeControls: inOvertime with Fischer or absolute time, inconsistent time control?",
        )));
      }
      if self.num_periods_left_including_current != 0 {
        return Err(KataError::StringError(String::from(
          "TimeControls: numPeriodsLeftIncludingCurrent != 0 with Fischer or absolute time, inconsistent time control?",
        )));
      }
      if self.main_time_limit < self.original_main_time {
        return Err(KataError::StringError(String::from("TimeControls: mainTimeLimit is smaller than original mainTime")));
      }

      //Note that some GTP controllers might give us a negative mainTimeLeft in weird cases. We tolerate this and do the best we can.
      if self.main_time_left <= self.increment {
        *min_time = cpp_min(
          cpp_max(0.0, self.main_time_left * 0.5),
          cpp_max(0.0, self.main_time_left + self.increment - self.main_time_limit),
        );
        //Apply lagbuffer an extra time to the mainTimeLeft, ensuring we get extra buffering
        *recommended_time = apply_lag_buffer(self.main_time_left, lag_buffer_to_use);
        *max_time = self.main_time_left;
      } else {
        //Apply lagbuffer an extra time to the excessMainTime, ensuring we get extra buffering
        let excess_main_time: f64 = apply_lag_buffer(self.main_time_left - self.increment, lag_buffer_to_use);
        *min_time = cpp_min(
          cpp_max(0.0, self.main_time_left * 0.5),
          cpp_max(0.0, self.main_time_left + self.increment - self.main_time_limit),
        );
        *recommended_time = self.increment + divide_time_evenly_for_game(excess_main_time, true, false);
        *max_time = cpp_min(self.main_time_left, self.increment + excess_main_time / 5.0);
      }
    }
    //Byo yomi or canadian time handling
    else {
      if self.main_time_limit < TimeControls::UNLIMITED_TIME_THRESHOLD {
        return Err(KataError::StringError(String::from(
          "TimeControls: mainTimeLimit is used with byo-yomiish periods, inconsistent time control?",
        )));
      }
      if self.num_stones_per_period <= 0 {
        return Err(KataError::StringError(String::from(
          "TimeControls: numStonesPerPeriod <= 0 with byo-yomiish periods, inconsistent time control?",
        )));
      }
      if !self.in_overtime && self.num_periods_left_including_current != self.original_num_periods {
        return Err(KataError::StringError(String::from(
          "TimeControls: not in overtime, but numPeriodsLeftIncludingCurrent != originalNumPeriods",
        )));
      }
      if self.in_overtime && self.num_stones_left_in_period < 1 {
        return Err(KataError::StringError(String::from(
          "TimeControls: numStonesLeftInPeriod < 1 while in overtime, inconsistent time control?",
        )));
      }

      let mut effective_main_time_left: f64 = self.main_time_left;
      let mut effectively_in_overtime: bool = self.in_overtime;
      let mut effective_num_periods_left_including_current: i32 = self.num_periods_left_including_current;
      let mut effective_time_left_in_period: f64 = self.time_left_in_period;
      let mut effective_num_stones_left_in_period: i32 = self.num_stones_left_in_period;

      //If somehow main time left is negative, then assume we've moved into byo yomi by the appropriate amount
      if effective_main_time_left < 0.0 && !effectively_in_overtime {
        effectively_in_overtime = true;
        effective_time_left_in_period = effective_main_time_left + self.per_period_time;
        effective_num_stones_left_in_period = self.num_stones_per_period;
      }
      //Similarly handle it if byo yomi time left is negative, including if main time negative overflowed into byo yomi negative
      if effectively_in_overtime {
        while effective_time_left_in_period < 0.0 && effective_num_periods_left_including_current > 1 {
          effective_num_periods_left_including_current -= 1;
          effective_time_left_in_period += self.per_period_time;
        }
      }

      //Crudely treat all but the last 5 periods as main time.
      const NUM_RESERVED_PERIODS: i32 = 5;
      if effective_num_periods_left_including_current > NUM_RESERVED_PERIODS {
        effectively_in_overtime = false;
        if !self.in_overtime {
          effective_main_time_left += self.per_period_time * (effective_num_periods_left_including_current - NUM_RESERVED_PERIODS) as f64;
        } else {
          effective_main_time_left += effective_time_left_in_period
            + self.per_period_time * (effective_num_periods_left_including_current - NUM_RESERVED_PERIODS - 1) as f64;
        }
      }

      if !effectively_in_overtime {
        //The upper limit of what we'll tolerate for spending on a move in byo yomi
        let large_byo_yomi_time_per_move: f64 = self.per_period_time / (0.75 * self.num_stones_per_period as f64 + 0.25);

        *min_time = 0.0;
        *recommended_time = divide_time_evenly_for_game(effective_main_time_left, false, true);
        *max_time = large_byo_yomi_time_per_move
          + cpp_max(cpp_min(large_byo_yomi_time_per_move * 1.75, effective_main_time_left), effective_main_time_left / 5.0);

        //If we're going into byo yomi, we might as well allow using the whole period
        if *max_time > effective_main_time_left && *max_time < effective_main_time_left + large_byo_yomi_time_per_move {
          *max_time = effective_main_time_left + large_byo_yomi_time_per_move;
        }

        //Increase the lagbuffer a little if upon entering byo yomi we're actually on the last byo yomi (i.e. running out actually kills us)
        if *max_time > effective_main_time_left && effective_num_periods_left_including_current <= 1 && self.num_stones_per_period <= 1 {
          lag_buffer_to_use *= 2.0;
        }
      } else {
        if effective_num_stones_left_in_period < 1 {
          return Err(KataError::StringError(String::from(
            "TimeControls: effectiveNumStonesLeftInPeriod < 1 while in overtime, inconsistent time control?",
          )));
        }

        //If we're somehow lagging or reconnected so that we ended up very far in the period, and we have some periods left, then
        //go ahead and use a period so that we get more thinking time.
        if effective_num_periods_left_including_current > 1 &&
          //TODO this should take into account previous-turn thinking time, if we have enough we should be willing to insta-move to
          //save a period
            apply_lag_buffer(effective_time_left_in_period, lag_buffer_to_use)
              < apply_lag_buffer(0.5 * self.per_period_time, lag_buffer_to_use) * (effective_num_periods_left_including_current - 1) as f64
                / (NUM_RESERVED_PERIODS - 1) as f64
        {
          effective_num_periods_left_including_current -= 1;
          effective_time_left_in_period += self.per_period_time;
        }

        *min_time = if effective_num_stones_left_in_period <= 1 { effective_time_left_in_period } else { 0.0 };
        *recommended_time = effective_time_left_in_period / effective_num_stones_left_in_period as f64;
        *max_time = effective_time_left_in_period / (0.75 * effective_num_stones_left_in_period as f64 + 0.25);

        //Increase the lagbuffer a little if we're actually on the last stone of the last byo yomi (i.e. running out actually kills us)
        if effective_num_periods_left_including_current <= 1 && effective_num_stones_left_in_period <= 1 {
          lag_buffer_to_use *= 2.0;
        }
      }
    }

    *max_time = cpp_min(*max_time, self.max_time_per_move);

    //Lag buffer
    *min_time = apply_lag_buffer(*min_time, lag_buffer_to_use);
    *recommended_time = apply_lag_buffer(*recommended_time, lag_buffer_to_use);
    *max_time = apply_lag_buffer(*max_time, lag_buffer_to_use);

    //Just in case
    if *max_time < 0.0 {
      *max_time = 0.0;
    }
    if *min_time < 0.0 {
      *min_time = 0.0;
    }
    if *recommended_time < 0.0 {
      *recommended_time = 0.0;
    }
    if *min_time > *max_time {
      *min_time = *max_time;
    }
    if *recommended_time > *max_time {
      *recommended_time = *max_time;
    }
    Ok(())
  }

  //If we'd think for a given time limit and actually it would lose time to stop at this limit, then bump the limit up
  //This is used for not partial-wasting byo yomi periods.
  /// upstream: TimeControls::roundUpTimeLimitIfNeeded (timecontrols.cpp)
  pub fn round_up_time_limit_if_needed(&self, lag_buffer: f64, time_used: f64, time_limit: f64) -> f64 {
    let mut time_limit = time_limit;
    if self.increment > 0.0 || self.num_periods_left_including_current <= 0 {
      return time_limit;
    }

    let mut effective_main_time_left: f64 = self.main_time_left;
    let mut effectively_in_overtime: bool = self.in_overtime;
    let mut effective_num_periods_left_including_current: i32 = self.num_periods_left_including_current;
    let mut effective_time_left_in_period: f64 = self.time_left_in_period;
    let mut effective_num_stones_left_in_period: f64 = self.num_stones_left_in_period as f64;

    //Scroll up to where we are based on time used
    if !effectively_in_overtime {
      effective_main_time_left -= time_used;
    } else {
      effective_time_left_in_period -= time_used;
    }

    //Roll from main time into overtime
    if effective_main_time_left < 0.0 && !effectively_in_overtime {
      effectively_in_overtime = true;
      effective_time_left_in_period = effective_main_time_left + self.per_period_time;
      effective_num_stones_left_in_period = self.num_stones_per_period as f64;
    }

    //Roll through any ends of periods
    if effectively_in_overtime {
      while effective_time_left_in_period < 0.0 && effective_num_periods_left_including_current > 1 {
        effective_num_periods_left_including_current -= 1;
        effective_time_left_in_period += self.per_period_time;
      }
    }

    #[allow(unused_assignments)] // upstream initializes; every path below assigns or returns
    let mut rounded_up_time_usage: f64 = time_used;
    let byo_yomi_time_per_move: f64 = self.per_period_time / self.num_stones_per_period as f64;
    let byo_yomi_time_per_move_buffered: f64 = apply_lag_buffer(self.per_period_time / self.num_stones_per_period as f64, lag_buffer);

    //Basically like lagbuffer, but bounded away from zero and capped at byoYomiTimePerMoveBuffered
    let bit_of_time: f64 = cpp_min(cpp_max(lag_buffer, byo_yomi_time_per_move_buffered * 0.01), byo_yomi_time_per_move_buffered);

    //Still in main time
    if !effectively_in_overtime {
      //If we have very little main time left, then we might as well use it all up
      if effective_main_time_left < byo_yomi_time_per_move * 0.5 {
        //Japanese - use it up, plus the whole period, so we don't waste it.
        if self.num_stones_per_period <= 1 {
          rounded_up_time_usage = time_used + effective_main_time_left + byo_yomi_time_per_move_buffered;
        }
        //Canadian - use it up, plus at least make sure we get a bit into our overtime period
        //We might reevaluate once we actually get in to overtime.
        else {
          rounded_up_time_usage = time_used + effective_main_time_left + bit_of_time;
        }
      } else {
        return time_limit;
      }
    }
    //Overtime
    else {
      //We probably lost on time! Just keep the limit the same and do what we would have done without rounding
      if effective_time_left_in_period <= 0.0 {
        return time_limit;
      }
      //If we have multiple stones left, then make sure we use at least a little fraction of our per-move time of the period
      //if we entered into main time this turn, so we don't lose time by accidentally submitting our move before finishing
      //our main time! We want to make sure to count 1 stone played in the new period.
      if effective_num_stones_left_in_period > 1.0 {
        //So, if we were not in overtime at the start of this move, but we used only a tiny bit of time in the overtime...
        if !self.in_overtime && (self.per_period_time - effective_time_left_in_period) < bit_of_time {
          rounded_up_time_usage = time_used + bit_of_time - (self.per_period_time - effective_time_left_in_period);
        }
        //If we have multiple stones left, there's no other situation where we want to artifically spend more time, we won't lose any.
        else {
          return time_limit;
        }
      }
      //If we have one stone left, time would in fact be wasted, so then we do want to round up.
      else {
        rounded_up_time_usage = apply_lag_buffer(time_used + effective_time_left_in_period, lag_buffer);
      }
    }

    if rounded_up_time_usage < time_used {
      return time_limit;
    }

    if time_limit < rounded_up_time_usage {
      time_limit = rounded_up_time_usage;
    }
    time_limit
  }
}

/// upstream: applyLagBuffer (timecontrols.cpp) static
fn apply_lag_buffer(time: f64, lag_buffer: f64) -> f64 {
  if time < 0.0 {
    time
  } else if time < 2.0 * lag_buffer {
    time * 0.5
  } else {
    time - lag_buffer
  }
}
