// Ported from KataGo cpp/core/datetime.cpp, cpp/core/datetime.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// SimpleDate only (sgfmetadata's gameDate). Not ported: namespace DateTime (getNow, getTimeFormat,
// timeFormat, gmTime, localTime, getDateString, getCompactDateTimeString, writeTimeToStream: wall-clock
// and local-time functions; core reads no clock, PORTING.md §12). DateTime::runTests (its SimpleDate
// checks) is replayed by tests/datetime.rs.

use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::core::global::Global;
use crate::core::global::KataError;
use crate::strprintf;
use crate::testAssert;

/// upstream: SimpleDate (datetime.h)
#[derive(Clone, Copy, Debug)]
pub struct SimpleDate {
  pub year: i32,
  pub month: i32,
  pub day: i32,
}

/// upstream: isLeapYear (datetime.cpp) static
fn is_leap_year(year: i32) -> bool {
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}
/// upstream: numDaysInMonth (datetime.cpp) static
fn num_days_in_month(year: i32, month: i32) -> i32 {
  if month == 2 {
    if is_leap_year(year) { 29 } else { 28 }
  } else if month == 4 || month == 6 || month == 9 || month == 11 {
    30
  } else {
    31
  }
}

/// upstream: isValid (datetime.cpp) static
fn is_valid(year: i32, month: i32, day: i32) -> bool {
  if month < 1 || month > 12 || day < 1 || day > num_days_in_month(year, month) {
    return false;
  }
  true
}

// Division rounding towards negative infinity, assuming d is positive but n might not be.
/// upstream: divFloor (datetime.cpp) static
fn div_floor(n: i32, d: i32) -> i32 {
  if n >= 0 { n / d } else { -1 - (-1 - n) / d }
}

/// upstream: numLeapYearsUpToAndIncluding (datetime.cpp) static
fn num_leap_years_up_to_and_including(year: i32) -> i32 {
  div_floor(year, 4) - div_floor(year, 100) + div_floor(year, 400)
}
/// upstream: CUMULATIVE_DAYS_UNTIL_MONTH, CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR (datetime.cpp) static
const CUMULATIVE_DAYS_UNTIL_MONTH: [i32; 14] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];
const CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR: [i32; 14] = [0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366];

impl Default for SimpleDate {
  /// upstream: SimpleDate::SimpleDate() (datetime.cpp)
  fn default() -> SimpleDate {
    SimpleDate::new()
  }
}

impl SimpleDate {
  /// upstream: SimpleDate::SimpleDate (datetime.cpp)
  pub fn new() -> SimpleDate {
    SimpleDate { year: 1970, month: 1, day: 1 }
  }

  /// upstream: SimpleDate::SimpleDate (datetime.cpp)
  /// PORT NOTE: throws -> Result.
  pub fn new_y_m_d(y: i32, m: i32, d: i32) -> Result<SimpleDate, KataError> {
    let s = SimpleDate { year: y, month: m, day: d };
    if !is_valid(s.year, s.month, s.day) {
      return Err(KataError::StringError(
        "SimpleDate: Invalid year month day:".to_string()
          + &Global::int_to_string(s.year)
          + ","
          + &Global::int_to_string(s.month)
          + ","
          + &Global::int_to_string(s.day),
      ));
    }
    Ok(s)
  }

  /// upstream: SimpleDate::toString (datetime.cpp)
  pub fn to_string(&self) -> String {
    strprintf!("%04d-%02d-%02d", self.year, self.month, self.day)
  }

  /// upstream: SimpleDate::SimpleDate (datetime.cpp)
  /// PORT NOTE: throws -> Result. `std::string` bytes: indexing is by byte.
  pub fn new_s(s: &str) -> Result<SimpleDate, KataError> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
      return Err(KataError::StringError("SimpleDate: Unable to parse as ISO8601 date: ".to_string() + s));
    }
    for i in 0..10 {
      if i != 4 && i != 7 {
        if b[i] < b'0' || b[i] > b'9' {
          return Err(KataError::StringError("SimpleDate: Unable to parse as ISO8601 date: ".to_string() + s));
        }
      }
    }

    // PORT NOTE: every byte checked above is an ASCII digit or '-', so these slices are on char boundaries.
    let year_str = &s[0..4];
    let month_str = &s[5..7];
    let day_str = &s[8..10];

    let year = Global::string_to_int(year_str)?;
    let month = Global::string_to_int(month_str)?;
    let day = Global::string_to_int(day_str)?;

    if !is_valid(year, month, day) {
      return Err(KataError::StringError("SimpleDate: Unable to parse as ISO8601 date: ".to_string() + s));
    }
    Ok(SimpleDate { year, month, day })
  }

  /// upstream: SimpleDate::isDuringLeapYear (datetime.cpp)
  pub fn is_during_leap_year(&self) -> bool {
    is_leap_year(self.year)
  }

  /// upstream: SimpleDate::numDaysIntoYear (datetime.cpp)
  pub fn num_days_into_year(&self) -> i32 {
    testAssert!(self.month >= 1 && self.month <= 12);
    if is_leap_year(self.year) {
      CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR[self.month as usize] + (self.day - 1)
    } else {
      CUMULATIVE_DAYS_UNTIL_MONTH[self.month as usize] + (self.day - 1)
    }
  }

  /// upstream: SimpleDate::numDaysAfter (datetime.cpp)
  pub fn num_days_after(&self, other: &SimpleDate) -> i32 {
    let mut flip = false;
    if self.year < other.year
      || (self.year == other.year && self.month < other.month)
      || (self.year == other.year && self.month == other.month && self.day < other.day)
    {
      flip = true;
    }

    let d_later: &SimpleDate = if flip { other } else { self };
    let d_early: &SimpleDate = if flip { self } else { other };

    let mut day_count: i32 = 0;
    // Start by counting days difference between Jan 01 of the respective years.
    day_count += 365 * (d_later.year - d_early.year);
    day_count += num_leap_years_up_to_and_including(d_later.year - 1) - num_leap_years_up_to_and_including(d_early.year - 1);

    testAssert!(d_early.month >= 1 && d_early.month <= 12);
    testAssert!(d_later.month >= 1 && d_later.month <= 12);

    // Now adjust for months and days
    day_count -= d_early.num_days_into_year();
    day_count += d_later.num_days_into_year();
    if flip { -day_count } else { day_count }
  }
}

impl PartialEq for SimpleDate {
  /// upstream: SimpleDate::operator==, operator!= (datetime.cpp)
  fn eq(&self, other: &SimpleDate) -> bool {
    self.year == other.year && self.month == other.month && self.day == other.day
  }
}

impl PartialOrd for SimpleDate {
  /// PORT NOTE: Rust's comparison operators go through partial_cmp / lt / le / gt / ge; the four are
  /// upstream's bodies, partial_cmp is consistent with them (a total order).
  fn partial_cmp(&self, other: &SimpleDate) -> Option<Ordering> {
    if self < other {
      Some(Ordering::Less)
    } else if self == other {
      Some(Ordering::Equal)
    } else {
      Some(Ordering::Greater)
    }
  }
  /// upstream: SimpleDate::operator< (datetime.cpp)
  fn lt(&self, other: &SimpleDate) -> bool {
    self.year < other.year
      || (self.year == other.year && self.month < other.month)
      || (self.year == other.year && self.month == other.month && self.day < other.day)
  }
  /// upstream: SimpleDate::operator<= (datetime.cpp)
  fn le(&self, other: &SimpleDate) -> bool {
    self.year < other.year
      || (self.year == other.year && self.month < other.month)
      || (self.year == other.year && self.month == other.month && self.day <= other.day)
  }
  /// upstream: SimpleDate::operator> (datetime.cpp)
  fn gt(&self, other: &SimpleDate) -> bool {
    !(self <= other)
  }
  /// upstream: SimpleDate::operator>= (datetime.cpp)
  fn ge(&self, other: &SimpleDate) -> bool {
    !(self < other)
  }
}

impl AddAssign<i32> for SimpleDate {
  /// upstream: SimpleDate::operator+= (datetime.cpp)
  fn add_assign(&mut self, mut n: i32) {
    // Normalize to Jan 1 while we deal with years.
    n += self.num_days_into_year();
    self.month = 1;
    self.day = 1;

    if n < 0 {
      // Add one, and divide by slightly too small because of leap years - we should overshoot.
      let approx_years_to_sub: i32 = 1 + (-n) / 365;
      let mut date2: SimpleDate = *self;
      date2.year -= approx_years_to_sub;
      n += self.num_days_after(&date2);
      self.year -= approx_years_to_sub;
      // Since we overshooted, this should be true.
      testAssert!(n >= 0);
    }
    while n >= 366 {
      // Divide by 366 for leap years, because of non leap years we should undershoot a bit.
      let approx_years_to_add: i32 = n / 366;
      let mut date2: SimpleDate = *self;
      date2.year += approx_years_to_add;
      n -= date2.num_days_after(self);
      self.year += approx_years_to_add;
      // Since we overshooted, this should be true.
      testAssert!(n >= 0);
    }
    if n == 365 {
      if is_leap_year(self.year) {
        self.month = 12;
        self.day = 31;
        return;
      } else {
        self.year += 1;
        // month = 1;  // already set
        // day = 1; // already set
        return;
      }
    }
    // Now we should be definitely on the right year, with n from 0 to 364 inclusive.
    testAssert!(n >= 0 && n <= 364);
    // Work out days and months.
    if is_leap_year(self.year) {
      while n >= CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR[(self.month + 1) as usize] {
        self.month += 1;
      }
      self.day = 1 + n - CUMULATIVE_DAYS_UNTIL_MONTH_LEAP_YEAR[self.month as usize];
    } else {
      while n >= CUMULATIVE_DAYS_UNTIL_MONTH[(self.month + 1) as usize] {
        self.month += 1;
      }
      self.day = 1 + n - CUMULATIVE_DAYS_UNTIL_MONTH[self.month as usize];
    }
  }
}

impl SubAssign<i32> for SimpleDate {
  /// upstream: SimpleDate::operator-= (datetime.cpp)
  fn sub_assign(&mut self, n: i32) {
    *self += -n;
  }
}

impl Add<i32> for SimpleDate {
  type Output = SimpleDate;
  /// upstream: operator+(SimpleDate, int) (datetime.cpp)
  fn add(mut self, b: i32) -> SimpleDate {
    self += b;
    self
  }
}

impl Add<SimpleDate> for i32 {
  type Output = SimpleDate;
  /// upstream: operator+(int, SimpleDate) (datetime.cpp)
  fn add(self, mut b: SimpleDate) -> SimpleDate {
    b += self;
    b
  }
}

impl Sub<i32> for SimpleDate {
  type Output = SimpleDate;
  /// upstream: operator-(SimpleDate, int) (datetime.cpp)
  fn sub(mut self, b: i32) -> SimpleDate {
    self -= b;
    self
  }
}
