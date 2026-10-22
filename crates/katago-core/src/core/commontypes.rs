// Ported from KataGo cpp/core/commontypes.h @ ba938676d7f42d70950b3a535af2466fb642008c

/// upstream: enabled_t (commontypes.h)
/// PORT NOTE: the struct wrapping `enum value { False, True, Auto }` in a field `x` (so that
/// `enabled_t::True` converts implicitly and `operator bool` is deleted) becomes one Rust enum
/// with the same variant names; `==`/`!=` are the derived `PartialEq`. The defaulted ctor leaves
/// `x` uninitialized upstream, so there is no `Default`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum enabled_t {
  False,
  True,
  Auto,
}

impl enabled_t {
  /// upstream: enabled_t::toString (commontypes.h)
  pub fn to_string(&self) -> String {
    let x = *self;
    (if x == enabled_t::True {
      "true"
    } else if x == enabled_t::False {
      "false"
    } else {
      "auto"
    })
    .to_string()
  }

  /// upstream: enabled_t::tryParse (commontypes.h)
  pub fn try_parse(v: &str, buf: &mut enabled_t) -> bool {
    if v == "1" || v == "t" || v == "true" || v == "enabled" || v == "y" || v == "yes" {
      *buf = enabled_t::True;
    } else if v == "0" || v == "f" || v == "false" || v == "disabled" || v == "n" || v == "no" {
      *buf = enabled_t::False;
    } else if v == "auto" {
      *buf = enabled_t::Auto;
    } else {
      return false;
    }
    return true;
  }
}

#[cfg(test)]
mod tests {
  use super::enabled_t;

  #[test]
  fn parse_and_print() {
    let mut b = enabled_t::Auto;
    for (s, want) in [("1", enabled_t::True), ("yes", enabled_t::True), ("disabled", enabled_t::False), ("auto", enabled_t::Auto)] {
      assert!(enabled_t::try_parse(s, &mut b));
      assert_eq!(b, want);
    }
    assert!(!enabled_t::try_parse("TRUE", &mut b));
    assert_eq!(b, enabled_t::Auto);
    assert_eq!(enabled_t::True.to_string(), "true");
    assert_eq!(enabled_t::False.to_string(), "false");
    assert_eq!(enabled_t::Auto.to_string(), "auto");
  }
}
