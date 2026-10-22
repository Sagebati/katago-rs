// Ported from KataGo cpp/program/playutils.cpp @ ba938676d7f42d70950b3a535af2466fb642008c
//
// M3 subset (port-manifest.toml): only PlayUtils::getHackedLCBForWinrate, which
// search/searchresults.cpp calls. Everything else in playutils.cpp/.h is not ported yet.

use crate::game::board::Player;
use crate::game::board::P_WHITE;
use crate::search::analysisdata::AnalysisData;
use crate::search::search::Search;

/// upstream: namespace PlayUtils (playutils.h)
pub struct PlayUtils;

impl PlayUtils {
  /// upstream: PlayUtils::getHackedLCBForWinrate (playutils.cpp)
  pub fn get_hacked_lcb_for_winrate(search: &Search, data: &AnalysisData, pla: Player) -> f64 {
    let winrate: f64 = 0.5 * (1.0 + data.win_loss_value);
    //Super hacky - in KataGo, lcb is on utility (i.e. also weighting score), not winrate, but if you're using
    //lz-analyze you probably don't know about utility and expect LCB to be about winrate. So we apply the LCB
    //radius to the winrate in order to get something reasonable to display, and also scale it proportionally
    //by how much winrate is supposed to matter relative to score.
    let mut radius_scale_hack_factor: f64 = search.search_params.win_loss_utility_factor
      / (search.search_params.win_loss_utility_factor
        + search.search_params.static_score_utility_factor
        + search.search_params.dynamic_score_utility_factor
        + 1.0e-20); //avoid divide by 0
    //Also another factor of 0.5 because winrate goes from only 0 to 1 instead of -1 to 1 when it's part of utility
    radius_scale_hack_factor *= 0.5;
    let lcb: f64 = if pla == P_WHITE {
      winrate - data.radius * radius_scale_hack_factor
    } else {
      winrate + data.radius * radius_scale_hack_factor
    };
    lcb
  }
}
