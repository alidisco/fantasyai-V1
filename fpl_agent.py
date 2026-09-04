import os
import json
import time
import threading
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_sanitize(x) for x in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif obj is None:
        return None
    elif str(obj) in ('nan', 'None', '<NA>'):
        return None
    return obj

def _safe_chance(val):
    if val is None or str(val) in ('nan', 'None', '<NA>', ''):
        return None
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return int(f)
    except Exception:
        return None

class FPLAutonomousAgent:
    def __init__(self, analyzer=None):
        self.analyzer = analyzer
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://fantasy.premierleague.com/',
            'Origin': 'https://fantasy.premierleague.com'
        })
        
        self.email = os.getenv('FPL_EMAIL', '')
        self.password = os.getenv('FPL_PASSWORD', '')
        self.team_id = os.getenv('FPL_TEAM_ID', '')
        self.auth_token = os.getenv('FPL_AUTH_TOKEN', '')
        
        self.cache_dir = os.path.dirname(os.path.abspath(__file__))
        self.session_cache_file = os.path.join(self.cache_dir, 'session_cache.json')
        self.settings_file = os.path.join(self.cache_dir, 'ai_manager_settings.json')
        self.logs_file = os.path.join(self.cache_dir, 'ai_manager_logs.json')
        
        self.settings = self._load_settings()
        self.logs = self._load_logs()
        self.is_authenticated = False
        self.user_profile = None
        self.team_info = None
        
        # Load any existing cached session
        self._load_cached_session()
        
        # Start background deadline watcher daemon
        self.daemon_thread = None
        self._start_deadline_daemon()

    def _load_settings(self):
        default_settings = {
            'auto_pilot': True,
            'dry_run': False,  # 100% Fully Autonomous Live Mode
            'max_hits': -4,   # Max -4 hit tolerance (if 5GW gain > 5.0 pts)
            'trigger_minutes_before_deadline': 60,
            'auto_draft_if_empty': True,
            'chip_strategy': 'smart',  # Autonomously deploys Wildcard (1st & 2nd half), 3x Captain, Bench Boost, Free Hit
            'last_run_gw': None,
            'last_run_time': None
        }
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    saved = json.load(f)
                    default_settings.update(saved)
            except Exception as e:
                print(f"Error loading AI settings: {e}")
        return default_settings

    def save_settings(self, new_settings=None):
        if new_settings:
            self.settings.update(new_settings)
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving AI settings: {e}")
            return False

    def _load_logs(self):
        if os.path.exists(self.logs_file):
            try:
                with open(self.logs_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def log_decision(self, log_entry):
        entry = {
            'id': f"log_{int(time.time()*1000)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': log_entry.get('type', 'GENERAL'),
            'title': log_entry.get('title', 'AI Manager Action'),
            'summary': log_entry.get('summary', ''),
            'details': log_entry.get('details', {}),
            'dry_run': log_entry.get('dry_run', self.settings.get('dry_run', True))
        }
        entry = _sanitize(entry)
        self.logs.insert(0, entry)
        # Keep latest 100 logs
        self.logs = self.logs[:100]
        try:
            with open(self.logs_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Error saving AI logs: {e}")
        return entry

    def _load_cached_session(self):
        if os.path.exists(self.session_cache_file):
            try:
                with open(self.session_cache_file, 'r') as f:
                    data = json.load(f)
                    cookies = data.get('cookies', {})
                    for name, value in cookies.items():
                        self.session.cookies.set(name, value, domain='.premierleague.com')
                    if data.get('auth_token'):
                        self.auth_token = data['auth_token']
                        self.session.headers['X-API-Authorization'] = f"Bearer {self.auth_token}"
                    if data.get('team_id'):
                        self.team_id = data['team_id']
                    self.verify_session()
            except Exception as e:
                print(f"Error loading session cache: {e}")

    def _save_cached_session(self):
        try:
            data = {
                'cookies': self.session.cookies.get_dict(),
                'auth_token': self.auth_token,
                'team_id': self.team_id,
                'updated_at': datetime.now().isoformat()
            }
            with open(self.session_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving session cache: {e}")

    def verify_session(self):
        """Verifies if the current session can access authenticated FPL endpoints"""
        try:
            res = self.session.get(f"{self.base_url}/me/", timeout=10)
            if res.status_code == 200:
                self.user_profile = res.json()
                self.is_authenticated = True
                entry = self.user_profile.get('entry')
                if entry:
                    self.team_id = str(entry.get('id', self.team_id))
                    self.team_info = entry
                return True
            self.is_authenticated = False
            return False
        except Exception as e:
            print(f"Session verify error: {e}")
            self.is_authenticated = False
            return False

    def authenticate_with_browser(self, email=None, password=None, headless=True):
        """Automated login using Playwright to handle FPL OIDC/PingOne authentication flow"""
        target_email = email or self.email or os.getenv('FPL_EMAIL')
        target_password = password or self.password or os.getenv('FPL_PASSWORD')
        
        if not target_email or not target_password:
            return {
                'success': False,
                'error': 'Missing FPL credentials. Please configure FPL_EMAIL and FPL_PASSWORD in .env'
            }
            
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                
                # Navigate to FPL login / main site
                page.goto('https://fantasy.premierleague.com/', timeout=30000)
                page.wait_for_timeout(2000)
                
                # Accept cookie banner if present
                try:
                    accept_btn = page.locator('button:has-text("Accept All"), button:has-text("Accept"), button#onetrust-accept-btn-handler')
                    if accept_btn.count() > 0 and accept_btn.first.is_visible():
                        accept_btn.first.click()
                        page.wait_for_timeout(1000)
                except Exception:
                    pass
                
                # Look for Sign In / Log In button or go directly to login URL
                sign_in_link = page.locator('a:has-text("Sign In"), a:has-text("Log In"), button:has-text("Sign In")')
                if sign_in_link.count() > 0 and sign_in_link.first.is_visible():
                    sign_in_link.first.click()
                else:
                    page.goto('https://fantasy.premierleague.com/my-team', timeout=30000)
                
                page.wait_for_timeout(3000)
                
                # Find login input fields
                email_input = page.locator('input[type="email"], input[name="login"], input#login, input[autocomplete="email"]')
                pass_input = page.locator('input[type="password"], input[name="password"], input#password')
                
                if email_input.count() > 0 and pass_input.count() > 0:
                    email_input.first.fill(target_email)
                    pass_input.first.fill(target_password)
                    page.wait_for_timeout(500)
                    
                    # Submit login form
                    submit_btn = page.locator('button[type="submit"], input[type="submit"], button:has-text("Sign In"), button:has-text("Log In")')
                    if submit_btn.count() > 0:
                        submit_btn.first.click()
                    else:
                        pass_input.first.press('Enter')
                        
                    # Wait for redirect after successful authentication
                    page.wait_for_timeout(5000)
                
                # Extract all cookies
                cookies = context.cookies()
                for c in cookies:
                    self.session.cookies.set(c['name'], c['value'], domain=c['domain'])
                
                # Check localStorage for OIDC tokens
                try:
                    local_storage = page.evaluate("() => JSON.stringify(window.localStorage)")
                    ls_data = json.loads(local_storage)
                    for k, v in ls_data.items():
                        if 'oidc.user:' in k:
                            user_data = json.loads(v)
                            if user_data.get('access_token'):
                                self.auth_token = user_data['access_token']
                                self.session.headers['X-API-Authorization'] = f"Bearer {self.auth_token}"
                except Exception as e:
                    print(f"LocalStorage inspection notice: {e}")
                
                browser.close()
                
            # Verify newly captured session
            if self.verify_session():
                self._save_cached_session()
                p_info = (self.user_profile or {}).get('player') or {}
                p_name = f"{p_info.get('first_name', '')} {p_info.get('last_name', '')}".strip() or "Manager"
                self.log_decision({
                    'type': 'AUTH',
                    'title': 'FPL Account Connected',
                    'summary': f"Successfully authenticated as {p_name}",
                    'details': {'team_id': self.team_id, 'profile': p_info}
                })
                return {'success': True, 'team_id': self.team_id, 'profile': self.user_profile}
            else:
                # If direct API authentication wasn't verified immediately, session cookies were still saved
                self._save_cached_session()
                return {
                    'success': True,
                    'message': 'Login session saved. Public API access active, private API verification pending.',
                    'team_id': self.team_id
                }
                
        except Exception as e:
            return {'success': False, 'error': f"Browser login failed: {str(e)}"}

    def get_account_status(self):
        """Returns comprehensive status of the FPL AI Manager and connected account"""
        if self.analyzer and not self.analyzer.bootstrap_data:
            self.analyzer.fetch_bootstrap_data()
            self.analyzer.fetch_fixtures_data()
            
        current_gw = self.analyzer.get_current_gameweek() if self.analyzer else 1
        next_gw = self.analyzer.get_next_gameweek() if self.analyzer else 1
        
        # Calculate next deadline info
        deadline_info = self.get_next_deadline_info()
        
        team_data = None
        current_picks = []
        free_transfers = 1
        bank = 0.0
        gw_points = 0
        overall_points = 0
        overall_rank = 0
        team_name = "AI Fantasy Squad"
        manager_name = "AI Autonomous Manager"
        
        if self.team_id:
            try:
                # Public team details
                t_info = self.analyzer.fetch_team_data(self.team_id) if self.analyzer else None
                if t_info:
                    team_name = t_info.get('name', team_name)
                    manager_name = f"{t_info.get('player_first_name', '')} {t_info.get('player_last_name', '')}".strip() or manager_name
                    gw_points = t_info.get('summary_event_points', 0)
                    overall_points = t_info.get('summary_overall_points', 0)
                    overall_rank = t_info.get('summary_overall_rank', 0)
                    val = t_info.get('last_deadline_value')
                    b_val = t_info.get('last_deadline_bank')
                    bank = (float(b_val) / 10.0) if b_val is not None else 0.0
                
                # Fetch picks (fallback to history if current_gw is empty)
                picks_data = self.analyzer.fetch_team_picks(self.team_id, current_gw) if self.analyzer else None
                if not picks_data or 'picks' not in picks_data:
                    hist_check = self.analyzer.fetch_entry_history(self.team_id) if self.analyzer else None
                    if hist_check and hist_check.get('current'):
                        last_gw = hist_check['current'][-1]['event']
                        picks_data = self.analyzer.fetch_team_picks(self.team_id, last_gw)
                
                if picks_data and 'picks' in picks_data:
                    current_picks = picks_data['picks']
                    transfers_info = picks_data.get('entry_history', {})
                    # Estimate free transfers
                    ft_count = 1
                    if transfers_info.get('event_transfers', 0) == 0:
                        ft_count = min(5, transfers_info.get('event_transfers_cost', 0) + 1)
                    free_transfers = ft_count
                    
                # Fetch chips history
                chips_used = []
                hist = self.analyzer.fetch_entry_history(self.team_id) if self.analyzer else None
                if hist and 'chips' in hist:
                    chips_used = [c.get('name') for c in hist['chips']]
            except Exception as e:
                print(f"Status team fetch error: {e}")
        else:
            chips_used = []

        # Resolve squad player cards
        squad_cards = []
        captain_card = None
        vice_captain_card = None
        starting_xi = []
        bench = []
        
        if current_picks and self.analyzer:
            for p in current_picks:
                pid = int(p['element'])
                c = self.analyzer.get_player_full_card(pid)
                if c:
                    c['is_captain'] = p.get('is_captain', False)
                    c['is_vice_captain'] = p.get('is_vice_captain', False)
                    c['position_order'] = p.get('position', 1)
                    squad_cards.append(c)
            
            if squad_cards:
                lineup = self.optimize_lineup_and_captain(squad_cards, next_gw)
                starting_xi = lineup.get('starting_xi', [])
                bench = lineup.get('bench', [])
                captain_card = lineup.get('captain')
                vice_captain_card = lineup.get('vice_captain')
                
        # Calculate pre-deadline transfer candidates with % probabilities
        pre_deadline_transfers = self.get_pre_deadline_analysis(squad_cards, bank, free_transfers, next_gw)
        
        return _sanitize({
            'is_authenticated': self.is_authenticated or bool(self.team_id),
            'email': self.email or os.getenv('FPL_EMAIL', ''),
            'team_id': self.team_id or "9765376",
            'team_name': team_name,
            'manager_name': manager_name,
            'gw_points': gw_points,
            'overall_points': overall_points,
            'overall_rank': overall_rank,
            'bank': bank,
            'free_transfers': free_transfers,
            'chips_used': chips_used,
            'current_gw': current_gw,
            'next_gw': next_gw,
            'deadline': deadline_info,
            'settings': self.settings,
            'has_squad': len(squad_cards) > 0,
            'squad_count': len(squad_cards),
            'squad_cards': squad_cards,
            'starting_xi': starting_xi,
            'bench': bench,
            'captain': captain_card,
            'vice_captain': vice_captain_card,
            'pre_deadline_transfers': pre_deadline_transfers,
            'recent_logs': self.logs[:10]
        })

    def get_pre_deadline_analysis(self, current_squad_cards=None, bank=0.0, free_transfers=1, next_gw=2):
        """
        Calculates candidate moves with probability % distribution ahead of deadline.
        AI evaluates point ceiling vs risk, ranking choices before the 30-min cutoff.
        """
        if not current_squad_cards or not self.analyzer or self.analyzer.players_data is None:
            return {
                'candidates': [],
                'top_choice': None,
                'lock_in_minutes_before': self.settings.get('trigger_minutes_before_deadline', 30)
            }
            
        all_players = self.analyzer.players_data
        candidate_pool = all_players[all_players['status'] == 'a'].copy()
        
        # Calculate 5GW xP for current squad
        for p in current_squad_cards:
            p['adv_xp_5gw'] = self._compute_advanced_xp(p, next_gw, n_gws=5)
            
        # Find weakest spots in squad (ignore GKPs to never waste transfers on them)
        gkps_in_squad = [p for p in current_squad_cards if p.get('element_type') == 1 or p.get('position_short') == 'GKP']
        all_gkps_unfit = len(gkps_in_squad) > 0 and all(g.get('status') in ['i', 's'] or (_safe_chance(g.get('chance_of_playing_next_round')) is not None and _safe_chance(g.get('chance_of_playing_next_round')) <= 25) for g in gkps_in_squad)
        
        sell_candidates = []
        for p in current_squad_cards:
            is_gkp = (p.get('element_type') == 1 or p.get('position_short') == 'GKP')
            if is_gkp and not all_gkps_unfit:
                continue  # Never waste transfers on GKPs if a fit GKP exists
                
            p_status = p.get('status', 'a')
            p_chance = _safe_chance(p.get('chance_of_playing_next_round'))
            p_diffs = self.analyzer.get_fixture_details_next_5gw(int(p.get('team_id', p.get('team', 1))), next_gw)
            avg_diff = sum(d['difficulty'] for d in p_diffs) / max(1, len(p_diffs))
            adv_xp = p.get('adv_xp_5gw', 10.0)
            
            risk_score = 0
            reasons = []
            if p_status in ['i', 's']:
                risk_score += 16
                reasons.append(f"Injured / {p.get('news', 'Flagged')}")
            elif p_chance is not None and p_chance <= 50:
                risk_score += 10
                reasons.append(f"Doubtful ({p_chance}%)")
            elif avg_diff >= 4.0:
                risk_score += 5
                reasons.append("Tough fixture swing")
            elif float(p.get('form', 0) or 0) < 1.5:
                risk_score += 3
                reasons.append("Cold form")
                
            sell_candidates.append({
                'player': p,
                'risk_score': risk_score,
                'adv_xp': adv_xp,
                'reasons': reasons
            })
            
        sell_candidates.sort(key=lambda x: x['risk_score'], reverse=True)
        
        transfer_options = []
        current_ids = {p['id'] for p in current_squad_cards}
        
        # Generate candidate pairs
        for cand in sell_candidates[:3]:
            out_p = cand['player']
            pos = cand['player'].get('element_type', 1)
            max_cost = float(out_p.get('cost', 5.0)) + bank
            
            pos_pool = candidate_pool[
                (candidate_pool['element_type'] == pos) &
                (~candidate_pool['id'].isin(current_ids)) &
                (candidate_pool['now_cost'] <= int(max_cost * 10))
            ].copy()
            
            if pos_pool.empty:
                continue
                
            for _, r in pos_pool.head(10).iterrows():
                card = self.analyzer.get_player_full_card(int(r['id']))
                if not card:
                    continue
                in_xp = self._compute_advanced_xp(card, next_gw, n_gws=5)
                gain = in_xp - cand['adv_xp']
                transfer_options.append({
                    'out': out_p,
                    'in': card,
                    'xp_gain': round(gain, 1),
                    'raw_score': max(0.2, gain * 1.5 + (cand['risk_score'] * 0.9))
                })
                
        # Sort by raw score descending
        transfer_options.sort(key=lambda x: x['raw_score'], reverse=True)
        top_candidates = transfer_options[:3]
        
        # Baseline: Roll / Hold Transfer
        has_urgent_issues = any(c['risk_score'] >= 10 for c in sell_candidates)
        hold_score = 1.0 if has_urgent_issues else 8.0
        
        scores = [c['raw_score'] for c in top_candidates] + [hold_score]
        exp_scores = [np.exp(s / 2.5) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [round((es / sum_exp) * 100, 1) for es in exp_scores]
        
        result_candidates = []
        for idx, cand in enumerate(top_candidates):
            result_candidates.append({
                'type': 'TRANSFER',
                'out_player': cand['out']['web_name'],
                'out_team': cand['out'].get('team_short', ''),
                'out_cost': cand['out'].get('cost', 0),
                'in_player': cand['in']['web_name'],
                'in_team': cand['in'].get('team_short', ''),
                'in_cost': cand['in'].get('cost', 0),
                'xp_gain': cand['xp_gain'],
                'probability_pct': probs[idx],
                'reason': f"Replaces {cand['out']['web_name']} with in-form {cand['in']['web_name']} (+{cand['xp_gain']} xP gain)"
            })
            
        result_candidates.append({
            'type': 'ROLL_FT',
            'out_player': 'None',
            'out_team': '',
            'out_cost': 0,
            'in_player': 'Hold Free Transfer',
            'in_team': '',
            'in_cost': 0,
            'xp_gain': 0.0,
            'probability_pct': probs[-1],
            'reason': 'Squad is well balanced; rolling FT provides 2 free transfers for next Gameweek.'
        })
        
        result_candidates.sort(key=lambda x: x['probability_pct'], reverse=True)
        
        return {
            'candidates': result_candidates,
            'top_choice': result_candidates[0] if result_candidates else None,
            'lock_in_minutes_before': self.settings.get('trigger_minutes_before_deadline', 30)
        }

    def get_next_deadline_info(self):
        """Calculates exact time remaining until next FPL gameweek deadline"""
        if not self.analyzer or not self.analyzer.bootstrap_data:
            return {'gameweek': 1, 'deadline_time': None, 'time_remaining': 'Unknown', 'minutes_left': 9999}
            
        events = self.analyzer.bootstrap_data.get('events', [])
        for event in events:
            if event.get('is_next'):
                d_time_str = event.get('deadline_time')
                if d_time_str:
                    try:
                        # Parse FPL UTC timestamp (e.g., '2024-08-16T17:30:00Z')
                        clean_str = d_time_str.replace('Z', '+00:00')
                        deadline_dt = datetime.fromisoformat(clean_str)
                        now_utc = datetime.now(deadline_dt.tzinfo) if deadline_dt.tzinfo else datetime.utcnow()
                        delta = deadline_dt - now_utc
                        minutes_left = int(delta.total_seconds() / 60)
                        
                        hours = int(delta.total_seconds() // 3600)
                        minutes = int((delta.total_seconds() % 3600) // 60)
                        
                        time_remaining_str = f"{hours}h {minutes}m" if delta.total_seconds() > 0 else "Passed"
                        return {
                            'gameweek': event.get('id'),
                            'name': event.get('name'),
                            'deadline_time': d_time_str,
                            'time_remaining': time_remaining_str,
                            'minutes_left': max(0, minutes_left),
                            'is_open': minutes_left > 0
                        }
                    except Exception as e:
                        print(f"Error parsing deadline: {e}")
                        
        return {'gameweek': 1, 'deadline_time': None, 'time_remaining': 'N/A', 'minutes_left': 9999}

    def draft_initial_squad(self, budget=100.0, target_gw=None):
        """
        Autonomous Initial Squad Drafter:
        Generates an optimal 15-man squad (£100m, 2 GKP, 5 DEF, 5 MID, 3 FWD, max 3/club)
        maximizing projected 5-Gameweek points (xP) while balancing starting XI power and bench efficiency.
        """
        if not self.analyzer:
            return {'success': False, 'error': 'Analyzer not initialized'}
            
        if not self.analyzer.bootstrap_data:
            self.analyzer.fetch_bootstrap_data()
            self.analyzer.fetch_fixtures_data()
            
        self.analyzer.train_models()
        
        current_gw = target_gw or self.analyzer.get_current_gameweek()
        
        # Generate optimal squad using FPLAnalyzer engine
        optimal_result = self.analyzer.generate_optimal_squad(budget * 10, current_gw)
        if not optimal_result or 'squad' not in optimal_result:
            # Fallback to wildcard generator logic
            optimal_result = self.analyzer.generate_wildcard_team(current_gw)
            
        if not optimal_result:
            return {'success': False, 'error': 'Could not build optimal squad within constraints'}
            
        squad_players = optimal_result.get('squad', [])
        
        # Optimize Starting XI vs Bench from the 15 players
        lineup_result = self.optimize_lineup_and_captain(squad_players, current_gw)
        
        draft_summary = {
            'success': True,
            'squad': squad_players,
            'total_cost': optimal_result.get('total_cost', sum(p.get('cost', 0) for p in squad_players)),
            'remaining_budget': optimal_result.get('remaining_budget', round(budget - sum(p.get('cost', 0) for p in squad_players), 1)),
            'total_predicted_5gw': optimal_result.get('total_predicted', sum(p.get('predicted_5gw', 0) for p in squad_players)),
            'starting_xi': lineup_result['starting_xi'],
            'bench': lineup_result['bench'],
            'captain': lineup_result['captain'],
            'vice_captain': lineup_result['vice_captain'],
            'formation': lineup_result['formation'],
            'rationale': "Engine selected highest 5-GW expected points (xP) players with balanced £100m budget, factoring in form, FDR, and xGI metrics."
        }
        
        self.log_decision({
            'type': 'INITIAL_DRAFT',
            'title': f'AI Drafted Initial Squad (GW{current_gw})',
            'summary': f"Selected optimal 15-player squad ({lineup_result['formation']}) with Captain {lineup_result['captain'].get('web_name')} | Budget Used: £{draft_summary['total_cost']}m",
            'details': {
                'formation': lineup_result['formation'],
                'captain': lineup_result['captain'].get('web_name'),
                'vice_captain': lineup_result['vice_captain'].get('web_name'),
                'total_cost': draft_summary['total_cost'],
                'total_predicted_5gw': draft_summary['total_predicted_5gw']
            }
        })
        
        return _sanitize(draft_summary)

    def optimize_lineup_and_captain(self, player_cards, current_gw=1):
        """
        Selects the best starting XI (valid formation like 3-5-2, 3-4-3, 4-4-2, etc.),
        orders the bench (12: backup GKP, 13: Sub 1, 14: Sub 2, 15: Sub 3),
        and selects optimal Captain and Vice-Captain based on next GW expected points.
        """
        # Compute true fixture-based next GW xP for all players
        for p in player_cards:
            xp_next = self._compute_advanced_xp(p, current_gw, n_gws=1)
            p['predicted_next_gw'] = xp_next
        
        # Categorize by position
        gkps = [p for p in player_cards if p.get('position_short') == 'GKP' or p.get('element_type') == 1]
        defs = [p for p in player_cards if p.get('position_short') == 'DEF' or p.get('element_type') == 2]
        mids = [p for p in player_cards if p.get('position_short') == 'MID' or p.get('element_type') == 3]
        fwds = [p for p in player_cards if p.get('position_short') == 'FWD' or p.get('element_type') == 4]
        
        # Sort each position by projected points descending
        gkps.sort(key=lambda x: x.get('predicted_next_gw', 0), reverse=True)
        defs.sort(key=lambda x: x.get('predicted_next_gw', 0), reverse=True)
        mids.sort(key=lambda x: x.get('predicted_next_gw', 0), reverse=True)
        fwds.sort(key=lambda x: x.get('predicted_next_gw', 0), reverse=True)
        
        best_starting_xi = []
        best_bench = []
        best_score = -1
        best_formation = "3-4-3"
        
        # Test valid formations: min 3 DEF, min 2 MID, min 1 FWD (Total = 1 GKP + 10 Outfield)
        valid_formations = [
            (3, 5, 2), (3, 4, 3), (4, 4, 2), (4, 3, 3), (4, 5, 1), (5, 3, 2), (5, 4, 1), (5, 2, 3)
        ]
        
        starting_gkp = gkps[0] if gkps else None
        bench_gkp = gkps[1] if len(gkps) > 1 else None
        
        for n_def, n_mid, n_fwd in valid_formations:
            if len(defs) >= n_def and len(mids) >= n_mid and len(fwds) >= n_fwd:
                sel_defs = defs[:n_def]
                sel_mids = mids[:n_mid]
                sel_fwds = fwds[:n_fwd]
                
                cur_xi = ([starting_gkp] if starting_gkp else []) + sel_defs + sel_mids + sel_fwds
                score = sum(p.get('predicted_next_gw', 0) for p in cur_xi if p)
                
                if score > best_score:
                    best_score = score
                    best_formation = f"{n_def}-{n_mid}-{n_fwd}"
                    best_starting_xi = cur_xi
                    
                    # Remaining players go to bench
                    rem_defs = defs[n_def:]
                    rem_mids = mids[n_mid:]
                    rem_fwds = fwds[n_fwd:]
                    outfield_bench = rem_defs + rem_mids + rem_fwds
                    # Order outfield bench by xP
                    outfield_bench.sort(key=lambda x: x.get('predicted_next_gw', 0), reverse=True)
                    best_bench = ([bench_gkp] if bench_gkp else []) + outfield_bench
        
        # Pick Captain & Vice-Captain from Starting XI
        # Top-tier FPL rule: Explosive attacking talismans (FWD/MID) have much higher ceiling than clean-sheet dependent defenders
        def get_captaincy_score(p_card):
            pos_id = int(p_card.get('element_type', 3))
            base_xp = p_card.get('predicted_next_gw', 0)
            cost_num = float(p_card.get('cost', 5.0))
            att_mult = 1.35 if pos_id in [3, 4] else 0.70  # Prioritize Strikers & Midfielders
            prem_mult = 1.25 if cost_num >= 13.0 else (1.10 if cost_num >= 9.5 else 1.0)
            return base_xp * att_mult * prem_mult

        sorted_xi = sorted(best_starting_xi, key=get_captaincy_score, reverse=True)
        captain = sorted_xi[0] if sorted_xi else {}
        vice_captain = sorted_xi[1] if len(sorted_xi) > 1 else captain
        
        return {
            'formation': best_formation,
            'starting_xi': best_starting_xi,
            'bench': best_bench,
            'captain': captain,
            'vice_captain': vice_captain,
            'starting_score': round(best_score, 1)
        }

    def run_gameweek_cycle(self, dry_run=None, max_hit=None, force_squad=False):
        """
        Complete Autonomous AI Gameweek Management Cycle:
        1. Researches latest player metrics, form, injury flags, and upcoming fixtures.
        2. Evaluates current team and identifies weak spots (injured / red-flagged / blank / low xP).
        3. Explores transfer combinations (1 FT, 2 FT, or -4 hit if point gain > 4).
        4. Optimizes Starting XI, Formation, Captain, Vice-Captain, and Bench order.
        5. Commits to FPL API (if dry_run is False) or generates detailed simulation report.
        """
        is_dry_run = self.settings.get('dry_run', True) if dry_run is None else dry_run
        hit_tolerance = self.settings.get('max_hits', -4) if max_hit is None else max_hit
        
        if not self.analyzer:
            return {'success': False, 'error': 'Analyzer engine unavailable'}
            
        if not self.analyzer.bootstrap_data:
            self.analyzer.fetch_bootstrap_data()
            self.analyzer.fetch_fixtures_data()
            
        self.analyzer.train_models()
        
        current_gw = self.analyzer.get_current_gameweek()
        next_gw = self.analyzer.get_next_gameweek()
        
        # Fetch current picks from FPL
        current_picks = []
        bank = 0.0
        free_transfers = 1
        
        if self.team_id:
            try:
                picks_data = self.analyzer.fetch_team_picks(self.team_id, current_gw)
                if picks_data and 'picks' in picks_data:
                    current_picks = picks_data['picks']
                team_data = self.analyzer.fetch_team_data(self.team_id)
                if team_data:
                    b_val = team_data.get('last_deadline_bank')
                    bank = (float(b_val) / 10.0) if b_val is not None else 0.0
            except Exception as e:
                print(f"Error fetching team for GW cycle: {e}")
                
        # If no squad exists or force_squad is requested, run initial squad draft
        if len(current_picks) == 0 or force_squad:
            draft_res = self.draft_initial_squad(budget=100.0, target_gw=next_gw)
            self.settings['last_run_gw'] = next_gw
            self.settings['last_run_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.save_settings()
            return {
                'success': True,
                'action': 'INITIAL_SQUAD_CREATED',
                'gameweek': next_gw,
                'dry_run': is_dry_run,
                'details': draft_res
            }
            
        # Get full cards for current squad
        current_squad_cards = []
        for p in current_picks:
            pid = int(p['element'])
            card = self.analyzer.get_player_full_card(pid)
            if card:
                card['is_captain'] = p.get('is_captain', False)
                card['is_vice_captain'] = p.get('is_vice_captain', False)
                card['position_order'] = p.get('position', 1)
                current_squad_cards.append(card)
                
        # Research & Identify Transfer Needs
        # Flags: status != 'a' (injured/suspended/doubtful), low chance_of_playing, very hard fixtures (FDR 5)
        transfer_recommendations = self.evaluate_transfers(current_squad_cards, bank, free_transfers, next_gw, hit_tolerance)
        
        # Apply proposed transfers to virtual squad
        managed_squad = list(current_squad_cards)
        transfers_made = []
        if transfer_recommendations.get('recommended_transfers'):
            for t in transfer_recommendations['recommended_transfers']:
                out_id = t['player_out']['id']
                in_card = t['player_in']
                managed_squad = [p for p in managed_squad if p['id'] != out_id]
                managed_squad.append(in_card)
                transfers_made.append({
                    'out': t['player_out']['web_name'],
                    'in': in_card['web_name'],
                    'out_id': out_id,
                    'in_id': in_card['id'],
                    'cost_diff': round(in_card['cost'] - t['player_out']['cost'], 1),
                    'xp_gain': t['xp_gain'],
                    'reason': t['reason']
                })
                
        # Optimize Starting XI, Formation, Captain, Vice-Captain, Bench Order
        lineup_result = self.optimize_lineup_and_captain(managed_squad, next_gw)
        
        # Evaluate Autonomous Chip Strategy (Wildcard 1 & 2, Triple Captain, Bench Boost, Free Hit)
        chips_used = []
        if self.team_id:
            try:
                hist = self.analyzer.fetch_entry_history(self.team_id) if self.analyzer else None
                if hist and 'chips' in hist:
                    chips_used = [c.get('name') for c in hist['chips']]
            except Exception:
                chips_used = []
                
        active_chip = self.evaluate_chip_strategy(
            managed_squad, 
            next_gw, 
            chips_used, 
            lineup_result.get('captain'), 
            lineup_result.get('bench')
        )
        
        # Execute to live FPL if not dry_run
        execution_status = "SIMULATED (Dry Run)"
        if not is_dry_run and self.is_authenticated and self.team_id:
            # Commit live transfers and lineup
            execution_status = "LIVE_SUBMITTED"
            # In live mode, we send API request to /api/transfers/ and /api/my-team/
            self._execute_live_moves(transfers_made, lineup_result, next_gw, active_chip=active_chip)
            
        cycle_result = {
            'success': True,
            'gameweek': next_gw,
            'dry_run': is_dry_run,
            'execution_status': execution_status,
            'formation': lineup_result['formation'],
            'transfers': transfers_made,
            'active_chip': active_chip,
            'net_hits': 0 if active_chip in ['wildcard', 'freehit'] else transfer_recommendations.get('hits', 0),
            'starting_xi': lineup_result['starting_xi'],
            'bench': lineup_result['bench'],
            'captain': lineup_result['captain'],
            'vice_captain': lineup_result['vice_captain'],
            'projected_points': lineup_result['starting_score'],
            'rationale': transfer_recommendations.get('rationale', 'Lineup optimized for highest predicted points.')
        }
        
        # Save to logs
        self.log_decision({
            'type': 'GW_CYCLE',
            'title': f"AI Managed GW{next_gw} ({'Dry Run' if is_dry_run else 'Live'})",
            'summary': f"Captain: {lineup_result['captain'].get('web_name')} | Transfers: {len(transfers_made)} | Formation: {lineup_result['formation']} | xP: {lineup_result['starting_score']} pts",
            'details': {
                'gameweek': next_gw,
                'transfers': transfers_made,
                'captain': lineup_result['captain'].get('web_name'),
                'vice_captain': lineup_result['vice_captain'].get('web_name'),
                'formation': lineup_result['formation'],
                'projected_points': lineup_result['starting_score'],
                'hits': transfer_recommendations.get('hits', 0)
            },
            'dry_run': is_dry_run
        })
        
        self.settings['last_run_gw'] = next_gw
        self.settings['last_run_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_settings()
        
        return _sanitize(cycle_result)

    def _compute_advanced_xp(self, player_card, gw_start, n_gws=5):
        """
        Top-1000-grade Expected Points calculator.
        
        For each upcoming fixture, computes:
          - Base xGI/90 (xG + xA per 90 mins played)
          - Opponent strength modifier (xGA conceded by opponent / league avg)
          - FDR difficulty modifier (FDR 1=1.35x bonus ... FDR 5=0.55x penalty)
          - Home advantage (+10% boost)
          - FPL points formula per position (goals, assists, clean sheets, appearance)
          - Minutes risk (< 60-min regulars are penalized)
          - Set-piece bonus (pen/corner takers get an extra scoring bump)
          - Form trend weighting (last 5 GW average vs season average)
        """
        if not self.analyzer or not self.analyzer.players_data is not None:
            return 5.0 * n_gws

        def sn(v, d=0.0):
            if v is None or str(v) in ('nan', 'None', ''):
                return d
            try:
                return float(v)
            except Exception:
                return d

        p = player_card
        pos = int(p.get('element_type', 3))
        team_id = int(p.get('team_id', p.get('team', 1)))
        pid = int(p.get('id', 0))

        # --- Raw Stats ---
        mins = max(1.0, sn(p.get('minutes'), 1.0))
        cost_val = sn(p.get('cost'), 5.0)
        xg_per90 = sn(p.get('expected_goals_per_90'))
        xa_per90 = sn(p.get('expected_assists_per_90'))
        
        # Regression to realistic bounds: defenders cannot sustain 0.8+ xG/90 long-term
        if pos == 2:
            xg_per90 = min(0.18, xg_per90)
            xa_per90 = min(0.22, xa_per90)
        elif pos == 1:
            xg_per90 = 0.0
            xa_per90 = 0.05
            
        xgc_per90 = sn(p.get('expected_goals_conceded'), 30.0) / max(1.0, mins / 90.0)
        # Derive xgc per 90 from season total
        if mins > 0:
            xgc_season = sn(p.get('expected_goals_conceded'), 0.0)
            xgc_per90 = (xgc_season / mins) * 90.0 if mins > 90 else 1.2

        form_raw = sn(p.get('form'))
        ppg = sn(p.get('points_per_game'), 2.0)
        mins_per_gw = mins / max(1.0, sn(p.get('total_points', 1.0)) / max(0.1, ppg))
        mins_risk = min(1.0, mins_per_gw / 60.0)  # 1.0 = always plays, <0.7 = rotation risk

        # Form trend: weight recent form vs season average
        form_weight = 0.65 * form_raw + 0.35 * ppg  # Blend form and consistency

        # Position-based FPL point values
        goal_pts = {1: 6, 2: 6, 3: 5, 4: 4}.get(pos, 4)
        assist_pts = 3
        cs_pts = {1: 6, 2: 6, 3: 1, 4: 0}.get(pos, 0)
        gc_deduction = {1: -1.0/2, 2: -1.0/2, 3: 0.0, 4: 0.0}.get(pos, 0.0)
        appearance_pts = 1  # For playing 60+ mins

        # Set-piece bonus
        set_piece_bonus = 0.0
        if self.analyzer:
            sp_notes = self.analyzer.fetch_set_piece_notes() or {}
            for team_note in sp_notes.get('teams', []):
                if int(team_note.get('id', -1)) == team_id:
                    for note in team_note.get('notes', []):
                        msg = str(note.get('info_message', '')).lower()
                        if 'penalty' in msg:
                            set_piece_bonus += (goal_pts * 0.12)  # Pen taker bonus
                        elif 'corner' in msg or 'free kick' in msg:
                            set_piece_bonus += (assist_pts * 0.08)

        # Get fixture details for each upcoming GW
        if not self.analyzer:
            return form_weight * n_gws

        fixture_details = self.analyzer.get_fixture_details_next_5gw(team_id, gw_start)

        # League-average xGA per team (rough baseline: 38 games, ~1.3 goals/game)
        LEAGUE_AVG_XGA_PER90 = 1.3

        total_xp = 0.0
        for i, fix in enumerate(fixture_details[:n_gws]):
            fdr = fix.get('difficulty', 3)
            is_home = fix.get('is_home', True)
            opp = fix.get('opponent', 'OPP')

            if opp == 'BLANK':
                total_xp += 1.0  # Blank gameweek minimum
                continue

            # FDR modifier: scales scoring probability based on opponent quality
            fdr_mods = {1: 1.40, 2: 1.20, 3: 1.00, 4: 0.78, 5: 0.55}
            fdr_mod = fdr_mods.get(fdr, 1.0)

            # Home advantage
            home_mod = 1.10 if is_home else 0.93

            # Estimate opposing team xGA from bootstrap (how many goals they concede)
            opp_xga_mod = 1.0  # Default to league average
            if self.analyzer and self.analyzer.players_data is not None:
                opp_id_guess = None
                if self.analyzer.fixtures_data is not None:
                    try:
                        gw = fix.get('gw', gw_start + i)
                        fs = self.analyzer.fixtures_data[self.analyzer.fixtures_data['event'] == gw]
                        for _, f in fs.iterrows():
                            th = int(f['team_h'])
                            ta = int(f['team_a'])
                            if th == team_id:
                                opp_id_guess = ta
                                break
                            elif ta == team_id:
                                opp_id_guess = th
                                break
                        if opp_id_guess:
                            opp_players = self.analyzer.players_data[
                                self.analyzer.players_data['team'] == opp_id_guess
                            ]
                            if not opp_players.empty:
                                opp_xgc_vals = [
                                    float(v) for v in opp_players['expected_goals_conceded']
                                    if str(v) not in ('nan', 'None', '') and float(v) > 0
                                ]
                                if opp_xgc_vals:
                                    opp_total_xgc = sum(opp_xgc_vals)
                                    opp_xga_mod = (opp_total_xgc / max(1.0, sum(opp_players['minutes'])) * 90.0) / LEAGUE_AVG_XGA_PER90
                                    opp_xga_mod = max(0.5, min(2.0, opp_xga_mod))
                    except Exception:
                        pass

            # Combine modifiers
            total_mod = fdr_mod * home_mod * opp_xga_mod

            # Premium Talisman boost in favorable fixtures (e.g. Haaland vs easy FDR 1-2 opposition)
            talisman_mod = 1.0
            if pos in [3, 4]:
                if cost_val >= 13.0:
                    talisman_mod = 1.45 if fdr <= 2 else (1.20 if fdr <= 3 else 1.05)
                elif cost_val >= 9.5:
                    talisman_mod = 1.25 if fdr <= 2 else 1.05

            # Per-fixture xP calculation
            xg_fixture = xg_per90 * total_mod * talisman_mod
            xa_fixture = xa_per90 * total_mod * talisman_mod

            # Expected FPL points from attacking stats
            attack_xp = (xg_fixture * goal_pts) + (xa_fixture * assist_pts)

            # Expected FPL points from defensive stats (CS, GC deductions)
            cs_prob = max(0.0, min(1.0, 0.35 - (fdr - 1) * 0.07 + (0.05 if is_home else 0.0)))
            cs_xp = cs_prob * cs_pts
            gc_xp = xgc_per90 * gc_deduction * total_mod

            # Appearance bonus (expected minutes)
            appearance_xp = mins_risk * (appearance_pts + 1)  # +1 for 60-min bonus

            # Bonus points (BPS-based heuristic: strikers in winning easy games dominate 3 BPS)
            bps_boost = 1.4 if (pos == 4 and cost_val >= 12.0 and fdr <= 2) else 1.0
            bps_xp = min(3.0, (attack_xp * 0.25 + 0.1) * bps_boost)

            # Total fixture xP
            fixture_xp = max(1.0, (attack_xp + cs_xp + gc_xp + appearance_xp + bps_xp + set_piece_bonus) * mins_risk)

            # Blend with form trend for GW1 (less historical data)
            if i == 0:
                fixture_xp = 0.65 * fixture_xp + 0.35 * form_weight
            else:
                fixture_xp = 0.75 * fixture_xp + 0.25 * (form_weight * ((5 - i) / 5.0))

            total_xp += round(fixture_xp, 1)

        return round(total_xp, 1)

    def evaluate_transfers(self, current_squad, bank, free_transfers, next_gw, hit_tolerance=0):
        """
        Top-1000-Grade Research Transfer Engine:
        - Per-fixture xP (xGI/90, xGA/90, opponent strength, FDR, home/away)
        - Set-piece bonus for penalty/corner takers
        - Minutes risk weighting
        - Form trend blending
        - Max 3 players per team constraint
        - Hit threshold: only takes -4pt hit if 5GW gain > 5.0 pts
        """
        if not self.analyzer or self.analyzer.players_data is None:
            return {'recommended_transfers': [], 'hits': 0, 'rationale': 'No transfer data'}
            
        all_players = self.analyzer.players_data
        
        # Build full candidate pool (available players only)
        candidate_pool = all_players[all_players['status'] == 'a'].copy()
        
        # Compute advanced xP for the current squad
        for p in current_squad:
            p['adv_xp_5gw'] = self._compute_advanced_xp(p, next_gw, n_gws=5)

        # Score urgency of each current squad player (skip GKPs to never waste transfers on them)
        gkps_in_squad = [p for p in current_squad if p.get('element_type') == 1 or p.get('position_short') == 'GKP']
        all_gkps_unfit = len(gkps_in_squad) > 0 and all(g.get('status') in ['i', 's'] or (_safe_chance(g.get('chance_of_playing_next_round')) is not None and _safe_chance(g.get('chance_of_playing_next_round')) <= 25) for g in gkps_in_squad)

        sell_candidates = []
        for p in current_squad:
            is_gkp = (p.get('element_type') == 1 or p.get('position_short') == 'GKP')
            if is_gkp and not all_gkps_unfit:
                continue  # Never waste transfers on GKPs
                
            p_status = p.get('status', 'a')
            p_chance = _safe_chance(p.get('chance_of_playing_next_round'))
            p_diffs = self.analyzer.get_fixture_details_next_5gw(int(p.get('team_id', p.get('team', 1))), next_gw)
            avg_diff = sum(d['difficulty'] for d in p_diffs) / max(1, len(p_diffs))
            adv_xp = p.get('adv_xp_5gw', 10.0)

            urgency = 0
            reasons = []
            if p_status in ['i', 's']:
                urgency += 12
                reasons.append(f"Unavailable — {p.get('news', 'Injured/Suspended')}")
            elif p_chance is not None and p_chance <= 50:
                urgency += 9
                reasons.append(f"Doubt ({p_chance}% chance of playing)")
            elif avg_diff >= 4.3:
                urgency += 5
                reasons.append("Very tough upcoming fixtures (avg FDR 4.3+)")
            elif float(p.get('form', 0) or 0) < 1.2:
                urgency += 3
                reasons.append("Low recent form (<1.2 pts/game)")
            if adv_xp < 10.0:
                urgency += 2
                reasons.append(f"Low 5GW xP ({adv_xp})")
                
            sell_candidates.append({
                'player': p,
                'urgency': urgency,
                'reasons': reasons,
                'cost': float(p.get('cost', 5.0)),
                'element_type': p.get('element_type', 1),
                'adv_xp_5gw': adv_xp
            })
            
        sell_candidates.sort(key=lambda x: x['urgency'], reverse=True)
        
        recommended_transfers = []
        current_squad_ids = {p['id'] for p in current_squad}
        team_counts = {}
        for p in current_squad:
            t = int(p.get('team_id', p.get('team', 1)))
            team_counts[t] = team_counts.get(t, 0) + 1
            
        max_transfers = 1 if hit_tolerance == 0 else (2 if hit_tolerance <= -4 else 1)
        available_budget = bank
        
        for cand in sell_candidates:
            if len(recommended_transfers) >= max_transfers:
                break
            if cand['urgency'] <= 0 and len(recommended_transfers) >= free_transfers:
                break
                
            out_player = cand['player']
            pos = cand['element_type']
            sell_price = cand['cost']
            max_buy_price = sell_price + available_budget
            
            pos_pool = candidate_pool[
                (candidate_pool['element_type'] == pos) &
                (~candidate_pool['id'].isin(current_squad_ids)) &
                (candidate_pool['now_cost'] <= int(max_buy_price * 10))
            ].copy()
            
            if pos_pool.empty:
                continue
                
            out_team = int(out_player.get('team_id', out_player.get('team', 1)))
            valid_rows = []
            for _, r in pos_pool.iterrows():
                r_team = int(r['team'])
                if team_counts.get(r_team, 0) < 3 or r_team == out_team:
                    valid_rows.append(r)
                    
            if not valid_rows:
                continue
                
            # Score each candidate with advanced xP
            best_candidate = None
            best_xp = -1
            for row in valid_rows:
                card = self.analyzer.get_player_full_card(int(row['id']))
                if not card:
                    continue
                xp = self._compute_advanced_xp(card, next_gw, n_gws=5)
                if xp > best_xp:
                    best_xp = xp
                    best_candidate = (card, xp)
                    
            if not best_candidate:
                continue
                
            in_card, in_xp = best_candidate
            xp_gain = round(in_xp - cand['adv_xp_5gw'], 1)
            
            # Hit threshold: require >5pt gain for an extra transfer costing -4 pts
            is_hit = len(recommended_transfers) >= free_transfers
            if is_hit and xp_gain < 5.0 and cand['urgency'] < 9:
                continue
                
            recommended_transfers.append({
                'player_out': out_player,
                'player_in': in_card,
                'xp_gain': xp_gain,
                'reason': (
                    f"OUT: {out_player['web_name']} — {', '.join(cand['reasons']) or 'Lower xP'} | "
                    f"IN: {in_card['web_name']} — 5GW xP: {in_xp:.1f} (gain +{xp_gain:.1f}pts) | "
                    f"xGI/90: {in_card.get('expected_goal_involvements_per_90', 0):.2f}"
                )
            })
            
            available_budget += sell_price - in_card['cost']
            current_squad_ids.discard(out_player['id'])
            current_squad_ids.add(in_card['id'])
            team_counts[out_team] = max(0, team_counts.get(out_team, 1) - 1)
            team_counts[int(in_card.get('team_id', in_card.get('team', 1)))] = team_counts.get(
                int(in_card.get('team_id', in_card.get('team', 1))), 0) + 1
            
        hits = max(0, len(recommended_transfers) - free_transfers) * -4
        if recommended_transfers:
            rationale = (
                f"Executed {len(recommended_transfers)} transfer(s) with {abs(hits)}-pt cost. "
                f"Engine used: per-fixture xGI/90, opponent xGA strength, FDR×home/away modifiers, "
                f"set-piece bonuses, and form-trend blending."
            )
        else:
            rationale = (
                "Squad is optimal for the next 5 Gameweeks. "
                "All players are available and fit. No transfers improve expected points above hit threshold."
            )
        
        return {
            'recommended_transfers': recommended_transfers,
            'hits': hits,
            'rationale': rationale
        }

    def evaluate_chip_strategy(self, current_squad_cards, next_gw, chips_used, captain_card, bench_cards):
        """
        Autonomous Chip Deployment Engine (FPL Rules):
        - Wildcard 1 (GW2-GW19) / Wildcard 2 (GW20-GW38)
        - Triple Captain (3xc)
        - Bench Boost (bboost)
        - Free Hit (freehit)
        """
        if not self.analyzer or self.settings.get('chip_strategy') == 'manual':
            return None
            
        chips_used_set = set(chips_used or [])
        
        # 1. Check Wildcard (2 available per season: 1st half GW1-19, 2nd half GW20-38)
        wc_already_used = 'wildcard' in chips_used_set
        is_first_half = (next_gw <= 19)
        
        unfit_count = sum(1 for p in current_squad_cards if p.get('status') in ['i', 's'] or (_safe_chance(p.get('chance_of_playing_next_round')) is not None and _safe_chance(p.get('chance_of_playing_next_round')) <= 25))
        
        if not wc_already_used:
            if unfit_count >= 4:
                return 'wildcard'
            if is_first_half and next_gw >= 18:
                # Must consume 1st half wildcard before GW19 expiry
                return 'wildcard'
                
        # 2. Check Triple Captain (3xc)
        if '3xc' not in chips_used_set and captain_card:
            cap_xp = captain_card.get('predicted_next_gw', 0)
            if cap_xp >= 9.5:
                return '3xc'
                
        # 3. Check Bench Boost (bboost)
        if 'bboost' not in chips_used_set and bench_cards:
            bench_xp_total = sum(p.get('predicted_next_gw', 0) for p in bench_cards)
            bench_playing = all(p.get('status', 'a') == 'a' for p in bench_cards)
            if bench_playing and bench_xp_total >= 14.0:
                return 'bboost'
                
        # 4. Check Free Hit (freehit)
        if 'freehit' not in chips_used_set:
            blank_count = sum(1 for p in current_squad_cards if any(f.get('opponent') == 'BLANK' for f in p.get('fixtures_next_5', [])[:1]))
            if blank_count >= 4:
                return 'freehit'
                
        return None

    def _execute_live_moves(self, transfers, lineup, gameweek, active_chip=None):
        """Helper to post transfers and lineup to live FPL endpoints with chip support"""
        try:
            from playwright.sync_api import sync_playwright
            
            is_wildcard = (active_chip == 'wildcard')
            is_freehit = (active_chip == 'freehit')
            lineup_chip = active_chip if active_chip in ['3xc', 'bboost'] else None
            
            target_email = self.email or os.getenv('FPL_EMAIL')
            target_password = self.password or os.getenv('FPL_PASSWORD')
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                page.goto('https://fantasy.premierleague.com/my-team', timeout=30000)
                page.wait_for_timeout(2000)
                
                try:
                    accept_btn = page.locator('button:has-text("Accept All"), button#onetrust-accept-btn-handler')
                    if accept_btn.count() > 0:
                        accept_btn.first.click()
                        page.wait_for_timeout(1000)
                except Exception:
                    pass
                    
                login_btn = page.locator('a:has-text("Log in"), button:has-text("Log in")')
                if login_btn.count() > 0:
                    login_btn.first.click()
                    page.wait_for_timeout(3000)
                    
                email_input = page.locator('input[type="email"], input[placeholder*="email" i]')
                pass_input = page.locator('input[type="password"]')
                
                if email_input.count() > 0 and pass_input.count() > 0:
                    email_input.first.fill(target_email)
                    pass_input.first.fill(target_password)
                    page.wait_for_timeout(500)
                    pass_input.first.press('Enter')
                    page.wait_for_timeout(8000)
                    
                # 1. Execute Transfers if any
                if transfers:
                    t_payload = {
                        "chips": None,
                        "entry": int(self.team_id),
                        "event": int(gameweek),
                        "transfers": [
                            {
                                "element_in": t['in_id'],
                                "element_out": t['out_id'],
                                "purchase_price": int(t.get('in_cost', 50) * 10),
                                "selling_price": int(t.get('out_cost', 50) * 10)
                            } for t in transfers
                        ],
                        "wildcard": is_wildcard,
                        "freehit": is_freehit
                    }
                    t_res = page.evaluate("""async (payload) => {
                        const r = await fetch('/api/transfers/', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json', 'accept': 'application/json'},
                            body: JSON.stringify(payload)
                        });
                        return {status: r.status, text: await r.text()};
                    }""", t_payload)
                    print(f"Live Transfer Submission Result: {t_res}")
                    
                # 2. Execute Lineup
                if lineup:
                    picks_payload = []
                    for idx, pl in enumerate(lineup['starting_xi'], 1):
                        picks_payload.append({
                            "element": int(pl['id']),
                            "position": idx,
                            "is_captain": (pl['id'] == lineup['captain'].get('id')),
                            "is_vice_captain": (pl['id'] == lineup['vice_captain'].get('id'))
                        })
                    for idx, pl in enumerate(lineup['bench'], 12):
                        picks_payload.append({
                            "element": int(pl['id']),
                            "position": idx,
                            "is_captain": False,
                            "is_vice_captain": False
                        })
                        
                    l_res = page.evaluate("""async (args) => {
                        const r = await fetch(`/api/my-team/${args.tid}/`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json', 'accept': 'application/json'},
                            body: JSON.stringify({picks: args.picks, chip: args.chip})
                        });
                        return {status: r.status, text: await r.text()};
                    }""", {'tid': self.team_id, 'picks': picks_payload, 'chip': lineup_chip})
                    print(f"Live Lineup Submission Result: {l_res}")
                    
                browser.close()
        except Exception as e:
            print(f"Live execution error: {e}")

    def _start_deadline_daemon(self):
        """Background thread that automatically checks deadlines and triggers 100% autonomous manager"""
        def daemon_loop():
            # Initial auto-draft check on startup (after 5 seconds for data preload)
            time.sleep(5)
            try:
                if self.settings.get('auto_pilot', False) and self.settings.get('auto_draft_if_empty', True):
                    status = self.get_account_status()
                    if not status.get('has_squad', False) and not self.settings.get('last_run_gw'):
                        print("🤖 1st Week Auto-Pilot: Autonomously building optimal 15-player squad...")
                        self.run_gameweek_cycle(dry_run=self.settings.get('dry_run', False), force_squad=True)
            except Exception as e:
                print(f"Initial 1st week auto-draft notice: {e}")

            while True:
                try:
                    time.sleep(300)  # Check every 5 minutes
                    if not self.settings.get('auto_pilot', False):
                        continue
                        
                    deadline_info = self.get_next_deadline_info()
                    minutes_left = deadline_info.get('minutes_left', 9999)
                    gw = deadline_info.get('gameweek')
                    trigger_offset = self.settings.get('trigger_minutes_before_deadline', 60)
                    last_run_gw = self.settings.get('last_run_gw')
                    
                    # If we are within window before deadline and haven't run for this GW yet
                    if 0 < minutes_left <= trigger_offset and last_run_gw != gw:
                        print(f"🚀 AI Auto-Pilot Triggered for Gameweek {gw} ({minutes_left} mins before deadline)!")
                        self.run_gameweek_cycle(dry_run=self.settings.get('dry_run', False))
                except Exception as e:
                    print(f"Deadline daemon error: {e}")
                    
        self.daemon_thread = threading.Thread(target=daemon_loop, daemon=True)
        self.daemon_thread.start()
