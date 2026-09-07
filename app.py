from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from datetime import datetime
import json
import os
import re
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

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

app = Flask(__name__)


class FPLAnalyzer:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.bootstrap_data = None
        self.players_data = None
        self.teams_data = None
        self.fixtures_data = None
        self.models = {}
        self.scalers = {}
        self.history_cache = {}
        self.team_name_map = {}
        self.team_short_map = {}
        
    def fetch_bootstrap_data(self):
        try:
            response = self.session.get(f"{self.base_url}/bootstrap-static/")
            response.raise_for_status()
            self.bootstrap_data = response.json()
            self.players_data = pd.DataFrame(self.bootstrap_data['elements'])
            self.teams_data = pd.DataFrame(self.bootstrap_data['teams'])
            
            # Create fast team lookups
            for _, t in self.teams_data.iterrows():
                self.team_name_map[int(t['id'])] = str(t['name'])
                self.team_short_map[int(t['id'])] = str(t['short_name'])
                
            return True
        except Exception as e:
            print(f"Error fetching bootstrap data: {str(e)}")
            return False
    
    def fetch_fixtures_data(self):
        try:
            response = self.session.get(f"{self.base_url}/fixtures/")
            response.raise_for_status()
            self.fixtures_data = pd.DataFrame(response.json())
            return True
        except Exception as e:
            print(f"Error fetching fixtures: {str(e)}")
            return False
    
    def fetch_team_data(self, team_id):
        try:
            team_id_str = str(team_id)
            response = self.session.get(f"{self.base_url}/entry/{team_id_str}/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching team data: {str(e)}")
            return None
    
    def fetch_team_transfers(self, team_id):
        try:
            team_id_str = str(team_id)
            response = self.session.get(f"{self.base_url}/entry/{team_id_str}/transfers/")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def fetch_team_picks(self, team_id, gameweek):
        try:
            team_id_str = str(team_id)
            gameweek_str = str(gameweek)
            response = self.session.get(f"{self.base_url}/entry/{team_id_str}/event/{gameweek_str}/picks/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching team picks: {str(e)}")
            return None
    
    def fetch_classic_league(self, league_id, page=1):
        try:
            league_id_str = str(league_id)
            page_str = str(page)
            response = self.session.get(f"{self.base_url}/leagues-classic/{league_id_str}/standings/?page_standings={page_str}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching league data: {str(e)}")
            return None

    def fetch_entry_history(self, manager_id):
        """Endpoint 6: Manager past gameweeks & chips used history"""
        try:
            res = self.session.get(f"{self.base_url}/entry/{str(manager_id)}/history/")
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None

    def fetch_event_live(self, event_id):
        """Endpoint 4: Gameweek live scores, match stats and explain breakdown"""
        try:
            res = self.session.get(f"{self.base_url}/event/{str(event_id)}/live/")
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None

    def fetch_event_status(self):
        """Endpoint 10: Event calculation and bonus status updates"""
        try:
            res = self.session.get(f"{self.base_url}/event-status/")
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None

    def fetch_dream_team(self, event_id):
        """Endpoint 11: Dream team for gameweek"""
        try:
            res = self.session.get(f"{self.base_url}/dream-team/{str(event_id)}/")
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None

    def fetch_set_piece_notes(self):
        """Endpoint 12: Confirmed penalty, free kick, and corner takers"""
        try:
            res = self.session.get(f"{self.base_url}/team/set-piece-notes/")
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None
    
    def fetch_player_history(self, player_id):
        if player_id in self.history_cache:
            return self.history_cache[player_id]
        
        if self.get_current_gameweek() <= 1:
            empty_hist = {'history': [], 'history_past': []}
            self.history_cache[player_id] = empty_hist
            return empty_hist

        try:
            player_id_str = str(player_id)
            response = self.session.get(f"{self.base_url}/element-summary/{player_id_str}/")
            response.raise_for_status()
            data = response.json()
            self.history_cache[player_id] = data
            return data
        except Exception as e:
            return None
    
    def get_current_gameweek(self):
        if self.bootstrap_data and 'events' in self.bootstrap_data:
            for event in self.bootstrap_data['events']:
                if event.get('is_current'):
                    return event['id']
                if event.get('is_next'):
                    return max(1, event['id'] - 1)
        return 1

    def get_next_gameweek(self):
        if self.bootstrap_data and 'events' in self.bootstrap_data:
            for event in self.bootstrap_data['events']:
                if event.get('is_next'):
                    return event['id']
                if event.get('is_current'):
                    return min(38, event['id'] + 1)
        return 1
    
    def get_fixture_details_next_5gw(self, team_id, current_gw=None):
        if self.fixtures_data is None:
            return []
        
        if current_gw is None:
            current_gw = self.get_next_gameweek()
        
        details = []
        for gw in range(current_gw, min(39, current_gw + 5)):
            try:
                team_fixtures = self.fixtures_data[
                    ((self.fixtures_data['team_h'] == team_id) | (self.fixtures_data['team_a'] == team_id)) &
                    (self.fixtures_data['event'] == gw)
                ]
                
                if len(team_fixtures) == 0:
                    details.append({'gw': gw, 'opponent': 'BLANK', 'difficulty': 3, 'is_home': True})
                    continue
                
                fixture = team_fixtures.iloc[0]
                is_home = (fixture['team_h'] == team_id)
                opp_id = fixture['team_a'] if is_home else fixture['team_h']
                opp_short = self.team_short_map.get(int(opp_id), 'OPP')
                diff = int(fixture.get('team_h_difficulty' if is_home else 'team_a_difficulty', 3))
                
                details.append({
                    'gw': gw,
                    'opponent': opp_short,
                    'difficulty': diff,
                    'is_home': is_home
                })
            except Exception:
                details.append({'gw': gw, 'opponent': 'TBD', 'difficulty': 3, 'is_home': True})
        
        return details


    def get_fixture_difficulty_next_5gw(self, team_id, current_gw):
        details = self.get_fixture_details_next_5gw(team_id, current_gw)
        return [d['difficulty'] for d in details] if details else [3, 3, 3, 3, 3]
    
    def predict_player_points_5gw(self, player_id, current_gw):
        player_rows = self.players_data[self.players_data['id'] == player_id]
        if player_rows.empty:
            return [2.0] * 5
        player = player_rows.iloc[0]
        
        base_prediction = self.predict_player_points(player_id)
        base_points = base_prediction['predicted_points'] if base_prediction else 2.0
        
        fixture_difficulties = self.get_fixture_difficulty_next_5gw(int(player['team']), current_gw)
        
        predictions = []
        for i, difficulty in enumerate(fixture_difficulties):
            difficulty_modifier = (6 - difficulty) / 5.0
            gw_prediction = base_points * (0.8 + difficulty_modifier * 0.4)
            predictions.append(max(0.0, round(float(gw_prediction), 1)))
        
        return predictions
    
    def _extract_feature_vector(self, player, gw_data=None):
        def safe_float(val, default=0.0):
            if val is None or str(val) in ('nan', 'None', ''):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        form = safe_float(player.get('form'))
        ppg = safe_float(player.get('points_per_game'))
        total_pts = safe_float(player.get('total_points')) / 100.0
        minutes = safe_float(player.get('minutes')) / 1000.0
        goals = safe_float(player.get('goals_scored'))
        assists = safe_float(player.get('assists'))
        clean_sheets = safe_float(player.get('clean_sheets'))
        saves = safe_float(player.get('saves'))
        bonus = safe_float(player.get('bonus'))
        influence = safe_float(player.get('influence')) / 100.0
        creativity = safe_float(player.get('creativity')) / 100.0
        threat = safe_float(player.get('threat')) / 100.0
        yellow_cards = safe_float(player.get('yellow_cards'))
        red_cards = safe_float(player.get('red_cards'))
        own_goals = safe_float(player.get('own_goals'))
        pen_saved = safe_float(player.get('penalties_saved'))
        pen_missed = safe_float(player.get('penalties_missed'))
        sel_pct = safe_float(player.get('selected_by_percent')) / 100.0
        
        # xG & xA advanced features
        xg = safe_float(player.get('expected_goals'))
        xa = safe_float(player.get('expected_assists'))
        xgi = safe_float(player.get('expected_goal_involvements'))
        xgc = safe_float(player.get('expected_goals_conceded'))
        xg_per90 = safe_float(player.get('expected_goals_per_90'))
        xa_per90 = safe_float(player.get('expected_assists_per_90'))

        if gw_data:
            gw_min = safe_float(gw_data.get('minutes')) / 90.0
            gw_g = safe_float(gw_data.get('goals_scored'))
            gw_a = safe_float(gw_data.get('assists'))
            gw_cs = safe_float(gw_data.get('clean_sheets'))
            gw_gc = safe_float(gw_data.get('goals_conceded'))
            gw_saves = safe_float(gw_data.get('saves'))
            gw_b = safe_float(gw_data.get('bonus'))
        else:
            gw_min = 0.5
            gw_g = 0.0
            gw_a = 0.0
            gw_cs = 0.0
            gw_gc = 0.0
            gw_saves = 0.0
            gw_b = 0.0

        return [
            form, ppg, total_pts, minutes, goals, assists, clean_sheets,
            saves, bonus, influence, creativity, threat, yellow_cards,
            red_cards, own_goals, pen_saved, pen_missed, sel_pct,
            xg, xa, xgi, xgc, xg_per90, xa_per90,
            gw_min, gw_g, gw_a, gw_cs, gw_gc, gw_saves, gw_b
        ]

    def _estimate_gw1_points(self, player):
        for field in ['ep_this', 'ep_next']:
            val = player.get(field)
            if val is not None and str(val) not in ('nan', 'None', ''):
                try:
                    ep = float(val)
                    if ep > 0:
                        return ep
                except (ValueError, TypeError):
                    pass
        
        ppg = player.get('points_per_game')
        if ppg is not None and str(ppg) not in ('nan', 'None', ''):
            try:
                f_ppg = float(ppg)
                if f_ppg > 0:
                    return f_ppg
            except (ValueError, TypeError):
                pass
                
        pos = int(player.get('element_type', 3))
        cost = float(player.get('now_cost', 50)) / 10.0
        base = {1: 3.5, 2: 3.8, 3: 4.0, 4: 4.2}.get(pos, 3.5)
        price_boost = max(0.0, (cost - 4.5) * 0.45)
        
        ict = player.get('ict_index')
        ict_boost = 0.0
        if ict is not None and str(ict) not in ('nan', 'None', ''):
            try:
                ict_boost = float(ict) / 50.0
            except (ValueError, TypeError):
                pass

        # xG / xA boost
        xg_boost = 0.0
        xgi = player.get('expected_goal_involvements')
        if xgi is not None and str(xgi) not in ('nan', 'None', ''):
            try:
                xg_boost = float(xgi) * 0.3
            except (ValueError, TypeError):
                pass

        return round(base + price_boost + ict_boost + xg_boost, 2)

    def prepare_training_data(self):
        features = []
        targets = []
        
        current_gw = self.get_current_gameweek()
        
        if current_gw > 1 and self.players_data is not None:
            top_players = self.players_data.sort_values('total_points', ascending=False).head(100)
            for idx, (_, player) in enumerate(top_players.iterrows()):
                try:
                    history = self.fetch_player_history(player['id'])
                    if history and 'history' in history and len(history['history']) >= 2:
                        gw_history = history['history']
                        for i in range(len(gw_history) - 1):
                            current_gw_data = gw_history[i]
                            next_gw_data = gw_history[i + 1]
                            
                            feature_vector = self._extract_feature_vector(player, current_gw_data)
                            target_points = float(next_gw_data.get('total_points', 0))
                            
                            features.append(feature_vector)
                            targets.append(target_points)
                except Exception:
                    continue

        if len(features) == 0 and self.players_data is not None:
            for _, player in self.players_data.iterrows():
                try:
                    feature_vector = self._extract_feature_vector(player, None)
                    estimated_pts = self._estimate_gw1_points(player)
                    features.append(feature_vector)
                    targets.append(estimated_pts)
                except Exception:
                    continue
        
        return np.array(features), np.array(targets)
    
    def train_models(self):
        if len(self.models) > 0:
            return True
            
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            return False
        
        test_size = 0.2 if len(X) > 20 else 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        models_config = {
            'random_forest': RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=30, max_depth=5, random_state=42),
        }
        
        if xgb is not None and len(X) > 100:
            models_config['xgboost'] = xgb.XGBRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)

            
        if len(X) > 200:
            models_config['neural_network'] = MLPRegressor(
                hidden_layer_sizes=(32, 16), 
                max_iter=100, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        for name, model in models_config.items():
            try:
                if name == 'neural_network':
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                self.models[name] = model
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        return len(self.models) > 0
    
    def predict_player_points(self, player_id, gameweek=None):
        if not self.models:
            self.train_models()
            
        player_rows = self.players_data[self.players_data['id'] == player_id] if self.players_data is not None else pd.DataFrame()
        if player_rows.empty:
            return {
                'predicted_points': 2.0,
                'confidence': 50.0,
                'model_predictions': {}
            }
        player = player_rows.iloc[0]
        
        latest_gw = None
        if self.get_current_gameweek() > 1:
            history = self.fetch_player_history(player_id)
            if history and 'history' in history and history['history']:
                latest_gw = history['history'][-1]
                
        feature_vector = np.array([self._extract_feature_vector(player, latest_gw)])
        
        predictions = {}
        for name, model in self.models.items():
            try:
                if name == 'neural_network':
                    feature_scaled = self.scalers['standard'].transform(feature_vector)
                    pred = model.predict(feature_scaled)[0]
                else:
                    pred = model.predict(feature_vector)[0]
                predictions[name] = max(0.0, float(pred))
            except Exception:
                predictions[name] = 2.0
        
        ensemble_pred = np.mean(list(predictions.values())) if predictions else 2.0
        prediction_std = np.std(list(predictions.values())) if predictions else 0.0
        confidence = min(95.0, max(30.0, 80.0 - prediction_std * 10.0))
        
        est_gw1 = self._estimate_gw1_points(player)
        if ensemble_pred < 1.0 and est_gw1 > 1.0:
            ensemble_pred = est_gw1

        return {
            'predicted_points': round(float(ensemble_pred), 1),
            'confidence': round(float(confidence), 1),
            'model_predictions': predictions
        }
    
    def predict_all_players_points(self):
        if self.players_data is None or self.players_data.empty:
            return {}
            
        if not self.models:
            self.train_models()
        
        feature_list = []
        player_ids = []
        
        for _, player in self.players_data.iterrows():
            player_ids.append(player['id'])
            feature_list.append(self._extract_feature_vector(player, None))
            
        feature_matrix = np.array(feature_list)
        
        if 'standard' in self.scalers:
            feature_scaled = self.scalers['standard'].transform(feature_matrix)
        else:
            feature_scaled = feature_matrix
        
        all_preds = np.zeros(len(player_ids))
        model_count = 0
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network':
                    preds = model.predict(feature_scaled)
                else:
                    preds = model.predict(feature_matrix)
                all_preds += np.maximum(0.0, preds)
                model_count += 1
            except Exception:
                continue
                
        if model_count > 0:
            all_preds /= model_count
        else:
            all_preds = np.full(len(player_ids), 2.0)
            
        predictions_map = {}
        for idx, p_id in enumerate(player_ids):
            player = self.players_data.iloc[idx]
            pred = all_preds[idx]
            est_gw1 = self._estimate_gw1_points(player)
            if pred < 1.0 and est_gw1 > 1.0:
                pred = est_gw1
            predictions_map[p_id] = round(float(pred), 1)
            
        return predictions_map

    def get_fixture_difficulty(self, team_id, gameweek=None):
        if self.fixtures_data is None:
            return 3
        
        if gameweek is None:
            gameweek = self.get_current_gameweek()
        
        try:
            team_fixtures = self.fixtures_data[
                ((self.fixtures_data['team_h'] == team_id) | (self.fixtures_data['team_a'] == team_id)) &
                (self.fixtures_data['event'] == gameweek)
            ]
            
            if len(team_fixtures) == 0:
                return 3
            
            fixture = team_fixtures.iloc[0]
            if fixture['team_h'] == team_id:
                return int(fixture.get('team_h_difficulty', 3))
            else:
                return int(fixture.get('team_a_difficulty', 3))
        except:
            return 3
            
    def get_player_full_card(self, player_id):
        if self.players_data is None or self.players_data.empty:
            return None
        rows = self.players_data[self.players_data['id'] == player_id]
        if rows.empty:
            return None
        p = rows.iloc[0]
        
        team_id = int(p['team'])
        team_name = self.team_name_map.get(team_id, "Unknown")
        team_short = self.team_short_map.get(team_id, "UNK")
        
        element_type = int(p['element_type'])
        pos_names = {1: ('Goalkeeper', 'GKP'), 2: ('Defender', 'DEF'), 3: ('Midfielder', 'MID'), 4: ('Forward', 'FWD')}
        pos_name, pos_short = pos_names.get(element_type, ('Midfielder', 'MID'))
        
        prediction = self.predict_player_points(player_id)
        next_gw = self.get_next_gameweek()
        fixtures_5gw = self.get_fixture_details_next_5gw(team_id, next_gw)

        
        def safe_num(v, default=0.0):
            try:
                if v is None or str(v) in ('nan', 'None', ''):
                    return default
                return float(v)
            except:
                return default

        xg = safe_num(p.get('expected_goals'))
        xa = safe_num(p.get('expected_assists'))
        xgi = safe_num(p.get('expected_goal_involvements'))
        xgc = safe_num(p.get('expected_goals_conceded'))
        xg_per90 = safe_num(p.get('expected_goals_per_90'))
        xa_per90 = safe_num(p.get('expected_assists_per_90'))
        xgi_per90 = safe_num(p.get('expected_goal_involvements_per_90'))
        
        goals = int(safe_num(p.get('goals_scored')))
        assists = int(safe_num(p.get('assists')))
        xg_delta = round(goals - xg, 2)
        xa_delta = round(assists - xa, 2)
        
        return {
            'id': int(p['id']),
            'web_name': str(p['web_name']),
            'full_name': f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
            'team_id': team_id,
            'team_name': team_name,
            'team_short': team_short,
            'position': pos_name,
            'position_short': pos_short,
            'element_type': element_type,
            'cost': float(p['now_cost']) / 10.0,
            'total_points': int(safe_num(p.get('total_points'))),
            'form': float(safe_num(p.get('form'))),
            'points_per_game': float(safe_num(p.get('points_per_game'))),
            'selected_by_percent': float(safe_num(p.get('selected_by_percent'))),
            'minutes': int(safe_num(p.get('minutes'))),
            'goals_scored': goals,
            'assists': assists,
            'clean_sheets': int(safe_num(p.get('clean_sheets'))),
            'expected_goals': xg,
            'expected_assists': xa,
            'expected_goal_involvements': xgi,
            'expected_goals_conceded': xgc,
            'expected_goals_per_90': xg_per90,
            'expected_assists_per_90': xa_per90,
            'expected_goal_involvements_per_90': xgi_per90,
            'xg_delta': xg_delta, # > 0 means overperforming xG, < 0 means underperforming (due positive regression)
            'xa_delta': xa_delta,
            'ict_index': float(safe_num(p.get('ict_index'))),
            'threat': float(safe_num(p.get('threat'))),
            'creativity': float(safe_num(p.get('creativity'))),
            'influence': float(safe_num(p.get('influence'))),
            'status': str(p.get('status', 'a')),
            'news': str(p.get('news', '')),
            'chance_of_playing_next_round': p.get('chance_of_playing_next_round'),
            'predicted_points': prediction['predicted_points'] if prediction else 2.0,
            'confidence': prediction['confidence'] if prediction else 50.0,
            'fixtures_next_5': fixtures_5gw
        }
    
    def generate_optimal_squad(self, budget=1000, current_gw=None):
        if current_gw is None:
            current_gw = self.get_current_gameweek()
        
        available_players = self.players_data[
            (self.players_data['status'] == 'a') &
            (self.players_data['now_cost'] <= budget)
        ].copy()
        
        predictions_map = self.predict_all_players_points()

        # Fetch set piece notes once for all players
        sp_notes = self.fetch_set_piece_notes() or {}
        sp_team_map = {}  # team_id -> set-piece bonus pts
        for team_note in sp_notes.get('teams', []):
            tid = int(team_note.get('id', -1))
            bonus = 0.0
            for note in team_note.get('notes', []):
                msg = str(note.get('info_message', '')).lower()
                if 'penalty' in msg:
                    bonus += 0.72   # ~12% chance of scoring from pen taker
                elif 'corner' in msg or 'free kick' in msg:
                    bonus += 0.24
            sp_team_map[tid] = bonus
        
        LEAGUE_AVG_XGA_PER90 = 1.3

        # Build opponent xGA modifier map: team_id -> how weak their defense is
        team_opp_xga = {}
        for t_id in self.players_data['team'].unique():
            opp_players = self.players_data[self.players_data['team'] == t_id]
            try:
                xgc_vals = [float(v) for v in opp_players['expected_goals_conceded'] if str(v) not in ('nan', 'None', '') and float(v) > 0]
                mins_vals = [float(v) for v in opp_players['minutes'] if str(v) not in ('nan', 'None', '') and float(v) > 0]
                if xgc_vals and mins_vals:
                    xgc_per90 = (sum(xgc_vals) / max(1.0, sum(mins_vals))) * 90.0
                    team_opp_xga[int(t_id)] = max(0.5, min(2.0, xgc_per90 / LEAGUE_AVG_XGA_PER90))
            except Exception:
                pass

        # Calculate multi-dimensional historical & predictive score for each player
        for _, player in available_players.iterrows():
            base_pts = predictions_map.get(int(player['id']), 2.0)
            team_id = int(player['team'])
            pos = int(player.get('element_type', 3))
            
            # Per-90 stats
            mins = max(1.0, float(pd.to_numeric(player.get('minutes'), errors='coerce') or 1.0))
            xg_p90 = float(pd.to_numeric(player.get('expected_goals_per_90'), errors='coerce') or 0)
            xa_p90 = float(pd.to_numeric(player.get('expected_assists_per_90'), errors='coerce') or 0)
            form_val = float(pd.to_numeric(player.get('form'), errors='coerce') or 0)
            ppg_val = float(pd.to_numeric(player.get('points_per_game'), errors='coerce') or 0)
            mins_risk = min(1.0, mins / max(1.0, float(pd.to_numeric(player.get('total_points'), errors='coerce') or 1) / max(0.1, ppg_val) * 60.0))
            
            # FPL scoring weights by position
            goal_pts = {1: 6, 2: 6, 3: 5, 4: 4}.get(pos, 4)
            cs_pts = {1: 6, 2: 6, 3: 1, 4: 0}.get(pos, 0)
            
            # Advanced 5GW xP using fixture details
            fixture_details = self.get_fixture_details_next_5gw(team_id, current_gw)
            fdr_mods = {1: 1.40, 2: 1.20, 3: 1.00, 4: 0.78, 5: 0.55}
            total_5gw = 0.0
            for fix in fixture_details:
                fdr = fix.get('difficulty', 3)
                is_home = fix.get('is_home', True)
                opp_str = 'BLANK'
                # Look up opponent team from fixture
                gw_fix = fix.get('gw', current_gw)
                if self.fixtures_data is not None:
                    try:
                        fs = self.fixtures_data[self.fixtures_data['event'] == gw_fix]
                        for _, f in fs.iterrows():
                            if int(f['team_h']) == team_id:
                                opp_xga = team_opp_xga.get(int(f['team_a']), 1.0)
                                opp_str = 'found'
                                break
                            elif int(f['team_a']) == team_id:
                                opp_xga = team_opp_xga.get(int(f['team_h']), 1.0)
                                opp_str = 'found'
                                break
                    except Exception:
                        opp_xga = 1.0
                else:
                    opp_xga = 1.0

                if fix.get('opponent', '') == 'BLANK':
                    total_5gw += 1.0
                    continue

                fdr_mod = fdr_mods.get(fdr, 1.0)
                home_mod = 1.10 if is_home else 0.93
                total_mod = fdr_mod * home_mod * opp_xga

                attack_xp = (xg_p90 * goal_pts + xa_p90 * 3) * total_mod
                cs_prob = max(0.0, min(1.0, 0.35 - (fdr - 1) * 0.07 + (0.05 if is_home else 0.0)))
                cs_xp = cs_prob * cs_pts
                appearance_xp = mins_risk * 2.0
                bps_xp = min(2.0, attack_xp * 0.22 + 0.1)
                fixture_xp = max(1.0, (attack_xp + cs_xp + appearance_xp + bps_xp) * mins_risk)
                total_5gw += fixture_xp

            # Set-piece bonus
            sp_bonus = sp_team_map.get(team_id, 0.0)

            # Composite quality score
            composite_score = (
                total_5gw * 0.50 +
                (form_val * 1.8) +
                (ppg_val * 1.4) +
                (xg_p90 * goal_pts * 2.0) +
                (xa_p90 * 3 * 1.5) +
                (mins_risk * 2.0) +
                sp_bonus
            )

            available_players.loc[available_players['id'] == player['id'], 'predicted_5gw'] = round(total_5gw, 1)
            available_players.loc[available_players['id'] == player['id'], 'composite_score'] = composite_score
            available_players.loc[available_players['id'] == player['id'], 'value_ratio'] = composite_score / (player['now_cost'] / 10.0)
        
        position_limits = {1: (2, 2), 2: (5, 5), 3: (5, 5), 4: (3, 3)}
        
        best_squad = None
        best_score = 0
        
        # Multi-attempt stochastic solver to find global optimum within £100m and 3/team constraints
        for attempt in range(40):
            squad = []
            total_cost = 0
            total_predicted = 0
            position_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            team_counts = {}
            used_ids = set()
            
            for position_type in [1, 2, 3, 4]:
                min_req, max_req = position_limits[position_type]
                pos_players = available_players[
                    (available_players['element_type'] == position_type) &
                    (~available_players['id'].isin(used_ids))
                ].copy()
                
                # Sort with exploration factor across attempts
                if attempt == 0:
                    pos_players = pos_players.sort_values('composite_score', ascending=False)
                elif attempt < 15:
                    pos_players = pos_players.sort_values('value_ratio', ascending=False)
                else:
                    # Blend of raw expected points and budget value
                    pos_players['blend_rank'] = pos_players['composite_score'] * (0.6 + (attempt % 5) * 0.1) + pos_players['value_ratio'] * 2.5
                    pos_players = pos_players.sort_values('blend_rank', ascending=False)
                
                selected_count = 0
                for _, player in pos_players.iterrows():
                    if selected_count >= max_req:
                        break
                    
                    p_team = int(player['team'])
                    if team_counts.get(p_team, 0) >= 3:
                        continue
                    
                    # Ensure remaining budget is sufficient for remaining slots (£4.0m per slot min)
                    slots_remaining = 15 - (len(squad) + 1)
                    min_cost_needed = slots_remaining * 40
                    
                    if (total_cost + player['now_cost'] + min_cost_needed) <= budget:
                        card = self.get_player_full_card(int(player['id']))
                        squad.append(card if card else {
                            'id': int(player['id']),
                            'web_name': str(player['web_name']),
                            'position_short': ['GKP', 'DEF', 'MID', 'FWD'][position_type-1],
                            'element_type': position_type,
                            'cost': float(player['now_cost']) / 10.0,
                            'predicted_5gw': float(player['predicted_5gw']),
                            'team_short': self.team_short_map.get(p_team, 'UNK'),
                            'expected_goals': float(player.get('expected_goals', 0) or 0),
                            'expected_assists': float(player.get('expected_assists', 0) or 0)
                        })
                        
                        total_cost += player['now_cost']
                        total_predicted += player['predicted_5gw']
                        selected_count += 1
                        position_counts[position_type] += 1
                        team_counts[p_team] = team_counts.get(p_team, 0) + 1
                        used_ids.add(player['id'])
            
            valid_squad = (len(squad) == 15 and 
                           all(position_counts[pos] == position_limits[pos][0] for pos in position_limits) and
                           total_cost <= budget)
            
            if valid_squad and total_predicted > best_score:
                best_score = total_predicted
                best_squad = {
                    'squad': squad,
                    'total_cost': round(total_cost / 10.0, 1),
                    'total_predicted': round(total_predicted, 1),
                    'remaining_budget': round((budget - total_cost) / 10.0, 1)
                }
        
        return best_squad
    
    def generate_wildcard_team(self, current_gw=None, team_id=None):
        if current_gw is None:
            current_gw = self.get_current_gameweek()
        
        budget = 1000
        if team_id:
            try:
                team_data = self.fetch_team_data(team_id)
                if team_data:
                    val = team_data.get('last_deadline_value')
                    bank = team_data.get('last_deadline_bank')
                    current_team_value = (float(val) / 10.0) if (val is not None and str(val) not in ('None', 'nan', '')) else 100.0
                    bank_balance = (float(bank) / 10.0) if (bank is not None and str(bank) not in ('None', 'nan', '')) else 0.0
                    budget = int((current_team_value + bank_balance) * 10)
            except Exception as e:
                print(f"Error fetching team details: {str(e)}")
        
        available_players = self.players_data[
            (self.players_data['status'] == 'a')
        ].copy()
        
        predictions_map = self.predict_all_players_points()
        
        for idx, (_, player) in enumerate(available_players.iterrows()):
            base_pts = predictions_map.get(player['id'], 2.0)
            diffs = self.get_fixture_difficulty_next_5gw(int(player['team']), current_gw)
            total_5gw = sum(max(0.0, round(base_pts * (0.8 + ((6 - d) / 5.0) * 0.4), 1)) for d in diffs)
            available_players.loc[available_players['id'] == player['id'], 'predicted_5gw'] = total_5gw
        
        position_limits = {1: (2, 2), 2: (5, 5), 3: (5, 5), 4: (3, 3)}
        
        best_team = None
        best_score = 0
        
        for attempt in range(25):
            team = []
            total_cost = 0
            total_5gw_predicted = 0
            position_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            team_counts = {}
            used_players = set()
            
            for position_type in [1, 2, 3, 4]:
                min_req, max_req = position_limits[position_type]
                pos_players = available_players[
                    (available_players['element_type'] == position_type) &
                    (~available_players['id'].isin(used_players))
                ].copy()
                
                pos_players['value_score'] = (
                    pos_players['predicted_5gw'] / (pos_players['now_cost'] / 10) * 
                    (1 + (pd.to_numeric(pos_players['form'], errors='coerce').fillna(0) / 10)) *
                    (1 + (pd.to_numeric(pos_players['expected_goal_involvements'], errors='coerce').fillna(0) / 15))
                )
                
                pos_players = pos_players.sort_values('value_score', ascending=False)
                
                selected_count = 0
                for _, player in pos_players.iterrows():
                    if selected_count >= max_req:
                        break
                    
                    p_team = int(player['team'])
                    if team_counts.get(p_team, 0) >= 3:
                        continue
                    
                    if total_cost + player['now_cost'] <= budget:
                        card = self.get_player_full_card(int(player['id']))
                        team.append(card if card else {
                            'id': int(player['id']),
                            'web_name': str(player['web_name']),
                            'position_short': ['GKP', 'DEF', 'MID', 'FWD'][position_type-1],
                            'element_type': position_type,
                            'cost': float(player['now_cost']) / 10.0,
                            'predicted_5gw': float(player['predicted_5gw']),
                            'team_short': self.team_short_map.get(p_team, 'UNK'),
                            'expected_goals': float(player.get('expected_goals', 0) or 0),
                            'expected_assists': float(player.get('expected_assists', 0) or 0)
                        })
                        
                        total_cost += player['now_cost']
                        total_5gw_predicted += player['predicted_5gw']
                        selected_count += 1
                        position_counts[position_type] += 1
                        team_counts[p_team] = team_counts.get(p_team, 0) + 1
                        used_players.add(player['id'])
            
            valid_team = (len(team) == 15 and 
                         all(position_counts[pos] >= position_limits[pos][0] for pos in position_limits))
            
            if valid_team and total_5gw_predicted > best_score:
                best_score = total_5gw_predicted
                best_team = {
                    'team': team,
                    'total_cost': round(float(total_cost) / 10.0, 1),
                    'total_5gw_predicted': round(float(total_5gw_predicted), 1),
                    'remaining_budget': round(float(budget - total_cost) / 10.0, 1)
                }
        
        return best_team
    
    def generate_free_hit_team(self, current_gw=None):
        if current_gw is None:
            current_gw = self.get_current_gameweek()
        
        budget = 1000
        available_players = self.players_data[
            (self.players_data['status'] == 'a')
        ].copy()
        
        predictions_map = self.predict_all_players_points()
        available_players['gw_prediction'] = available_players['id'].map(lambda pid: predictions_map.get(pid, 2.0))
        
        position_limits = {1: (1, 1), 2: (3, 5), 3: (3, 5), 4: (1, 3)}
        
        best_xi = None
        best_score = 0
        
        for attempt in range(25):
            starting_xi = []
            total_cost = 0
            total_predicted = 0
            position_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            team_counts = {}
            used_players = set()
            
            for position_type in [1, 2, 3, 4]:
                min_req, max_req = position_limits[position_type]
                pos_players = available_players[
                    (available_players['element_type'] == position_type) &
                    (~available_players['id'].isin(used_players))
                ].copy()
                
                pos_players['value_score'] = (
                    pos_players['gw_prediction'] / (pos_players['now_cost'] / 10.0) * 
                    (1 + (pd.to_numeric(pos_players['form'], errors='coerce').fillna(0) / 10.0)) *
                    (1 + (pd.to_numeric(pos_players['expected_goal_involvements'], errors='coerce').fillna(0) / 15.0))
                )
                
                pos_players = pos_players.sort_values('value_score', ascending=False)
                
                selected_count = 0
                for _, player in pos_players.iterrows():
                    if selected_count >= max_req:
                        break
                    
                    p_team = int(player['team'])
                    if team_counts.get(p_team, 0) >= 3:
                        continue
                    
                    if total_cost + player['now_cost'] <= budget:
                        card = self.get_player_full_card(int(player['id']))
                        starting_xi.append(card if card else {
                            'id': int(player['id']),
                            'web_name': str(player['web_name']),
                            'position_short': ['GKP', 'DEF', 'MID', 'FWD'][position_type-1],
                            'element_type': position_type,
                            'cost': float(player['now_cost']) / 10.0,
                            'gw_prediction': float(player['gw_prediction']),
                            'team_short': self.team_short_map.get(p_team, 'UNK'),
                            'expected_goals': float(player.get('expected_goals', 0) or 0),
                            'expected_assists': float(player.get('expected_assists', 0) or 0)
                        })
                        
                        total_cost += player['now_cost']
                        total_predicted += player['gw_prediction']
                        selected_count += 1
                        position_counts[position_type] += 1
                        team_counts[p_team] = team_counts.get(p_team, 0) + 1
                        used_players.add(player['id'])
            
            valid_xi = (len(starting_xi) == 11 and 
                       all(position_counts[pos] >= position_limits[pos][0] for pos in position_limits))
            
            if valid_xi and total_predicted > best_score:
                best_score = total_predicted
                best_xi = {
                    'starting_xi': starting_xi,
                    'total_cost': round(float(total_cost) / 10.0, 1),
                    'total_predicted': round(float(total_predicted), 1),
                    'formation': f"{position_counts[2]}-{position_counts[3]}-{position_counts[4]}"
                }
        
        return best_xi
    
    def analyze_team_composition(self, team_picks):
        analysis = {
            'formation': {},
            'total_value': 0.0,
            'predicted_points': 0.0,
            'captain_multiplier': 0.0,
            'vice_captain_backup': 0.0,
            'bench_value': 0.0,
            'formation_str': '3-4-3'
        }
        
        starting_xi = [p for p in team_picks['picks'][:11]]
        
        for pick in team_picks['picks']:
            try:
                player = self.players_data[self.players_data['id'] == pick['element']].iloc[0]
                position = self.bootstrap_data['element_types'][int(player['element_type']) - 1]['singular_name_short']
                
                cost = float(player['now_cost']) / 10.0
                analysis['total_value'] += cost
                
                if pick in starting_xi:
                    analysis['formation'][position] = analysis['formation'].get(position, 0) + 1
                    
                    prediction = self.predict_player_points(pick['element'])
                    if prediction:
                        points = float(prediction['predicted_points'])
                        if pick.get('is_captain'):
                            analysis['predicted_points'] += points * 2
                            analysis['captain_multiplier'] = points
                        elif pick.get('is_vice_captain'):
                            analysis['predicted_points'] += points
                            analysis['vice_captain_backup'] = points
                        else:
                            analysis['predicted_points'] += points
                else:
                    analysis['bench_value'] += cost
            except Exception as e:
                continue
        
        analysis['total_value'] = round(analysis['total_value'], 1)
        analysis['bench_value'] = round(analysis['bench_value'], 1)
        analysis['predicted_points'] = round(analysis['predicted_points'], 1)
        
        def_count = analysis['formation'].get('DEF', 3)
        mid_count = analysis['formation'].get('MID', 4)
        fwd_count = analysis['formation'].get('FWD', 3)
        analysis['formation_str'] = f"{def_count}-{mid_count}-{fwd_count}"
        
        return analysis
    
    def get_transfer_recommendations(self, team_picks, budget=0.0):
        recommendations = []
        current_player_ids = [p['element'] for p in team_picks['picks']]
        
        for position_type in range(1, 5):
            try:
                position_picks = [p for p in team_picks['picks'] 
                                  if int(self.players_data[self.players_data['id'] == p['element']].iloc[0]['element_type']) == position_type]
                
                for pick in position_picks:
                    current_card = self.get_player_full_card(pick['element'])
                    if not current_card:
                        continue
                    
                    max_price = (current_card['cost'] + budget) * 10
                    same_position = self.players_data[
                        (self.players_data['element_type'] == position_type) &
                        (~self.players_data['id'].isin(current_player_ids)) &
                        (self.players_data['now_cost'] <= max_price) &
                        (self.players_data['status'] == 'a')
                    ].sort_values('form', ascending=False)
                    
                    for _, alternative in same_position.head(4).iterrows():
                        alt_card = self.get_player_full_card(int(alternative['id']))
                        if not alt_card:
                            continue
                            
                        point_gain = alt_card['predicted_points'] - current_card['predicted_points']
                        cost_diff = round(alt_card['cost'] - current_card['cost'], 1)
                        xgi_gain = round(alt_card['expected_goal_involvements'] - current_card['expected_goal_involvements'], 2)
                        
                        if point_gain > 0.4 or (alt_card['form'] > current_card['form'] + 1.5):
                            reasons = []
                            if alt_card['form'] > current_card['form']:
                                reasons.append(f"Form advantage: {alt_card['form']} vs {current_card['form']}")
                            if xgi_gain > 0.5:
                                reasons.append(f"+{xgi_gain} xGI involvement upside")
                            if alt_card['xg_delta'] < -1.0:
                                reasons.append(f"xG Underperformer due for a massive haul")
                            if current_card.get('status') != 'a':
                                reasons.append(f"Replaces flagged/injured asset ({current_card.get('news', 'Flagged')})")
                            
                            recommendations.append({
                                'out_player': current_card,
                                'in_player': alt_card,
                                'position': current_card['position_short'],
                                'cost_change': cost_diff,
                                'point_gain': round(point_gain, 1),
                                'confidence': round((alt_card['confidence'] + current_card['confidence']) / 2, 1),
                                'xgi_gain': xgi_gain,
                                'reason': " • ".join(reasons) if reasons else "Superior form & fixture schedule"
                            })
            except Exception as e:
                print(f"Error processing transfers for position {position_type}: {str(e)}")
                continue
        
        return sorted(recommendations, key=lambda x: (x['point_gain'] + x['xgi_gain'] * 0.5), reverse=True)[:6]
    
    def get_captain_recommendations(self, team_picks):
        starting_xi = [p for p in team_picks['picks'][:11]]
        captain_options = []
        
        for pick in starting_xi:
            try:
                card = self.get_player_full_card(pick['element'])
                if not card:
                    continue
                
                fdr = self.get_fixture_difficulty(card['team_id'], self.get_current_gameweek())
                
                # Captaincy score combining ML prediction, form, xGI/90, and FDR
                xgi_p90 = card.get('expected_goal_involvements_per_90', 0.0)
                form_val = card.get('form', 0.0)
                pred_pts = card.get('predicted_points', 2.0)
                
                captain_score = (
                    pred_pts * 0.35 +
                    form_val * 0.25 +
                    (6 - fdr) * 0.20 +
                    (xgi_p90 * 5.0) * 0.20
                )
                
                captain_options.append({
                    'player': card,
                    'player_name': card['web_name'],
                    'player_id': card['id'],
                    'predicted_points': pred_pts,
                    'captain_score': round(captain_score, 2),
                    'form': form_val,
                    'fixture_difficulty': fdr,
                    'confidence': card['confidence'],
                    'xgi_per_90': xgi_p90
                })
            except Exception as e:
                continue
        
        return sorted(captain_options, key=lambda x: x['captain_score'], reverse=True)[:4]

analyzer = FPLAnalyzer()
from fpl_agent import FPLAutonomousAgent
ai_agent = FPLAutonomousAgent(analyzer=analyzer)

# ----------------- LLM Integration & AI Assistant Logic -----------------

DEFAULT_NVIDIA_API_KEY = "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"

def call_external_llm(messages, api_key=None, provider='auto', model=None):
    """
    Call NVIDIA NIM, Gemini, OpenAI, or Groq based on configuration.
    """
    nvidia_key = api_key if (api_key and api_key.startswith('nvapi-')) else (os.getenv('NVIDIA_API_KEY') or DEFAULT_NVIDIA_API_KEY)
    gemini_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    openai_key = api_key or os.getenv('OPENAI_API_KEY')
    groq_key = api_key or os.getenv('GROQ_API_KEY')
    
    # 1. NVIDIA NIM (Nemotron / Llama / DeepSeek on integrate.api.nvidia.com)
    if (provider in ('nvidia', 'nemotron', 'auto') and nvidia_key) or (provider == 'nvidia'):
        target_model = model or "nvidia/nemotron-3-ultra-550b-a55b"
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {nvidia_key}"
        }
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048
        }
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=25)
            if res.status_code == 200:
                data = res.json()
                if data.get('choices') and len(data['choices']) > 0:
                    choice = data['choices'][0]
                    content = choice.get('message', {}).get('content')
                    reasoning = choice.get('message', {}).get('reasoning_content')
                    if content and len(content.strip()) > 0:
                        return content
                    elif reasoning:
                        return reasoning
            else:
                print(f"NVIDIA API status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"Error calling NVIDIA API: {str(e)}")

    # 2. Google Gemini
    if (provider in ('gemini', 'auto') and gemini_key and (gemini_key.startswith('AIza') or provider == 'gemini')):
        target_model = model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={gemini_key}"
        
        # Convert chat messages to gemini format
        contents = []
        system_instruction = None
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = {"parts": [{"text": msg['content']}]}
            elif msg['role'] == 'user':
                contents.append({"role": "user", "parts": [{"text": msg['content']}]})
            elif msg['role'] == 'assistant':
                contents.append({"role": "model", "parts": [{"text": msg['content']}]})
                
        payload = {"contents": contents}
        if system_instruction:
            payload["system_instruction"] = system_instruction
            
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=20)
            if res.status_code == 200:
                data = res.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"Gemini API returned {res.status_code}: {res.text}")
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")

    # 3. OpenAI
    if (provider in ('openai', 'auto') and openai_key and (openai_key.startswith('sk-') or provider == 'openai')):
        target_model = model or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}"
        }
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": 0.7
        }
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=20)
            if res.status_code == 200:
                data = res.json()
                return data['choices'][0]['message']['content']
            else:
                print(f"OpenAI API returned {res.status_code}: {res.text}")
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")

    # 4. Groq
    if (provider in ('groq', 'auto') and groq_key and (groq_key.startswith('gsk_') or provider == 'groq')):
        target_model = model or "llama-3.1-70b-versatile"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {groq_key}"
        }
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": 0.7
        }
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=20)
            if res.status_code == 200:
                data = res.json()
                return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            
    return None

def generate_domain_fpl_analysis(user_query, team_data=None, team_picks=None, current_gw=1):
    """
    Built-in High-Intelligence Football Analytics Engine.
    Generates structured, statistical answers using actual xG, xA, xGI, Form, Price, and FDR data.
    """
    q_lower = user_query.lower()
    
    # Pre-gather context
    all_players = analyzer.players_data
    if all_players is None or all_players.empty:
        return "⚠️ FPL dataset is still loading. Please try again in a few seconds."
        
    current_squad_cards = []
    bank_money = 0.0
    if team_picks and 'picks' in team_picks:
        for p in team_picks['picks']:
            c = analyzer.get_player_full_card(p['element'])
            if c:
                c['is_captain'] = p.get('is_captain', False)
                c['is_vice_captain'] = p.get('is_vice_captain', False)
                current_squad_cards.append(c)
    if team_data:
        b_raw = team_data.get('last_deadline_bank')
        bank_money = (float(b_raw) / 10.0) if (b_raw is not None and str(b_raw) not in ('None', 'nan', '')) else 0.0
        
    # 1. Captaincy Queries
    if any(k in q_lower for k in ['captain', 'armband', 'who to captain', 'c pick', 'vice captain']):
        if team_picks and current_squad_cards:
            cap_recs = analyzer.get_captain_recommendations(team_picks)
            if cap_recs:
                top = cap_recs[0]['player']
                second = cap_recs[1]['player'] if len(cap_recs) > 1 else None
                
                resp = f"### 👑 FPL Captaincy Intelligence (GW{current_gw})\n\n"
                resp += f"Based on machine learning points projections, recent form, underlying **xG/xA threat**, and fixture difficulty:\n\n"
                resp += f"#### 🥇 **Primary Captain**: **{top['web_name']}** ({top['team_short']} - £{top['cost']}M)\n"
                resp += f"- **Projected GW Points**: `{top['predicted_points']} pts` (Confidence: `{top['confidence']}%`)\n"
                resp += f"- **Expected Goals (xG)**: `{top['expected_goals']}` | **Expected Assists (xA)**: `{top['expected_assists']}`\n"
                resp += f"- **xGI per 90 mins**: `{top['expected_goal_involvements_per_90']}` (Elite goal involvement rate)\n"
                resp += f"- **Recent Form**: `{top['form']}` points/match\n"
                if top['fixtures_next_5']:
                    opp = top['fixtures_next_5'][0]
                    resp += f"- **Fixture**: vs **{opp['opponent']}** (FDR Rating: `{opp['difficulty']}/5`)\n\n"
                
                if second:
                    resp += f"#### 🥈 **Vice-Captain / Differential Pick**: **{second['web_name']}** ({second['team_short']} - £{second['cost']}M)\n"
                    resp += f"- **Projected Points**: `{second['predicted_points']} pts` | **Form**: `{second['form']}` | **xGI/90**: `{second['expected_goal_involvements_per_90']}`\n"
                    if second['fixtures_next_5']:
                        opp2 = second['fixtures_next_5'][0]
                        resp += f"- **Fixture**: vs **{opp2['opponent']}** (FDR: `{opp2['difficulty']}/5`)\n\n"
                
                resp += f"> **Tactical Verdict**: Give the armband to **{top['web_name']}**. Their underlying xGI numbers and attacking volume give them the highest theoretical ceiling this gameweek."
                return resp

    # 2. Underperforming xG / Hidden Gems / Differentials
    if any(k in q_lower for k in ['underperform', 'regression', 'hidden gem', 'differential', 'due a goal', 'burst', 'haul', 'stats']):
        cards = []
        top_candidates = analyzer.players_data[
            (analyzer.players_data['status'] == 'a') &
            (pd.to_numeric(analyzer.players_data['minutes'], errors='coerce') > 270) &
            (pd.to_numeric(analyzer.players_data['expected_goals'], errors='coerce') > 1.5)
        ].copy()
        
        for _, r in top_candidates.iterrows():
            c = analyzer.get_player_full_card(int(r['id']))
            if c:
                cards.append(c)
                
        underperformers = sorted(cards, key=lambda x: x['xg_delta'])[:5]
        
        resp = f"### 🎯 Top xG Underperformers (Due for Massive Hauls)\n\n"
        resp += f"In football analytics, players with high **Expected Goals (xG)** but fewer actual goals are creating elite chances and are statistically primed for explosive positive regression:\n\n"
        
        for p in underperformers:
            resp += f"- **{p['web_name']}** ({p['team_short']} • £{p['cost']}M • {p['position_short']})\n"
            resp += f"  - **xG**: `{p['expected_goals']}` vs **Actual Goals**: `{p['goals_scored']}` (Deficit: `{-p['xg_delta']} xG`)\n"
            resp += f"  - **xGI / 90**: `{p['expected_goal_involvements_per_90']}` | **Ownership**: `{p['selected_by_percent']}%`\n"
            if p['fixtures_next_5']:
                resp += f"  - **Next 3 Fixtures**: {' → '.join([f.get('opponent', '') for f in p['fixtures_next_5'][:3]])}\n\n"
                
        resp += f"> **Strategy Tip**: Bringing in an xG underperformer with low ownership before the rest of the FPL template catches on is the single fastest way to gain green arrows."
        return resp

    # 3. Transfer Recommendations
    if any(k in q_lower for k in ['transfer', 'sell', 'buy', 'replace', 'swap', 'upgrade', 'fix']):
        if team_picks:
            recs = analyzer.get_transfer_recommendations(team_picks, budget=bank_money)
            if recs:
                resp = f"### 🔄 Top AI Transfer Recommendations (Bank: £{bank_money:.1f}M)\n\n"
                resp += f"Here are the highest expected-value transfer moves calculated from predictive ML models, xG/xA output, and upcoming fixture swings:\n\n"
                
                for idx, r in enumerate(recs[:4], 1):
                    out_p = r['out_player']
                    in_p = r['in_player']
                    cost_sign = f"+£{r['cost_change']}M" if r['cost_change'] > 0 else f"-£{-r['cost_change']}M"
                    
                    resp += f"#### Option {idx}: **Sell {out_p['web_name']}** ➔ **Buy {in_p['web_name']}**\n"
                    resp += f"- **Position**: `{r['position']}` | **Cost Delta**: `{cost_sign}` | **Projected Gain**: `+{r['point_gain']} pts`\n"
                    resp += f"- **xGI Comparison**: {in_p['web_name']} (`{in_p['expected_goal_involvements']}`) vs {out_p['web_name']} (`{out_p['expected_goal_involvements']}`)\n"
                    resp += f"- **Form**: `{in_p['form']}` vs `{out_p['form']}`\n"
                    resp += f"- **Tactical Rationale**: {r['reason']}\n\n"
                    
                resp += f"> **Actionable Next Step**: If you have 1 Free Transfer, executing **Option 1** offers the best risk-adjusted point gain for the upcoming gameweek."
                return resp

    # 4. Rate My Team (RMT) / Squad Evaluation
    if any(k in q_lower for k in ['rate', 'rmt', 'review', 'my team', 'squad', 'weakness', 'evaluate']) and current_squad_cards:
        total_pts = sum(c['predicted_points'] for c in current_squad_cards[:11])
        injured_flagged = [c for c in current_squad_cards if c.get('status') != 'a']
        low_xgi = [c for c in current_squad_cards[:11] if c['element_type'] in (3, 4) and c['expected_goal_involvements'] < 1.0]
        
        resp = f"### 📋 Comprehensive FPL Squad Audit (GW{current_gw})\n\n"
        resp += f"- **Starting XI Predicted Points**: `{round(total_pts, 1)} pts`\n"
        resp += f"- **Bank Balance**: `£{bank_money:.1f}M`\n\n"
        
        if injured_flagged:
            resp += f"#### ⚠️ **Urgent Attention (Flags/Injuries)**\n"
            for p in injured_flagged:
                resp += f"- **{p['web_name']}** ({p['team_short']}): `{p.get('news', 'Flagged for injury/suspension')}`\n"
            resp += "\n"
            
        resp += f"#### 🌟 **Key Strengths (High xGI Drivers)**\n"
        sorted_xgi = sorted([c for c in current_squad_cards if c['element_type'] in (3, 4)], key=lambda x: x['expected_goal_involvements'], reverse=True)[:3]
        for p in sorted_xgi:
            resp += f"- **{p['web_name']}** ({p['team_short']}): `{p['expected_goal_involvements']} xGI` | `{p['form']} Form` | `{p['predicted_points']} pts` projected\n"
        resp += "\n"
        
        if low_xgi:
            resp += f"#### 📉 **Vulnerable Assets (Low Underlying Output)**\n"
            for p in low_xgi[:3]:
                resp += f"- **{p['web_name']}** ({p['team_short']} • £{p['cost']}M): Only `{p['expected_goal_involvements']} xGI` this season. Consider transferring out.\n"
            resp += "\n"
            
        resp += f"> **Overall Rating**: **8.5 / 10**. Your starting core is competitive. Resolve any flagged assets first to avoid dead bench points."
        return resp

    # 5. Player Comparison (e.g. Saka vs Palmer, Haaland vs Salah)
    names_found = []
    for _, p in analyzer.players_data.iterrows():
        w_name = str(p['web_name']).lower()
        f_name = str(p.get('second_name', '')).lower()
        if (len(w_name) > 3 and w_name in q_lower) or (len(f_name) > 3 and f_name in q_lower):
            card = analyzer.get_player_full_card(int(p['id']))
            if card and card not in names_found:
                names_found.append(card)
                if len(names_found) >= 2:
                    break
                    
    if len(names_found) >= 2:
        p1, p2 = names_found[0], names_found[1]
        winner = p1 if (p1['predicted_points'] + p1['expected_goal_involvements_per_90']*3) > (p2['predicted_points'] + p2['expected_goal_involvements_per_90']*3) else p2
        
        resp = f"### ⚔️ Player Head-to-Head: **{p1['web_name']}** vs **{p2['web_name']}**\n\n"
        resp += f"| Metric | **{p1['web_name']}** ({p1['team_short']}) | **{p2['web_name']}** ({p2['team_short']}) |\n"
        resp += f"| :--- | :--- | :--- |\n"
        resp += f"| **Price** | £{p1['cost']}M | £{p2['cost']}M |\n"
        resp += f"| **Predicted Points (GW{current_gw})** | **`{p1['predicted_points']} pts`** | **`{p2['predicted_points']} pts`** |\n"
        resp += f"| **Expected Goals (xG)** | `{p1['expected_goals']}` | `{p2['expected_goals']}` |\n"
        resp += f"| **Expected Assists (xA)** | `{p1['expected_assists']}` | `{p2['expected_assists']}` |\n"
        resp += f"| **xGI per 90 mins** | `{p1['expected_goal_involvements_per_90']}` | `{p2['expected_goal_involvements_per_90']}` |\n"
        resp += f"| **Recent Form** | `{p1['form']}` | `{p2['form']}` |\n"
        resp += f"| **ICT Index (Threat/Creativity)** | `{p1['ict_index']}` | `{p2['ict_index']}` |\n"
        resp += f"| **Ownership %** | `{p1['selected_by_percent']}%` | `{p2['selected_by_percent']}%` |\n\n"
        
        resp += f"> **AI Verdict**: **{winner['web_name']}** edges the comparison with superior underlying per-90 metrics and attacking volume."
        return resp

    # 6. Default Tactical Overview
    top_xg_players = analyzer.players_data.sort_values('expected_goal_involvements', ascending=False).head(5)
    resp = f"### 🤖 FPL AI Tactical Assistant (GW{current_gw})\n\n"
    resp += f"Here is the latest Premier League attacking intelligence across the league:\n\n"
    resp += f"**Current xGI (Expected Goal Involvement) League Leaders:**\n"
    for _, p in top_xg_players.iterrows():
        resp += f"- **{p['web_name']}** ({analyzer.team_short_map.get(int(p['team']), '')}): `{p.get('expected_goal_involvements', 0)} xGI` (xG: `{p.get('expected_goals', 0)}`, xA: `{p.get('expected_assists', 0)}`) • £{float(p['now_cost'])/10}M\n"
    resp += f"\n💡 *You can ask me specific questions like:* \n"
    resp += f"- *\"Who should I transfer out of my squad?\"*\n"
    resp += f"- *\"Who is the safest captain for this week?\"*\n"
    resp += f"- *\"Compare Saka vs Palmer using xG and xA stats\"*\n"
    resp += f"- *\"Show me underperforming players with high xG under 7.5m\"*\n"
    
    return resp

# ----------------- Flask Routes -----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json() or {}
        team_id = data.get('team_id')
        gameweek = data.get('gameweek')
        
        if not team_id:
            return jsonify({'error': 'Team ID is required'}), 400
        
        if not analyzer.fetch_bootstrap_data():
            return jsonify({'error': 'Failed to fetch FPL bootstrap data'}), 500
        
        analyzer.fetch_fixtures_data()
        
        team_data = analyzer.fetch_team_data(team_id)
        if not team_data:
            return jsonify({'error': f'Team with ID {team_id} not found'}), 404
        
        current_gw = int(gameweek) if gameweek else analyzer.get_current_gameweek()
        next_gw = analyzer.get_next_gameweek()
        team_picks = analyzer.fetch_team_picks(team_id, current_gw)
        
        if not team_picks:
            for gw_try in range(current_gw - 1, 0, -1):
                team_picks = analyzer.fetch_team_picks(team_id, gw_try)
                if team_picks:
                    break
            if not team_picks:
                return jsonify({'error': f'Team picks for Team ID {team_id} not found'}), 404
        
        analyzer.train_models()
        
        composition = analyzer.analyze_team_composition(team_picks)
        captain_recs = analyzer.get_captain_recommendations(team_picks)
        
        bank_raw = team_data.get('last_deadline_bank')
        bank_budget = (float(bank_raw) / 10.0) if (bank_raw is not None and str(bank_raw) not in ('None', 'nan', '')) else 0.0
        val_raw = team_data.get('last_deadline_value')
        team_value = (float(val_raw) / 10.0) if (val_raw is not None and str(val_raw) not in ('None', 'nan', '')) else 100.0
        transfer_recs = analyzer.get_transfer_recommendations(team_picks, budget=bank_budget)
        
        # Build complete 15-player cards (starting XI + bench)
        squad_cards = []
        for idx, pick in enumerate(team_picks.get('picks', [])):
            card = analyzer.get_player_full_card(pick['element'])
            if card:
                card['is_captain'] = pick.get('is_captain', False)
                card['is_vice_captain'] = pick.get('is_vice_captain', False)
                card['multiplier'] = pick.get('multiplier', 1)
                card['position_order'] = pick.get('position', idx + 1)
                card['is_starting'] = (idx < 11)
                squad_cards.append(card)
        
        # Optimize lineup with AI model
        lineup_opt = ai_agent.optimize_lineup_and_captain(squad_cards, next_gw)
        starting_xi = lineup_opt.get('starting_xi') or [p for p in squad_cards if p.get('is_starting')]
        bench = lineup_opt.get('bench') or [p for p in squad_cards if not p.get('is_starting')]
        top_captain = lineup_opt.get('captain') or (captain_recs[0]['player'] if captain_recs else None)
        top_vice = lineup_opt.get('vice_captain') or (captain_recs[1]['player'] if len(captain_recs) > 1 else None)
        
        # Fetch chips history
        hist = analyzer.fetch_entry_history(team_id)
        chips_used = [c.get('name') for c in hist.get('chips', [])] if hist else []
        chip_advice = ai_agent.evaluate_chip_strategy(squad_cards, next_gw, chips_used, top_captain, bench)
        
        # Calculate pre-deadline transfer candidates with % probabilities
        pre_deadline_transfers = ai_agent.get_pre_deadline_analysis(squad_cards, bank_budget, 1, next_gw)
        
        return jsonify(_sanitize({
            'team_id': team_id,
            'team_name': team_data.get('name', f'Team {team_id}'),
            'manager_name': f"{team_data.get('player_first_name', '')} {team_data.get('player_last_name', '')}".strip(),
            'overall_points': team_data.get('summary_overall_points', 0),
            'overall_rank': team_data.get('summary_overall_rank', 0),
            'gw_points': team_data.get('summary_event_points', 0),
            'team_data': team_data,
            'composition': composition,
            'starting_xi': starting_xi,
            'bench': bench,
            'squad': squad_cards,
            'captain': top_captain,
            'vice_captain': top_vice,
            'captain_recommendations': captain_recs,
            'transfer_recommendations': transfer_recs,
            'pre_deadline_transfers': pre_deadline_transfers,
            'chips_used': chips_used,
            'chip_advice': chip_advice,
            'current_gameweek': current_gw,
            'next_gw': next_gw,
            'bank': bank_budget,
            'bank_balance': bank_budget,
            'team_value': team_value
        }))


        
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get('message', '').strip()
        team_id = data.get('team_id')
        gameweek = data.get('gameweek')
        api_key = data.get('api_key')
        provider = data.get('provider', 'auto')
        model = data.get('model')
        chat_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
            
        if not analyzer.bootstrap_data:
            analyzer.fetch_bootstrap_data()
            analyzer.fetch_fixtures_data()
            
        current_gw = int(gameweek) if gameweek else analyzer.get_current_gameweek()
        
        # Fetch team picks context if team_id provided
        team_data = None
        team_picks = None
        if team_id:
            team_data = analyzer.fetch_team_data(team_id)
            team_picks = analyzer.fetch_team_picks(team_id, current_gw)
            
        # Try external frontier LLM (NVIDIA Nemotron / Gemini / OpenAI / Groq)
        llm_response = None
        has_llm_key = bool(api_key or DEFAULT_NVIDIA_API_KEY or os.getenv('NVIDIA_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('GROQ_API_KEY'))
        if has_llm_key:
            # Build structured context for the prompt
            context_summary = f"Current FPL Gameweek: GW{current_gw}.\n"
            if team_picks and 'picks' in team_picks:
                squad_names = []
                for p in team_picks['picks']:
                    c = analyzer.get_player_full_card(p['element'])
                    if c:
                        cap_tag = " (C)" if p.get('is_captain') else (" (VC)" if p.get('is_vice_captain') else "")
                        squad_names.append(f"{c['web_name']} ({c['team_short']} £{c['cost']}M, xG: {c['expected_goals']}, xA: {c['expected_assists']}){cap_tag}")
                context_summary += f"User's Squad: {', '.join(squad_names)}.\n"
            if team_data:
                b_raw = team_data.get('last_deadline_bank')
                b_val = (float(b_raw) / 10.0) if (b_raw is not None and str(b_raw) not in ('None', 'nan', '')) else 0.0
                v_raw = team_data.get('last_deadline_value')
                t_val = (float(v_raw) / 10.0) if (v_raw is not None and str(v_raw) not in ('None', 'nan', '')) else 100.0
                context_summary += f"Bank Budget: £{b_val:.1f}M, Total Value: £{t_val:.1f}M.\n"
                
            system_prompt = (
                "You are FPL AI Oracle, a world-class Fantasy Premier League analyst and data scientist. "
                "You specialize in using Expected Goals (xG), Expected Assists (xA), Expected Goal Involvements (xGI), "
                "Fixture Difficulty Ratings (FDR), and form to give elite transfer advice, captain choices, and squad fixes. "
                "Always format your response with clean Markdown headers, bullet points, and highlight specific stats. "
                f"\nContext:\n{context_summary}"
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            for h in chat_history[-6:]: # Last 6 messages
                messages.append({"role": h.get('role', 'user'), "content": h.get('content', '')})
            messages.append({"role": "user", "content": user_message})
            
            try:
                llm_response = call_external_llm(messages, api_key=api_key, provider=provider, model=model)
            except Exception as e:
                print(f"External LLM call failed: {str(e)}")
                
        # If no external LLM response or no key, use our built-in statistical engine
        if not llm_response:
            llm_response = generate_domain_fpl_analysis(user_message, team_data, team_picks, current_gw)
            
        return jsonify({
            'response': llm_response,
            'gameweek': current_gw,
            'source': 'frontier_llm' if llm_response else 'fpl_neural_engine'
        })
        
    except Exception as e:
        print(f"Error in /chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats/xg_leaders', methods=['GET', 'POST'])
def xg_leaders():
    try:
        if not analyzer.bootstrap_data:
            analyzer.fetch_bootstrap_data()
            analyzer.fetch_fixtures_data()
            
        data = request.get_json() or {}
        position = data.get('position') # 1, 2, 3, 4 or None
        max_cost = float(data.get('max_cost', 20.0))
        sort_by = data.get('sort_by', 'expected_goal_involvements')
        
        players_df = analyzer.players_data.copy()
        if position:
            players_df = players_df[players_df['element_type'] == int(position)]
        players_df = players_df[players_df['now_cost'] <= max_cost * 10]
        
        # Convert numeric fields
        for col in ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded', 
                    'expected_goals_per_90', 'expected_assists_per_90', 'expected_goal_involvements_per_90', 'form', 'total_points', 'now_cost']:
            if col in players_df.columns:
                players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0.0)
                
        if sort_by in players_df.columns:
            sorted_df = players_df.sort_values(sort_by, ascending=False).head(30)
        else:
            sorted_df = players_df.sort_values('expected_goal_involvements', ascending=False).head(30)
            
        cards = []
        for _, p in sorted_df.iterrows():
            c = analyzer.get_player_full_card(int(p['id']))
            if c:
                cards.append(c)
                
        return jsonify({'leaders': cards})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_players():
    try:
        if not analyzer.bootstrap_data:
            analyzer.fetch_bootstrap_data()
            analyzer.fetch_fixtures_data()
            
        data = request.get_json() or {}
        player_ids = data.get('player_ids', [])
        
        if not player_ids:
            return jsonify({'error': 'Player IDs required'}), 400
            
        cards = []
        for pid in player_ids[:4]:
            c = analyzer.get_player_full_card(int(pid))
            if c:
                cards.append(c)
                
        return jsonify({'players': cards})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/wildcard', methods=['POST'])
def wildcard_team():
    try:
        gameweek = request.json.get('gameweek')
        team_id = request.json.get('team_id')
        
        if not analyzer.fetch_bootstrap_data():
            return jsonify({'error': 'Failed to fetch FPL data'}), 500
        
        analyzer.fetch_fixtures_data()
        analyzer.train_models()
        
        current_gw = int(gameweek) if gameweek else analyzer.get_current_gameweek()
        wildcard_team = analyzer.generate_wildcard_team(current_gw, team_id)
        
        if not wildcard_team:
            return jsonify({'error': 'Failed to generate wildcard team'}), 500
        
        return jsonify(wildcard_team)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/free_hit', methods=['POST'])
def free_hit_team():
    try:
        gameweek = request.json.get('gameweek')
        
        if not analyzer.fetch_bootstrap_data():
            return jsonify({'error': 'Failed to fetch FPL data'}), 500
        
        analyzer.fetch_fixtures_data()
        analyzer.train_models()
        
        current_gw = int(gameweek) if gameweek else analyzer.get_current_gameweek()
        free_hit_team = analyzer.generate_free_hit_team(current_gw)
        
        if not free_hit_team:
            return jsonify({'error': 'Failed to generate free hit team'}), 500
        
        return jsonify(free_hit_team)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimal_squad', methods=['POST'])
def optimal_squad():
    try:
        budget = float(request.json.get('budget', 100))
        gameweek = request.json.get('gameweek')
        
        if not analyzer.fetch_bootstrap_data():
            return jsonify({'error': 'Failed to fetch FPL data'}), 500
        
        analyzer.fetch_fixtures_data()
        analyzer.train_models()
        
        current_gw = int(gameweek) if gameweek else analyzer.get_current_gameweek()
        optimal_team = analyzer.generate_optimal_squad(budget * 10, current_gw)
        
        if not optimal_team:
            return jsonify({'error': 'Failed to generate optimal squad'}), 500
        
        return jsonify(optimal_team)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/league', methods=['POST'])
def league_analysis():
    try:
        league_id = request.json.get('league_id')
        page = request.json.get('page', 1)
        
        if not league_id:
            return jsonify({'error': 'League ID is required'}), 400
        
        league_data = analyzer.fetch_classic_league(league_id, page)
        
        if not league_data:
            return jsonify({'error': 'League not found'}), 404
        
        return jsonify(league_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------- Autonomous AI Manager Routes -----------------

@app.route('/api/ai-manager/status', methods=['GET'])
def ai_manager_status():
    try:
        status = ai_agent.get_account_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-manager/connect', methods=['POST'])
def ai_manager_connect():
    try:
        data = request.get_json() or {}
        email = data.get('email')
        password = data.get('password')
        team_id = data.get('team_id')
        
        if team_id:
            ai_agent.team_id = str(team_id)
            ai_agent._save_cached_session()
            
        res = ai_agent.authenticate_with_browser(email=email, password=password)
        return jsonify(res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai-manager/settings', methods=['POST'])
def ai_manager_save_settings():
    try:
        data = request.get_json() or {}
        success = ai_agent.save_settings(data)
        return jsonify({'success': success, 'settings': ai_agent.settings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-manager/run-cycle', methods=['POST'])
def ai_manager_run_cycle():
    try:
        data = request.get_json() or {}
        dry_run = data.get('dry_run', ai_agent.settings.get('dry_run', True))
        max_hit = data.get('max_hits', ai_agent.settings.get('max_hits', -4))
        force_squad = data.get('force_squad', False)
        
        result = ai_agent.run_gameweek_cycle(dry_run=dry_run, max_hit=max_hit, force_squad=force_squad)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-manager/draft-squad', methods=['POST'])
def ai_manager_draft_squad():
    try:
        data = request.get_json() or {}
        budget = float(data.get('budget', 100.0))
        target_gw = data.get('gameweek')
        
        result = ai_agent.draft_initial_squad(budget=budget, target_gw=target_gw)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai-manager/logs', methods=['GET'])
def ai_manager_get_logs():
    try:
        return jsonify({'logs': ai_agent.logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predicted-players', methods=['GET'])
def get_predicted_players():
    try:
        if analyzer.players_data is None or analyzer.players_data.empty:
            analyzer.fetch_bootstrap_data()
            analyzer.fetch_fixtures_data()
            analyzer.train_models()
            
        pos_filter = request.args.get('position', '').upper()
        search_query = request.args.get('search', '').lower().strip()
        sort_by = request.args.get('sort', 'xp_next')
        
        preds_map = analyzer.predict_all_players_points()
        current_gw = analyzer.get_current_gameweek()
        
        results = []
        for _, p in analyzer.players_data.iterrows():
            pid = int(p['id'])
            pos_names = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            pos_short = pos_names.get(int(p['element_type']), 'MID')
            
            if pos_filter and pos_short != pos_filter:
                continue
                
            web_name = str(p['web_name'])
            team_short = analyzer.team_short_map.get(int(p['team']), 'UNK')
            team_name = analyzer.team_name_map.get(int(p['team']), 'Unknown')
            
            if search_query:
                full_search = f"{web_name} {p.get('first_name', '')} {p.get('second_name', '')} {team_short} {team_name}".lower()
                if search_query not in full_search:
                    continue
                    
            xp_next = preds_map.get(pid, 2.0)
            cost = float(p['now_cost']) / 10.0
            fixtures_5 = analyzer.get_fixture_details_next_5gw(int(p['team']), current_gw)
            
            # Compute 5-GW estimated xP sum based on fixture difficulties
            xp_5gw = 0.0
            for i, f in enumerate(fixtures_5):
                diff = f.get('difficulty', 3)
                diff_mult = 1.3 if diff <= 2 else (1.0 if diff == 3 else (0.75 if diff == 4 else 0.55))
                xp_5gw += round(xp_next * diff_mult, 1)
                
            results.append({
                'id': pid,
                'web_name': web_name,
                'full_name': f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
                'team_short': team_short,
                'team_name': team_name,
                'position_short': pos_short,
                'cost': cost,
                'form': float(p.get('form', 0.0) or 0.0),
                'total_points': int(p.get('total_points', 0) or 0),
                'selected_by_percent': float(p.get('selected_by_percent', 0.0) or 0.0),
                'status': str(p.get('status', 'a')),
                'chance_of_playing_next_round': p.get('chance_of_playing_next_round'),
                'news': str(p.get('news', '')),
                'predicted_points': xp_next,
                'predicted_5gw': round(xp_5gw, 1),
                'value_rating': round(xp_next / max(cost, 4.0), 2),
                'fixtures_next_5': fixtures_5
            })
            
        if sort_by == 'cost':
            results.sort(key=lambda x: x['cost'], reverse=True)
        elif sort_by == 'form':
            results.sort(key=lambda x: x['form'], reverse=True)
        elif sort_by == 'value':
            results.sort(key=lambda x: x['value_rating'], reverse=True)
        elif sort_by == '5gw':
            results.sort(key=lambda x: x['predicted_5gw'], reverse=True)
        else:
            results.sort(key=lambda x: x['predicted_points'], reverse=True)
            
        return jsonify(_sanitize({'players': results[:60], 'total': len(results)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player/<int:player_id>', methods=['GET'])
def get_player_details(player_id):
    try:
        card = analyzer.get_player_full_card(player_id)
        if not card:
            return jsonify({'error': 'Player not found'}), 404
        return jsonify(_sanitize(card))
    except Exception as e:
        return jsonify({'error': str(e)}), 500



def preload_data():
    try:
        analyzer.fetch_bootstrap_data()
        analyzer.fetch_fixtures_data()
    except Exception as e:
        print(f"Preload info: {e}")

if __name__ == '__main__':
    preload_data()
    app.run(debug=True, port=5000, threaded=True)