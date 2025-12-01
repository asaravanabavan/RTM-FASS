import numpy as np
from collections import defaultdict

class FightAnalyzer:
    def __init__(self, rounds=5, round_duration=180):
        self.rounds = rounds
        self.round_duration = round_duration
        
        self.scoring_weights = {
            'strikes': 0.4, #landed strikes most important
            'aggression': 0.25, #ring control and forward pressure
            'defense': 0.15, #blocks and evasions
            'technique': 0.2 #variety and execution
        }
        
        self.strike_effectiveness = {
            'blocked': 0.2, #minimal points for blocked strikes
            'partially_blocked': 0.5, #reduced points
            'clean': 1.0, #full value
            'counter': 1.2, #bonus for counters
            'knockdown': 2.0 #significant bonus for knockdowns
        }
        
        self.initialize_stats()
        
        self.current_round = 1
        self.round_start_frame = 0
        self.current_frame = 0
        self.frames_per_second = 30
    
    def initialize_stats(self):
        self.fight_stats = {
            0: self._create_fighter_stats(),
            1: self._create_fighter_stats()
        }
        
        self.round_scores = {
            round_number: {0: 0, 1: 0} for round_number in range(1, self.rounds + 1)
        }
        
        self.fight_progression = []
    
    def _create_fighter_stats(self):
        return {
            'strikes': {
                'total_attempted': 0,
                'total_landed': 0,
                'by_type': defaultdict(lambda: {'attempted': 0, 'landed': 0})
            },
            'defense': {
                'strikes_blocked': 0,
                'strikes_evaded': 0,
                'total_incoming': 0
            },
            'movement': {
                'distance_traveled': 0,
                'direction_changes': 0,
                'center_control_time': 0
            },
            'fatigue': {
                'current_level': 0,
                'recovery_rate': 100
            },
            'score': 0
        }
    
    def set_video_params(self, frames_per_second, total_frames):
        self.frames_per_second = frames_per_second
        self.frames_per_round = int(self.round_duration * frames_per_second)
        self.total_frames = total_frames
    
    def update_round(self, frame_index):
        self.current_frame = frame_index
        
        elapsed_seconds = frame_index / self.frames_per_second
        current_round = int(elapsed_seconds / self.round_duration) + 1
        
        if current_round > self.current_round and current_round <= self.rounds:
            self._end_round(self.current_round) #score the completed round
            self.current_round = current_round
            self.round_start_frame = frame_index
    
    def _end_round(self, round_number):
        if round_number > self.rounds:
            return
            
        for fighter_id in [0, 1]:
            strikes_score = self._calculate_strikes_score(fighter_id)
            aggression_score = self._calculate_aggression_score(fighter_id)
            defense_score = self._calculate_defense_score(fighter_id)
            technique_score = self._calculate_technique_score(fighter_id)
            
            weighted_score = (
                strikes_score * self.scoring_weights['strikes'] +
                aggression_score * self.scoring_weights['aggression'] +
                defense_score * self.scoring_weights['defense'] +
                technique_score * self.scoring_weights['technique']
            )
            
            self.round_scores[round_number][fighter_id] = min(10, max(7, weighted_score)) #10-point must system
            
        self._update_fatigue_between_rounds()
    
    def _calculate_strikes_score(self, fighter_id):
        stats = self.fight_stats[fighter_id]
        
        if stats['strikes']['total_attempted'] == 0:
            return 7.0 #default score for no activity
            
        accuracy = (stats['strikes']['total_landed'] / 
                   max(1, stats['strikes']['total_attempted']))
        
        weighted_strikes = 0
        total_weights = 0
        
        for strike_type, counts in stats['strikes']['by_type'].items():
            if 'kick' in strike_type:
                weight = 1.5 #kicks worth more
            elif 'knee' in strike_type or 'elbow' in strike_type:
                weight = 1.3 #knees/elbows worth more than punches
            else:
                weight = 1.0 #baseline for punches
                
            weighted_strikes += counts['landed'] * weight
            total_weights += counts['attempted'] * weight
        
        weighted_accuracy = weighted_strikes / max(1, total_weights)
        
        score = 7 + (weighted_accuracy * 3) #scale from 7-10
        
        return score
    
    def _calculate_aggression_score(self, fighter_id):
        stats = self.fight_stats[fighter_id]
        
        center_control_percentage = (stats['movement']['center_control_time'] / 
                                    max(1, self.current_frame / self.frames_per_second))
        
        strike_attempt_rate = (stats['strikes']['total_attempted'] / 
                              max(1, self.current_frame / self.frames_per_second))
        
        control_score = center_control_percentage * 5 #reward ring control
        pressure_score = min(5, strike_attempt_rate / 2) #reward offensive output
        
        return control_score + pressure_score
    
    def _calculate_defense_score(self, fighter_id):
        stats = self.fight_stats[fighter_id]
        
        if stats['defense']['total_incoming'] == 0:
            return 8.5 #good defense if not attacked
            
        blocks_and_evades = (stats['defense']['strikes_blocked'] + 
                            stats['defense']['strikes_evaded'])
        
        defense_percentage = blocks_and_evades / max(1, stats['defense']['total_incoming'])
        
        return 7 + (defense_percentage * 3) #scale from 7-10
    
    def _calculate_technique_score(self, fighter_id):
        stats = self.fight_stats[fighter_id]
        
        if stats['strikes']['total_attempted'] == 0:
            return 7.5 #default for no activity
            
        strike_types_used = len(stats['strikes']['by_type'])
        
        variety_score = min(3, strike_types_used / 3) #reward diverse techniques
        
        accuracy = (stats['strikes']['total_landed'] / 
                   max(1, stats['strikes']['total_attempted']))
        execution_score = 4 + (accuracy * 3) #reward effective execution
        
        return variety_score + execution_score
    
    def _update_fatigue_between_rounds(self):
        for fighter_id in [0, 1]:
            recovery = self.fight_stats[fighter_id]['fatigue']['recovery_rate'] / 100
            current_fatigue = self.fight_stats[fighter_id]['fatigue']['current_level']
            
            new_fatigue = max(0, current_fatigue - (current_fatigue * recovery)) #recover between rounds
            self.fight_stats[fighter_id]['fatigue']['current_level'] = new_fatigue
    
    def record_strike(self, fighter_id, strike_type, landed=True, effectiveness='clean'):
        if fighter_id not in [0, 1]:
            return
            
        self.fight_stats[fighter_id]['strikes']['total_attempted'] += 1
        
        if landed:
            self.fight_stats[fighter_id]['strikes']['total_landed'] += 1
            
            effectiveness_multiplier = self.strike_effectiveness.get(effectiveness, 1.0)
            
            if 'kick' in strike_type:
                base_points = 4 #kicks score highest
            elif 'knee' in strike_type or 'elbow' in strike_type:
                base_points = 3 #knees and elbows score well
            else:
                base_points = 2 #punches score less
                
            score_increase = base_points * effectiveness_multiplier
            self.fight_stats[fighter_id]['score'] += score_increase
            
        self.fight_stats[fighter_id]['strikes']['by_type'][strike_type]['attempted'] += 1
        if landed:
            self.fight_stats[fighter_id]['strikes']['by_type'][strike_type]['landed'] += 1
            
        opponent_id = 1 - fighter_id #opposite fighter
        self.fight_stats[opponent_id]['defense']['total_incoming'] += 1
        
        if not landed:
            if effectiveness == 'blocked':
                self.fight_stats[opponent_id]['defense']['strikes_blocked'] += 1
            else:
                self.fight_stats[opponent_id]['defense']['strikes_evaded'] += 1
                
        self._update_fatigue_from_strike(fighter_id, strike_type, landed)
    
    def _update_fatigue_from_strike(self, fighter_id, strike_type, landed):
        if 'kick' in strike_type:
            fatigue_increase = 1.2 #kicks most tiring
        elif 'knee' in strike_type:
            fatigue_increase = 0.9 #knees moderately tiring
        elif 'elbow' in strike_type:
            fatigue_increase = 0.7 #elbows less tiring
        else:
            fatigue_increase = 0.5 #punches least tiring
            
        if not landed:
            fatigue_increase *= 1.2 #missing is more tiring
            
        current_fatigue = self.fight_stats[fighter_id]['fatigue']['current_level']
        new_fatigue = min(100, current_fatigue + fatigue_increase)
        self.fight_stats[fighter_id]['fatigue']['current_level'] = new_fatigue
    
    def update_movement(self, fighter_id, current_position, center_of_ring, frame_delta=1):
        if fighter_id not in self.fight_stats:
            return
            
        if 'last_position' not in self.fight_stats[fighter_id]:
            self.fight_stats[fighter_id]['last_position'] = current_position
            self.fight_stats[fighter_id]['last_direction'] = (0, 0)
            return
            
        last_position = self.fight_stats[fighter_id]['last_position']
        
        distance = np.linalg.norm(
            np.array(current_position) - np.array(last_position)
        )
        self.fight_stats[fighter_id]['movement']['distance_traveled'] += distance
        
        if distance > 0:
            current_direction = (
                (current_position[0] - last_position[0]) / distance,
                (current_position[1] - last_position[1]) / distance
            )
            
            last_direction = self.fight_stats[fighter_id]['last_direction']
            
            if last_direction != (0, 0):
                dot_product = (current_direction[0] * last_direction[0] + 
                              current_direction[1] * last_direction[1])
                
                if dot_product < 0.7: #detect significant direction change
                    self.fight_stats[fighter_id]['movement']['direction_changes'] += 1
            
            self.fight_stats[fighter_id]['last_direction'] = current_direction
        
        distance_to_center = np.linalg.norm(
            np.array(current_position) - np.array(center_of_ring)
        )
        
        opponent_id = 1 - fighter_id
        if 'last_position' in self.fight_stats[opponent_id]:
            opponent_position = self.fight_stats[opponent_id]['last_position']
            opponent_distance = np.linalg.norm(
                np.array(opponent_position) - np.array(center_of_ring)
            )
            
            if distance_to_center < opponent_distance: #closer to center gets credit
                self.fight_stats[fighter_id]['movement']['center_control_time'] += frame_delta / self.frames_per_second
        
        self.fight_stats[fighter_id]['last_position'] = current_position
    
    def get_current_scores(self):
        total_scores = {0: 0, 1: 0}
        
        for round_number in range(1, self.current_round):
            if round_number in self.round_scores:
                for fighter_id in [0, 1]:
                    total_scores[fighter_id] += self.round_scores[round_number][fighter_id]
        
        for fighter_id in [0, 1]:
            raw_score = self.fight_stats[fighter_id]['score']
            
            round_score = min(10, 7 + (raw_score / 100)) #scale current round score
            total_scores[fighter_id] += round_score
            
        return total_scores
    
    def get_dominant_fighter(self):
        scores = self.get_current_scores()
        
        if scores[0] > scores[1]:
            return 0
        elif scores[1] > scores[0]:
            return 1
        else:
            return -1 #draw
    
    def get_fight_summary(self):
        summary = {
            'rounds': self.current_round - 1,
            'scores': self.get_current_scores(),
            'round_scores': self.round_scores,
            'stats': {
                0: self._get_fighter_summary(0),
                1: self._get_fighter_summary(1)
            },
            'dominant_fighter': self.get_dominant_fighter(),
            'prediction': self._get_fight_prediction()
        }
        
        return summary
    
    def _get_fighter_summary(self, fighter_id):
        stats = self.fight_stats[fighter_id]
        
        strike_accuracy = 0
        if stats['strikes']['total_attempted'] > 0:
            strike_accuracy = stats['strikes']['total_landed'] / stats['strikes']['total_attempted']
        
        defense_rate = 0
        if stats['defense']['total_incoming'] > 0:
            defense_rate = (stats['defense']['strikes_blocked'] + stats['defense']['strikes_evaded']) / stats['defense']['total_incoming']
        
        center_control_percentage = 0
        if self.current_frame > 0:
            center_control_percentage = stats['movement']['center_control_time'] / (self.current_frame / self.frames_per_second)
        
        summary = {
            'strike_accuracy': strike_accuracy,
            'strike_count': stats['strikes']['total_landed'],
            'defense_rate': defense_rate,
            'fatigue_level': stats['fatigue']['current_level'],
            'most_used_strikes': self._get_most_used_strikes(fighter_id),
            'center_control_percentage': center_control_percentage
        }
        
        return summary
    
    def _get_most_used_strikes(self, fighter_id, top_n=3):
        by_type = self.fight_stats[fighter_id]['strikes']['by_type']
        
        if not by_type:
            return []
            
        sorted_strikes = sorted(
            by_type.items(),
            key=lambda x: x[1]['attempted'],
            reverse=True
        )
        
        return [
            {
                'type': strike_type,
                'attempted': counts['attempted'],
                'landed': counts['landed']
            }
            for strike_type, counts in sorted_strikes[:top_n]
        ]
    
    def _get_fight_prediction(self):
        scores = self.get_current_scores()
        completed_rounds = self.current_round - 1
        remaining_rounds = self.rounds - completed_rounds
        
        score_difference = abs(scores[0] - scores[1])
        
        max_remaining_points = remaining_rounds * 3 #estimate max possible points
        
        if score_difference > max_remaining_points:
            winner = 0 if scores[0] > scores[1] else 1
            return {
                'predicted_winner': winner,
                'confidence': 0.9,
                'method': 'decision',
                'notes': 'Insurmountable lead on scorecards'
            }
        
        for fighter_id in [0, 1]:
            if self.fight_stats[fighter_id]['fatigue']['current_level'] > 85: #extreme fatigue
                opponent = 1 - fighter_id
                return {
                    'predicted_winner': opponent,
                    'confidence': 0.7,
                    'method': 'stoppage',
                    'notes': f'Fighter {fighter_id} showing extreme fatigue'
                }
        
        if score_difference < 2: #close fight
            leader = 0 if scores[0] > scores[1] else 1
            return {
                'predicted_winner': leader,
                'confidence': 0.55,
                'method': 'decision',
                'notes': 'Very close fight, could go either way'
            }
        
        leader = 0 if scores[0] > scores[1] else 1
        return {
            'predicted_winner': leader,
            'confidence': 0.7,
            'method': 'decision',
            'notes': 'Leading on scorecards'
        }
    
    def reset(self):
        self.initialize_stats()
        self.current_round = 1
        self.round_start_frame = 0
        self.current_frame = 0