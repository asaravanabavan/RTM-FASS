import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io

class FightVisualizer:
    def __init__(self):
        self.colors = {
            0: (0, 255, 0), #green for fighter 0
            1: (0, 0, 255), #blue for fighter 1
        }
        
        self.strike_colors = {
            'jab': (255, 255, 0), #yellow for jabs
            'cross': (255, 165, 0), #orange for crosses
            'hook': (255, 0, 255), #magenta for hooks
            'uppercut': (128, 0, 128), #purple for uppercuts
            'teep': (0, 255, 255), #cyan for teeps
            'roundhouse': (255, 0, 0), #red for roundhouse kicks
            'knee': (0, 165, 255), #orange-blue for knees
            'elbow': (0, 128, 255) #light blue for elbows
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.line_thickness = 2
        self.text_padding = 5
        
        self.active_strikes = {}
        self.strike_display_frames = 45 #how long strikes show on screen
    
    def visualize_frame(self, frame, people, strikes, stats, frame_index, frames_per_second, round_number):
        visualization_frame = frame.copy()
        
        self._add_round_timer(visualization_frame, frame_index, frames_per_second, round_number)
        
        self._add_score_display(visualization_frame, stats)
        
        self._update_active_strikes(strikes, frame_index)
        
        self._draw_strike_indicators(visualization_frame, frame_index)
        
        self._add_stats_panel(visualization_frame, stats)
        
        return visualization_frame
    
    def _add_round_timer(self, frame, frame_index, frames_per_second, round_number):
        height, width = frame.shape[:2]
        
        total_seconds = int(frame_index / frames_per_second)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        round_text = f"Round {round_number}"
        timer_text = f"{minutes:02d}:{seconds:02d}"
        
        (round_width, round_height), _ = cv2.getTextSize(round_text, self.font, self.font_scale, self.line_thickness)
        (timer_width, timer_height), _ = cv2.getTextSize(timer_text, self.font, self.font_scale, self.line_thickness)
        
        round_x = (width - round_width) // 2
        timer_x = (width - timer_width) // 2
        
        cv2.rectangle(frame, 
                     (round_x - self.text_padding, 10 - self.text_padding),
                     (round_x + round_width + self.text_padding, 10 + round_height + self.text_padding + timer_height + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, round_text, (round_x, 10 + round_height), 
                   self.font, self.font_scale, (255, 255, 255), self.line_thickness)
        cv2.putText(frame, timer_text, (timer_x, 10 + round_height + 5 + timer_height), 
                   self.font, self.font_scale, (255, 255, 255), self.line_thickness)
    
    def _add_score_display(self, frame, stats):
        height, width = frame.shape[:2]
        
        scores = stats['scores']
        
        fighter0_text = f"Fighter 0: {scores[0]:0.1f}"
        fighter1_text = f"Fighter 1: {scores[1]:0.1f}"
        
        (fighter0_width, fighter0_height), _ = cv2.getTextSize(fighter0_text, self.font, self.font_scale, self.line_thickness)
        (fighter1_width, fighter1_height), _ = cv2.getTextSize(fighter1_text, self.font, self.font_scale, self.line_thickness)
        
        cv2.rectangle(frame, 
                     (10 - self.text_padding, 10 - self.text_padding),
                     (10 + max(fighter0_width, fighter1_width) + self.text_padding, 10 + fighter0_height + fighter1_height + 15),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, fighter0_text, (10, 10 + fighter0_height), 
                   self.font, self.font_scale, self.colors[0], self.line_thickness)
        cv2.putText(frame, fighter1_text, (10, 10 + fighter0_height + 5 + fighter1_height), 
                   self.font, self.font_scale, self.colors[1], self.line_thickness)
    
    def _update_active_strikes(self, strikes, frame_index):
        for fighter_id, fighter_strikes in strikes.items():
            for strike in fighter_strikes:
                self.active_strikes[(fighter_id, frame_index)] = strike
        
        to_remove = []
        for (fighter_id, strike_frame), strike in self.active_strikes.items():
            if frame_index - strike_frame > self.strike_display_frames:
                to_remove.append((fighter_id, strike_frame))
        
        for key in to_remove:
            self.active_strikes.pop(key, None) #remove old strikes
    
    def _draw_strike_indicators(self, frame, frame_index):
        height, width = frame.shape[:2]
        
        for (fighter_id, strike_frame), strike in self.active_strikes.items():
            if frame_index - strike_frame > self.strike_display_frames:
                continue
            
            fade_factor = 1.0 - (frame_index - strike_frame) / self.strike_display_frames #fade out over time
            
            strike_type = strike['type']
            outcome = strike.get('outcome', 'unknown')
            
            text = f"{strike_type.capitalize()}"
            if outcome == 'successful':
                text += " ✓" #checkmark for hits
            elif outcome == 'unsuccessful':
                text += " ✗" #x for misses
            
            color = self.strike_colors.get(strike_type.lower(), (255, 255, 255))
            
            fighter_color = self.colors[fighter_id]
            blended_color = (
                int((color[0] + fighter_color[0]) / 2 * fade_factor),
                int((color[1] + fighter_color[1]) / 2 * fade_factor),
                int((color[2] + fighter_color[2]) / 2 * fade_factor)
            )
            
            x_position = 20 if fighter_id == 0 else width - 220
            y_position = 100 + (frame_index - strike_frame) * 2
            
            (text_width, text_height), _ = cv2.getTextSize(text, self.font, self.font_scale, self.line_thickness)
            cv2.rectangle(frame, 
                         (x_position - self.text_padding, y_position - text_height - self.text_padding),
                         (x_position + text_width + self.text_padding, y_position + self.text_padding),
                         (0, 0, 0, int(255 * fade_factor)), -1)
            
            cv2.putText(frame, text, (x_position, y_position), 
                       self.font, self.font_scale, blended_color, self.line_thickness)
    
    def _add_stats_panel(self, frame, stats):
        height, width = frame.shape[:2]
        
        panel_height = 120
        panel_y = height - panel_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame) #semi-transparent overlay
        
        for fighter_id in [0, 1]:
            fighter_stats = stats['stats'][fighter_id]
            
            x_position = 20 if fighter_id == 0 else width - 220
            
            accuracy = fighter_stats['strike_accuracy'] * 100
            landed = fighter_stats['strike_count']
            defense = fighter_stats['defense_rate'] * 100
            
            stats_text = [
                f"Fighter {fighter_id}",
                f"Strikes: {landed} ({accuracy:0.1f}%)",
                f"Defense: {defense:0.1f}%"
            ]
            
            if fighter_stats['most_used_strikes']:
                stats_text.append("Most used:")
                for strike in fighter_stats['most_used_strikes'][:2]:
                    accuracy = 0
                    if strike['attempted'] > 0:
                        accuracy = (strike['landed'] / strike['attempted']) * 100
                    stats_text.append(f"  {strike['type']}: {strike['landed']}/{strike['attempted']} ({accuracy:0.1f}%)")
            
            for i, text in enumerate(stats_text):
                y = panel_y + 20 + (i * 20)
                cv2.putText(frame, text, (x_position, y), 
                           self.font, self.font_scale, self.colors[fighter_id], self.line_thickness)
    
    def create_fight_summary(self, summary):
        image_width, image_height = 1280, 720
        background_color = (32, 32, 32) #dark gray background
        
        image = Image.new('RGB', (image_width, image_height), background_color)
        draw = ImageDraw.Draw(image)
        
        title = "FIGHT ANALYSIS SUMMARY"
        title_font_size = 40
        try:
            title_font = ImageFont.truetype("arial.ttf", title_font_size)
        except IOError:
            title_font = ImageFont.load_default() #fallback font
        
        title_width = draw.textlength(title, font=title_font)
        draw.text(((image_width - title_width) // 2, 20), title, fill=(255, 255, 255), font=title_font)
        
        self._draw_round_scores(draw, summary, image_width, title_font)
        
        self._draw_fighter_stats(draw, summary, image_width, image_height)
        
        self._draw_strike_breakdown(draw, summary, image_width, image_height)
        
        self._draw_prediction(draw, summary, image_width, image_height)
        
        return np.array(image)[:, :, ::-1] #convert PIL to OpenCV format
    
    def _draw_round_scores(self, draw, summary, image_width, title_font):
        y_position = 80
        
        header = "ROUND SCORES"
        header_width = draw.textlength(header, font=title_font)
        draw.text(((image_width - header_width) // 2, y_position), header, fill=(255, 255, 255), font=title_font)
        
        table_top = y_position + 60
        rounds = summary['rounds']
        
        try:
            table_font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            table_font = ImageFont.load_default()
        
        draw.text(((image_width // 4), table_top), "ROUND", fill=(255, 255, 255), font=table_font)
        draw.text(((image_width // 2) - 50, table_top), "FIGHTER 0", fill=(0, 255, 0), font=table_font)
        draw.text(((image_width // 2) + 100, table_top), "FIGHTER 1", fill=(0, 0, 255), font=table_font)
        
        draw.line([(image_width // 4) - 20, table_top + 30, (image_width // 4) + 320, table_top + 30], fill=(200, 200, 200), width=2)
        
        for round_number in range(1, rounds + 1):
            round_y = table_top + 40 + ((round_number - 1) * 30)
            
            draw.text(((image_width // 4), round_y), f"Round {round_number}", fill=(255, 255, 255), font=table_font)
            
            if round_number in summary['round_scores']:
                round_scores = summary['round_scores'][round_number]
                draw.text(((image_width // 2) - 50, round_y), f"{round_scores[0]:0.1f}", fill=(0, 255, 0), font=table_font)
                draw.text(((image_width // 2) + 100, round_y), f"{round_scores[1]:0.1f}", fill=(0, 0, 255), font=table_font)
        
        total_y = table_top + 40 + (rounds * 30) + 10
        draw.line([(image_width // 4) - 20, total_y, (image_width // 4) + 320, total_y], fill=(200, 200, 200), width=2)
        
        draw.text(((image_width // 4), total_y + 10), "TOTAL", fill=(255, 255, 255), font=table_font)
        draw.text(((image_width // 2) - 50, total_y + 10), f"{summary['scores'][0]:0.1f}", fill=(0, 255, 0), font=table_font)
        draw.text(((image_width // 2) + 100, total_y + 10), f"{summary['scores'][1]:0.1f}", fill=(0, 0, 255), font=table_font)
    
    def _draw_fighter_stats(self, draw, summary, image_width, image_height):
        y_position = 350
        
        try:
            stats_font = ImageFont.truetype("arial.ttf", 20)
            header_font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            stats_font = ImageFont.load_default()
            header_font = stats_font
        
        header = "FIGHTER STATISTICS"
        header_width = draw.textlength(header, font=header_font)
        draw.text(((image_width - header_width) // 2, y_position), header, fill=(255, 255, 255), font=header_font)
        
        fighter0_stats = summary['stats'][0]
        fighter1_stats = summary['stats'][1]
        
        stats_data = [
            ("Strike Accuracy", f"{fighter0_stats['strike_accuracy']*100:0.1f}%", f"{fighter1_stats['strike_accuracy']*100:0.1f}%"),
            ("Strikes Landed", f"{fighter0_stats['strike_count']}", f"{fighter1_stats['strike_count']}"),
            ("Defense Rate", f"{fighter0_stats['defense_rate']*100:0.1f}%", f"{fighter1_stats['defense_rate']*100:0.1f}%"),
            ("Center Control", f"{fighter0_stats['center_control_percentage']*100:0.1f}%", f"{fighter1_stats['center_control_percentage']*100:0.1f}%")
        ]
        
        stats_y = y_position + 50
        column_width = 150
        
        draw.text((image_width // 2 - column_width - 50, stats_y), "FIGHTER 0", fill=(0, 255, 0), font=stats_font)
        draw.text((image_width // 2 - 50, stats_y), "METRIC", fill=(255, 255, 255), font=stats_font)
        draw.text((image_width // 2 + column_width - 50, stats_y), "FIGHTER 1", fill=(0, 0, 255), font=stats_font)
        
        for i, (name, fighter0_value, fighter1_value) in enumerate(stats_data):
            row_y = stats_y + 30 + (i * 30)
            
            name_width = draw.textlength(name, font=stats_font)
            name_x = (image_width // 2) - (name_width // 2)
            
            draw.text((image_width // 2 - column_width - 50, row_y), fighter0_value, fill=(0, 255, 0), font=stats_font)
            draw.text((name_x, row_y), name, fill=(255, 255, 255), font=stats_font)
            draw.text((image_width // 2 + column_width - 50, row_y), fighter1_value, fill=(0, 0, 255), font=stats_font)
    
    def _draw_strike_breakdown(self, draw, summary, image_width, image_height):
        y_position = 530
        
        try:
            stats_font = ImageFont.truetype("arial.ttf", 20)
            header_font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            stats_font = ImageFont.load_default()
            header_font = stats_font
        
        header = "STRIKE BREAKDOWN"
        header_width = draw.textlength(header, font=header_font)
        draw.text(((image_width - header_width) // 2, y_position), header, fill=(255, 255, 255), font=header_font)
        
        fighter0_strikes = summary['stats'][0]['most_used_strikes']
        fighter1_strikes = summary['stats'][1]['most_used_strikes']
        
        column1_x = image_width // 4
        column2_x = (image_width * 3) // 4
        
        draw.text((column1_x - 100, y_position + 50), "FIGHTER 0 STRIKES", fill=(0, 255, 0), font=stats_font)
        draw.text((column2_x - 100, y_position + 50), "FIGHTER 1 STRIKES", fill=(0, 0, 255), font=stats_font)
        
        for i, strike in enumerate(fighter0_strikes):
            if i >= 3: #show top 3 strikes
                break
                
            row_y = y_position + 80 + (i * 30)
            accuracy = 0
            if strike['attempted'] > 0:
                accuracy = (strike['landed'] / strike['attempted']) * 100
                
            draw.text((column1_x - 100, row_y), 
                     f"{strike['type'].capitalize()}: {strike['landed']}/{strike['attempted']} ({accuracy:0.1f}%)", 
                     fill=(0, 255, 0), font=stats_font)
        
        for i, strike in enumerate(fighter1_strikes):
            if i >= 3: #show top 3 strikes
                break
                
            row_y = y_position + 80 + (i * 30)
            accuracy = 0
            if strike['attempted'] > 0:
                accuracy = (strike['landed'] / strike['attempted']) * 100
                
            draw.text((column2_x - 100, row_y), 
                     f"{strike['type'].capitalize()}: {strike['landed']}/{strike['attempted']} ({accuracy:0.1f}%)", 
                     fill=(0, 0, 255), font=stats_font)
    
    def _draw_prediction(self, draw, summary, image_width, image_height):
        y_position = image_height - 100
        
        try:
            prediction_font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            prediction_font = ImageFont.load_default()
        
        prediction = summary['prediction']
        winner_id = prediction['predicted_winner']
        confidence = prediction['confidence']
        method = prediction['method']
        notes = prediction['notes']
        
        prediction_text = f"PREDICTION: Fighter {winner_id} wins by {method.upper()} ({confidence*100:0.0f}% confidence)"
        prediction_width = draw.textlength(prediction_text, font=prediction_font)
        
        draw.text(((image_width - prediction_width) // 2, y_position), 
                 prediction_text, 
                 fill=self.colors[winner_id], font=prediction_font)
        
        notes_width = draw.textlength(notes, font=prediction_font)
        draw.text(((image_width - notes_width) // 2, y_position + 30), 
                 notes, 
                 fill=(200, 200, 200), font=prediction_font) #gray for notes text