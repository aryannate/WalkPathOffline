# -*- coding: utf-8 -*-
import cv2
import tkinter as tk
from tkinter import messagebox, Label, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch # NEW: Required for the MiDaS depth model
import pyttsx3
import time
import threading
import os

# --- MODERN UI THEME CONFIGURATION ---
class ModernTheme:
    PRIMARY_BLACK = "#000000"
    PRIMARY_WHITE = "#FFFFFF"
    SECONDARY_GRAY = "#F8F8F8"
    ACCENT_GRAY = "#E0E0E0"
    TEXT_GRAY = "#666666"
    SUCCESS_GREEN = "#00C851"
    DANGER_RED = "#FF4444"
    FONT_LARGE = ("Segoe UI", 24, "bold")
    FONT_MEDIUM = ("Segoe UI", 14, "normal")
    FONT_SMALL = ("Segoe UI", 12, "normal")
    FONT_BUTTON = ("Segoe UI", 16, "bold")

# --- MAIN APPLICATION CLASS ---
class SanjayaNavApp:
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)
        self.window.geometry("1200x800")
        self.window.configure(bg=ModernTheme.PRIMARY_WHITE)
        self.window.resizable(True, True)

        self.is_running = False
        self.cap = None
        self.latest_frame = None
        self.ai_thread = None
        self.last_spoken_time = 0

        try:
            # --- 1. Load YOLO Model ---
            self.yolo_model = YOLO("yolov8s.pt")
            
            # --- 2. Load MiDaS Depth Estimation Model ---
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.to(self.device)
            self.midas.eval()
            
            # --- Load MiDaS transforms to prepare the images ---
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform

            # --- 3. Initialize Text-to-Speech Engine ---
            self.tts_engine = pyttsx3.init()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize models or TTS engine: {e}")
            self.window.destroy()
            return

        self.setup_modern_gui()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_modern_gui(self):
        # ... (All GUI setup code remains the same)
        main_container = Frame(self.window, bg=ModernTheme.PRIMARY_WHITE)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        header_frame = Frame(main_container, bg=ModernTheme.PRIMARY_WHITE, height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        title_label = Label(header_frame, text="SANJAYA", font=ModernTheme.FONT_LARGE, bg=ModernTheme.PRIMARY_WHITE, fg=ModernTheme.PRIMARY_BLACK)
        title_label.pack(side=tk.LEFT, anchor=tk.W, pady=10)
        subtitle_label = Label(header_frame, text="AI Indoor Navigation Assistant", font=ModernTheme.FONT_MEDIUM, bg=ModernTheme.PRIMARY_WHITE, fg=ModernTheme.TEXT_GRAY)
        subtitle_label.pack(side=tk.LEFT, anchor=tk.W, padx=(20, 0), pady=15)
        content_frame = Frame(main_container, bg=ModernTheme.PRIMARY_WHITE)
        content_frame.pack(fill=tk.BOTH, expand=True)
        video_container = Frame(content_frame, bg=ModernTheme.PRIMARY_BLACK, relief=tk.FLAT, bd=2)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        video_container.pack_propagate(False)
        self.video_label = Label(video_container, bg=ModernTheme.PRIMARY_BLACK, text="CAMERA FEED\n\nPress 'Start Navigation' to begin", fg=ModernTheme.PRIMARY_WHITE, font=ModernTheme.FONT_MEDIUM, justify=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        control_panel = Frame(content_frame, bg=ModernTheme.SECONDARY_GRAY, width=350, relief=tk.FLAT, bd=1)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        control_panel.pack_propagate(False)
        control_header = Label(control_panel, text="CONTROL PANEL", font=("Segoe UI", 16, "bold"), bg=ModernTheme.SECONDARY_GRAY, fg=ModernTheme.PRIMARY_BLACK, pady=20)
        control_header.pack(fill=tk.X, padx=20)
        nav_frame = Frame(control_panel, bg=ModernTheme.SECONDARY_GRAY)
        nav_frame.pack(fill=tk.X, padx=20, pady=(0, 30))
        self.btn_start = tk.Button(nav_frame, text="START NAVIGATION", font=ModernTheme.FONT_BUTTON, command=self.start_navigation, bg=ModernTheme.SUCCESS_GREEN, fg=ModernTheme.PRIMARY_WHITE, relief=tk.FLAT, cursor="hand2", pady=15)
        self.btn_start.pack(fill=tk.X, pady=(0, 10))
        self.btn_stop = tk.Button(nav_frame, text="STOP NAVIGATION", font=ModernTheme.FONT_BUTTON, command=self.stop_navigation, state=tk.DISABLED, bg=ModernTheme.DANGER_RED, fg=ModernTheme.PRIMARY_WHITE, relief=tk.FLAT, cursor="hand2", pady=15)
        self.btn_stop.pack(fill=tk.X)
        audio_section = Frame(control_panel, bg=ModernTheme.PRIMARY_WHITE, relief=tk.FLAT, bd=1)
        audio_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        audio_header = Label(audio_section, text="🔊 AUDIO GUIDANCE", font=("Segoe UI", 14, "bold"), bg=ModernTheme.PRIMARY_WHITE, fg=ModernTheme.PRIMARY_BLACK, pady=15)
        audio_header.pack(fill=tk.X, padx=15)
        status_container = Frame(audio_section, bg=ModernTheme.PRIMARY_WHITE)
        status_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        status_label = Label(status_container, text="STATUS:", font=("Segoe UI", 11, "bold"), bg=ModernTheme.PRIMARY_WHITE, fg=ModernTheme.TEXT_GRAY)
        status_label.pack(anchor=tk.W, pady=(0, 5))
        self.ai_status_label = Label(status_container, text="System ready. Waiting for activation...", bg=ModernTheme.PRIMARY_WHITE, fg=ModernTheme.PRIMARY_BLACK, wraplength=280, justify=tk.LEFT, font=ModernTheme.FONT_SMALL, pady=10, relief=tk.FLAT, bd=1)
        self.ai_status_label.pack(fill=tk.BOTH, expand=True, padx=5)
        cue_label = Label(status_container, text="LAST GUIDANCE:", font=("Segoe UI", 11, "bold"), bg=ModernTheme.PRIMARY_WHITE, fg=ModernTheme.TEXT_GRAY)
        cue_label.pack(anchor=tk.W, pady=(15, 5))
        self.audio_cue_display = Label(status_container, text="No guidance yet", bg=ModernTheme.ACCENT_GRAY, fg=ModernTheme.PRIMARY_BLACK, wraplength=280, justify=tk.LEFT, font=("Segoe UI", 12, "italic"), pady=15, relief=tk.FLAT)
        self.audio_cue_display.pack(fill=tk.X, padx=5)
        footer_frame = Frame(main_container, bg=ModernTheme.PRIMARY_WHITE, height=40)
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        footer_frame.pack_propagate(False)
        footer_label = Label(footer_frame, text="Powered by YOLOv8 + MiDaS Depth • Designed for Accessibility", font=("Segoe UI", 10, "normal"), bg=ModernTheme.PRIMARY_WHITE, fg=ModernTheme.TEXT_GRAY)
        footer_label.pack(anchor=tk.CENTER, pady=10)


    # --- THE UPGRADED PATHFINDING ENGINE ---
    def run_rule_based_assistant(self):
        """Runs in a thread, using both YOLO and MiDaS depth to find a safe path."""
        while self.is_running:
            if (time.time() - self.last_spoken_time > 4) and self.latest_frame is not None:
                self.last_spoken_time = time.time()
                try:
                    self.update_status_label("🔍 Analyzing scene...")
                    frame_for_analysis = self.latest_frame.copy()
                    
                    # --- Step 1: YOLO Object Detection ---
                    yolo_results = self.yolo_model(frame_for_analysis, verbose=False)[0]
                    frame_height, frame_width, _ = frame_for_analysis.shape

                    # --- Step 2: MiDaS Depth Estimation ---
                    img_rgb = cv2.cvtColor(frame_for_analysis, cv2.COLOR_BGR2RGB)
                    input_batch = self.transform(img_rgb).to(self.device)
                    with torch.no_grad():
                        prediction = self.midas(input_batch)
                        prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img_rgb.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()
                    depth_map = prediction.cpu().numpy()

                    # --- Step 3: Map the Lanes using BOTH models ---
                    lanes = { "at 10 o'clock": "Open", "at 11 o'clock": "Open", "directly ahead": "Open", "at 1 o'clock": "Open", "at 2 o'clock": "Open" }
                    lane_order = list(lanes.keys())
                    zone_width = frame_width / 5
                    
                    # Block lanes based on YOLO objects
                    for box in yolo_results.boxes:
                        if box.conf[0].item() > 0.65:
                            lane_name = self.get_object_location(frame_width, box)
                            if lane_name in lanes:
                                lanes[lane_name] = "Blocked by " + self.yolo_model.names[int(box.cls[0].item())]

                    # Block lanes based on Depth (for walls, etc.)
                    depth_threshold = depth_map.max() * 0.8 # Anything in the closest 20% of depth is a threat
                    for i, lane_name in enumerate(lane_order):
                        if lanes[lane_name] == "Open": # Only check lanes not already blocked by an object
                            lane_start_x = int(i * zone_width)
                            lane_end_x = int((i + 1) * zone_width)
                            lane_depth_area = depth_map[:, lane_start_x:lane_end_x]
                            if lane_depth_area.size > 0 and lane_depth_area.mean() > depth_threshold:
                                lanes[lane_name] = "Blocked"

                    # --- Step 4: Find the Best Path ---
                    start_index, path_length = self.find_best_path(lanes)

                    # --- Step 5: Generate Command Based on Best Path ---
                    advice = "Obstacles detected. Please stop and evaluate."

                    if path_length >= 3 and lanes["directly ahead"] == "Open":
                        advice = "Clear path ahead. Proceed."
                    elif path_length >= 2:
                        path_center_index = start_index + path_length / 2
                        if path_center_index < 2:  # Path is to the left
                            threat_reason = self.get_primary_threat_reason(lanes, 'right')
                            advice = f"{threat_reason} on your right. Path is clearer to the left."
                        elif path_center_index > 2:  # Path is to the right
                            threat_reason = self.get_primary_threat_reason(lanes, 'left')
                            advice = f"{threat_reason} on your left. Path is clearer to the right."
                        else:
                            advice = "Clear path ahead. Proceed."
                    
                    self.update_status_label("🎯 Guidance ready")
                    self.update_audio_cue_display(advice)
                    self.speak(advice)

                except Exception as e:
                    print(f"[AI Error]: {e}")
                    self.update_status_label("❌ Rule Engine Error")
            time.sleep(0.5)

    def get_primary_threat_reason(self, lanes, side):
        """Finds the reason a side is blocked."""
        if side == 'left':
            locations = ["at 11 o'clock", "at 10 o'clock"]
        else: # side == 'right'
            locations = ["at 1 o'clock", "at 2 o'clock"]

        for loc in locations:
            if lanes[loc].startswith("Blocked by "):
                return lanes[loc].replace("Blocked by ", "").capitalize()
        
        if lanes["directly ahead"].startswith("Blocked by"):
            return lanes["directly ahead"].replace("Blocked by ", "").capitalize()
            
        return "Obstacle"

    # All other methods (find_best_path, start/stop_navigation, etc.) remain the same...
    def find_best_path(self, lanes):
        lane_order = ["at 10 o'clock", "at 11 o'clock", "directly ahead", "at 1 o'clock", "at 2 o'clock"]
        max_len, best_start_index = 0, -1
        current_len, current_start_index = 0, -1
        for i, lane in enumerate(lane_order):
            if lanes[lane] == "Open":
                if current_len == 0: current_start_index = i
                current_len += 1
            else:
                if current_len > max_len: max_len, best_start_index = current_len, current_start_index
                current_len = 0
        if current_len > max_len: max_len, best_start_index = current_len, current_start_index
        return best_start_index, max_len

    def get_object_location(self, frame_width, box):
        box_center_x = box.xywh[0][0].item()
        zone_width = frame_width / 5
        if box_center_x < zone_width: return "at 10 o'clock"
        elif box_center_x < zone_width * 2: return "at 11 o'clock"
        elif box_center_x < zone_width * 3: return "directly ahead"
        elif box_center_x < zone_width * 4: return "at 1 o'clock"
        else: return "at 2 o'clock"

    def start_navigation(self):
        if self.is_running: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the webcam.")
            return
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.update_status_label("🟢 Navigation system activated")
        self.speak("Sanjaya navigation system activated.")
        self.ai_thread = threading.Thread(target=self.run_rule_based_assistant, daemon=True)
        self.ai_thread.start()
        self.update_frame()
        
    def stop_navigation(self):
        if not self.is_running: return
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='', text="CAMERA FEED\n\nPress 'Start Navigation' to begin", bg=ModernTheme.PRIMARY_BLACK)
        self.update_status_label("🔴 System offline")
        self.update_audio_cue_display("No guidance yet")
        self.speak("Sanjaya navigation system shutting down.")
        
    def on_close(self):
        self.stop_navigation()
        self.window.after(200, self.window.destroy)

    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame.copy()
                results = self.yolo_model(frame, verbose=False)
                annotated_frame = results[0].plot()
                label_w = self.video_label.winfo_width()
                label_h = self.video_label.winfo_height()
                if label_w > 1 and label_h > 1:
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    img = img.resize((label_w, label_h), Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.config(image=imgtk, text="")
            self.window.after(20, self.update_frame)
        
    def speak(self, text):
        def do_speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"[TTS Error]: {e}")
        self.window.after(0, do_speak)
        
    def update_status_label(self, text):
        self.window.after(0, lambda: self.ai_status_label.config(text=text))
        
    def update_audio_cue_display(self, text):
        self.window.after(0, lambda: self.audio_cue_display.config(text=text))

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SanjayaNavApp(root, "Sanjaya - AI Navigation with Depth")
    root.mainloop()
