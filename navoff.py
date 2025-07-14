# -*- coding: utf-8 -*-
"""
Sanjaya - Concise, Object-Aware Navigation (Final Optimized Version)
"""

import cv2
import tkinter as tk
from tkinter import messagebox, Label, Frame, Button
from PIL import Image, ImageTk
import time
import threading
import numpy as np
import torch
from ultralytics import YOLO
import pyttsx3
import traceback

class ModernTheme:
    PRIMARY_BLACK = "#000000"; PRIMARY_WHITE = "#FFFFFF"; SECONDARY_GRAY = "#F8F8F8"
    ACCENT_GRAY = "#E0E0E0"; TEXT_GRAY = "#666666"; SUCCESS_GREEN = "#00C851"
    DANGER_RED = "#FF4444"; FONT_BUTTON = ("Segoe UI", 16, "bold")
    FONT_SMALL = ("Segoe UI", 12, "normal")

class SanjayaNavApp:
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)
        self.window.geometry("1400x800")
        self.window.configure(bg=ModernTheme.PRIMARY_WHITE)

        self.DEPTH_TO_FEET_FACTOR = 18.0 
        self.DANGER_DEPTH_VALUE = 1.2  # Tune for your camera/model

        self.is_running = False; self.tts_engine = None; self.is_tts_busy = False
        self.cap = None; self.ai_thread = None; self.last_spoken_time = 0
        self.latest_raw_frame = None; self.latest_annotated_frame = None; self.latest_depth_heatmap = None

        try:
            print("Initializing AI Engine...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Engine using device: {self.device}")

            self.yolo_model = YOLO('yolov8l.pt')
            self.midas = torch.hub.load("intel-isl/MiDaS", 'DPT_Large')
            self.midas.to(self.device); self.midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
            if self.device == 'cuda':
                self.yolo_model.half(); self.midas.half()
            self.tts_engine = pyttsx3.init()
            self.tts_engine.say("Audio engine initialized")
            self.tts_engine.runAndWait()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize models.\n\nError: {e}")
            traceback.print_exc()
            self.window.destroy()
            return

        self.setup_modern_gui()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_modern_gui(self):
        main_container = Frame(self.window, bg=ModernTheme.PRIMARY_WHITE)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1); main_container.grid_columnconfigure(1, weight=0)
        display_frame = Frame(main_container, bg=ModernTheme.PRIMARY_WHITE)
        display_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        display_frame.grid_rowconfigure(0, weight=1); display_frame.grid_rowconfigure(1, weight=1); display_frame.grid_columnconfigure(0, weight=1)
        video_container = Frame(display_frame, bg=ModernTheme.PRIMARY_BLACK, relief=tk.FLAT)
        video_container.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        Label(video_container, text="LIVE CAMERA (OBJECT DETECTION)", font=ModernTheme.FONT_SMALL, bg=ModernTheme.PRIMARY_BLACK, fg=ModernTheme.PRIMARY_WHITE).pack(pady=5)
        self.video_label = Label(video_container, bg=ModernTheme.PRIMARY_BLACK)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        depth_container = Frame(display_frame, bg=ModernTheme.PRIMARY_BLACK, relief=tk.FLAT)
        depth_container.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        Label(depth_container, text="DEPTH HEATMAP (DISTANCE)", font=ModernTheme.FONT_SMALL, bg=ModernTheme.PRIMARY_BLACK, fg=ModernTheme.PRIMARY_WHITE).pack(pady=5)
        self.depth_label = Label(depth_container, bg=ModernTheme.PRIMARY_BLACK)
        self.depth_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        control_panel = Frame(main_container, bg=ModernTheme.SECONDARY_GRAY, width=400, relief=tk.FLAT)
        control_panel.grid(row=0, column=1, sticky="ns"); control_panel.grid_propagate(False)
        Label(control_panel, text="CONTROL PANEL", font=("Segoe UI", 16, "bold"), bg=ModernTheme.SECONDARY_GRAY, fg=ModernTheme.PRIMARY_BLACK, pady=20).pack(fill=tk.X, padx=20, pady=(0, 10))
        nav_frame = Frame(control_panel, bg=ModernTheme.SECONDARY_GRAY)
        nav_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        self.btn_start = Button(nav_frame, text="START NAVIGATION", font=ModernTheme.FONT_BUTTON, command=self.start_navigation, bg=ModernTheme.SUCCESS_GREEN, fg=ModernTheme.PRIMARY_WHITE, relief=tk.FLAT, cursor="hand2", pady=15)
        self.btn_start.pack(fill=tk.X, pady=(0, 10))
        self.btn_stop = Button(nav_frame, text="STOP NAVIGATION", font=ModernTheme.FONT_BUTTON, command=self.stop_navigation, state=tk.DISABLED, bg=ModernTheme.DANGER_RED, fg=ModernTheme.PRIMARY_WHITE, relief=tk.FLAT, cursor="hand2", pady=15)
        self.btn_stop.pack(fill=tk.X)
        audio_section = Frame(control_panel, bg=ModernTheme.PRIMARY_WHITE, relief=tk.RIDGE, bd=1)
        audio_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        Label(audio_section, text="🔊 AUDIO GUIDANCE", font=("Segoe UI", 14, "bold"), bg=ModernTheme.PRIMARY_WHITE, fg=ModernTheme.PRIMARY_BLACK, pady=15).pack(fill=tk.X, padx=15)
        self.audio_cue_display = Label(audio_section, text="System is ready.", font=("Segoe UI", 12, "italic"), wraplength=340, justify=tk.LEFT, bg=ModernTheme.ACCENT_GRAY, pady=15, relief=tk.FLAT)
        self.audio_cue_display.pack(fill=tk.X, pady=10, padx=15)

    def start_navigation(self):
        if self.is_running: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access webcam."); return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED); self.btn_stop.config(state=tk.NORMAL)
        self.speak("Navigation system activated.")
        self.ai_thread = threading.Thread(target=self.run_ai_assistant, daemon=True)
        self.ai_thread.start()
        self.update_gui_loop()
        
    def stop_navigation(self):
        if not self.is_running: return
        self.is_running = False
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release(); self.cap = None
        self.btn_start.config(state=tk.NORMAL); self.btn_stop.config(state=tk.DISABLED)
        self.video_label.config(image=''); self.depth_label.config(image='')
        self.update_audio_cue_display("Navigation stopped.")
        self.speak("Navigation system shutting down.")
        
    def on_close(self):
        self.stop_navigation()
        if self.tts_engine: self.tts_engine.stop()
        self.window.destroy()

    def run_ai_assistant(self):
        while self.is_running:
            try:
                if self.latest_raw_frame is not None:
                    frame_for_analysis = self.latest_raw_frame.copy()
                    annotated_frame, depth_heatmap, advice = self.process_full_pipeline(frame_for_analysis)
                    self.latest_annotated_frame = annotated_frame
                    self.latest_depth_heatmap = depth_heatmap
                    
                    if (time.time() - self.last_spoken_time > 4) and advice:
                        self.last_spoken_time = time.time()
                        self.update_audio_cue_display(advice)
                        self.speak(advice)
                time.sleep(0.1)
            except Exception as e:
                print(f"[AI Thread Error]: {e}"); traceback.print_exc(); time.sleep(1)

    def update_gui_loop(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if ret:
            self.latest_raw_frame = frame
            if self.latest_annotated_frame is not None:
                self.display_image(self.latest_annotated_frame, self.video_label)
            else:
                self.display_image(frame, self.video_label)
            if self.latest_depth_heatmap is not None:
                self.display_image(self.latest_depth_heatmap, self.depth_label)
        self.window.after(33, self.update_gui_loop)

    def process_full_pipeline(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        if self.device == 'cuda': input_batch = input_batch.half()
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map[np.isnan(depth_map) | np.isinf(depth_map)] = 0
        yolo_results = self.yolo_model(frame, verbose=False, imgsz=320)[0]

        frame_width = frame.shape[1]
        advice = self._get_zone_based_navigation(depth_map, yolo_results, frame_width)
        annotated_frame = yolo_results.plot()
        depth_heatmap = self._visualize_depth_map(depth_map)
        return annotated_frame, depth_heatmap, advice

    def _get_zone_based_navigation(self, depth_map, yolo_results, frame_width):
        zones = ['extreme left', 'left', 'center', 'right', 'extreme right']
        zone_indices = [0, frame_width//5, 2*frame_width//5, 3*frame_width//5, 4*frame_width//5, frame_width]
        zone_status = {}
        zone_objects = {z: [] for z in zones}

        for i in range(5):
            zone = depth_map[:, zone_indices[i]:zone_indices[i+1]]
            avg_depth = np.mean(zone)
            # For iPhone/Camo: clear if avg_depth >= threshold (meters)
            zone_status[zones[i]] = 'blocked' if avg_depth < self.DANGER_DEPTH_VALUE else 'clear'

        # Assign objects to zones, always mention them if detected
        for box in yolo_results.boxes:
            x_center = int(box.xywh[0][0].item())
            obj_class = int(box.cls[0])
            obj_name = self.yolo_model.names[obj_class] if obj_class < len(self.yolo_model.names) else "object"
            for i in range(5):
                if zone_indices[i] <= x_center < zone_indices[i+1]:
                    zone_objects[zones[i]].append(obj_name)
                    break

        # 1. Center object warning has highest priority
        if zone_objects['center']:
            obj = zone_objects['center'][0]
            if zone_status['center'] == 'blocked':
                return f"{obj} ahead, proceed with caution."
            else:
                return f"{obj} ahead, you can walk straight, but proceed with caution."

        # 2. If center is blocked but no object, generic caution
        if zone_status['center'] == 'blocked':
            return "Obstacle ahead, proceed with caution."

        # 3. Left/right object warnings
        if zone_objects['left']:
            obj = zone_objects['left'][0]
            if zone_status['left'] == 'blocked':
                return f"{obj} on your left, proceed with caution."
            else:
                return f"{obj} on your left, you can go straight, but proceed with caution."
        if zone_objects['right']:
            obj = zone_objects['right'][0]
            if zone_status['right'] == 'blocked':
                return f"{obj} on your right, proceed with caution."
            else:
                return f"{obj} on your right, you can go straight, but proceed with caution."

        # 4. If center is clear and no object, positive cue
        if zone_status['center'] == 'clear':
            return "You can walk straight."
        elif zone_status['left'] == 'clear':
            return "You can go left."
        elif zone_status['right'] == 'clear':
            return "You can go right."
        elif zone_status['extreme left'] == 'clear':
            return "You can go extreme left."
        elif zone_status['extreme right'] == 'clear':
            return "You can go extreme right."
        else:
            return "No clear path detected, proceed with caution."

    def _visualize_depth_map(self, depth_map):
        if depth_map is None or depth_map.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        depth_map_32f = depth_map.astype(np.float32)
        output_normalized = cv2.normalize(depth_map_32f, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        return cv2.applyColorMap(output_normalized, cv2.COLORMAP_INFERNO)

    def speak(self, text):
        def do_speak():
            if self.is_tts_busy: return
            try:
                self.is_tts_busy = True
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e: print(f"[TTS Error]: {e}")
            finally: self.is_tts_busy = False
        threading.Thread(target=do_speak, daemon=True).start()

    def display_image(self, frame, label):
        if not label.winfo_exists(): return 
        label_w, label_h = label.winfo_width(), label.winfo_height()
        if label_w <= 1 or label_h <= 1: return
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.resize((label_w, label_h), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        label.imgtk = imgtk; label.config(image=imgtk)

    def update_audio_cue_display(self, text):
        if self.audio_cue_display.winfo_exists():
            self.window.after(0, lambda: self.audio_cue_display.config(text=text))

if __name__ == "__main__":
    root = tk.Tk()
    app = SanjayaNavApp(root, "Sanjaya - Concise Positive Navigation")
    root.mainloop()
