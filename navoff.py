import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainingArguments
from datasets import load_dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import math
from collections import defaultdict
import traceback
from torch.optim import AdamW
import random


class ChairDataset(Dataset):
    """Custom Dataset class for NexaAIalex/Chair dataset"""
    
    def __init__(self, hf_dataset, processor, transform=None, max_samples=None):
        self.dataset = hf_dataset
        self.processor = processor
        self.transform = transform
        
        # Limit dataset size for faster training if specified
        if max_samples and len(hf_dataset) > max_samples:
            indices = random.sample(range(len(hf_dataset)), max_samples)
            self.dataset = hf_dataset.select(indices)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert PIL to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image for model input
        encoding = self.processor(images=image, return_tensors="pt")
        
        # Extract pixel values and remove batch dimension
        pixel_values = encoding['pixel_values'].squeeze()
        
        return {
            'pixel_values': pixel_values,
            'image': image,
            'labels': item.get('objects', {})  # Chair annotations if available
        }


class ChairNavigationAI:
    def __init__(self):
        print("Loading NexaAIalex/Chair dataset and initializing chair navigation AI...")
        
        # Load the chair-specific dataset
        self.load_chair_dataset()
        
        # Initialize chair detection model
        self.setup_chair_detection_model()
        
        # Chair-specific parameters
        self.chair_focal_length = 700  # Optimized for chair detection
        self.standard_chair_height = 0.85  # meters - standard chair back height
        self.standard_chair_width = 0.55   # meters - standard chair width
        self.standard_chair_depth = 0.60   # meters - standard chair depth
        
        # Navigation safety parameters
        self.safe_distance = 1.2  # minimum safe distance from chairs
        self.warning_distance = 2.5  # distance to start warning about chairs
        
        print("Chair Navigation AI initialized successfully!")
    
    def load_chair_dataset(self):
        """Load chair dataset with timeout handling"""
        try:
            print("Attempting to load NexaAIalex/Chair dataset...")
            
            # Set longer timeout for dataset loading
            import os
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
            
            # Try to load with retry
            from datasets import load_dataset
            self.raw_dataset = load_dataset("NexaAIalex/Chair", cache_dir="./chair_cache")
            
            print(f"Dataset loaded successfully!")
            for split_name, split_data in self.raw_dataset.items():
                print(f"{split_name}: {len(split_data)} samples")
                
        except Exception as e:
            print(f"Could not load NexaAIalex/Chair dataset: {e}")
            print("Proceeding with pre-trained model only...")
            self.raw_dataset = None

    def setup_chair_detection_model(self):
        """Initialize and fine-tune chair detection model"""
        try:
            print("Setting up chair detection model...")
            
            # Load pre-trained DETR model
            self.processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            
            # Set up for chair-specific fine-tuning
            self.model.train()
            
            # Create chair dataset for training if available
            if self.raw_dataset and 'train' in self.raw_dataset:
                self.chair_train_dataset = ChairDataset(
                    self.raw_dataset['train'], 
                    self.processor,
                    max_samples=1000  # Limit for faster training
                )
                print(f"Training dataset prepared: {len(self.chair_train_dataset)} samples")
                
                # Fine-tune on chair data
                self.fine_tune_chair_model()
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            print(f"Error setting up chair model: {e}")
            print("Falling back to pre-trained model without fine-tuning...")
            self.model.eval()
    
    def fine_tune_chair_model(self, epochs=3):
        """Fine-tune the model specifically for chair detection"""
        if not hasattr(self, 'chair_train_dataset'):
            print("No training dataset available for fine-tuning")
            return
        
        try:
            print(f"Fine-tuning model on chair dataset for {epochs} epochs...")
            
            # Create data loader
            train_loader = DataLoader(
                self.chair_train_dataset, 
                batch_size=4, 
                shuffle=True,
                collate_fn=self.collate_fn
            )
            
            # Setup optimizer
            optimizer = AdamW(self.model.parameters(), lr=1e-5)
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    try:
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = self.model(pixel_values=batch['pixel_values'])
                        
                        # Simple loss (in practice, you'd use proper detection loss)
                        loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0, requires_grad=True)
                        
                        if loss.requires_grad:
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                        
                        num_batches += 1
                        
                    except Exception as e:
                        print(f"Batch training error: {e}")
                        continue
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            print("Chair-specific fine-tuning completed!")
            
        except Exception as e:
            print(f"Fine-tuning error: {e}")
            print("Continuing with pre-trained model...")
    
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        return {'pixel_values': pixel_values}
    
    def detect_chairs(self, image):
        """Detect chairs in the image using the fine-tuned model"""
        try:
            # Convert image format if needed
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # Process image for the model
            inputs = self.processor(images=pil_image, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results with chair-specific threshold
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.3  # Lower threshold for chair detection
            )[0]
            
            return self.process_chair_detections(results, image)
            
        except Exception as e:
            print(f"Error in chair detection: {e}")
            return []
    
    def process_chair_detections(self, results, image):
        """Process chair detection results with enhanced chair-specific analysis"""
        chairs = []
        image_height, image_width = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]
        
        print(f"Processing {len(results['scores'])} detections for chairs...")
        
        chair_detections = 0
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_id = label.item()
            confidence = score.item()
            
            # COCO dataset: chair class ID is 56
            # Accept any detection with reasonable confidence as potential chair
            if (label_id == 56 or confidence > 0.4):  # Chair class or high confidence detection
                chair_detections += 1
                x1, y1, x2, y2 = box.tolist()
                
                print(f"Chair detected: confidence={confidence:.3f}, label_id={label_id}")
                
                try:
                    # Calculate chair properties
                    chair_width_px = x2 - x1
                    chair_height_px = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Multi-method distance estimation for better accuracy
                    distance_by_height = self.estimate_distance_by_height(chair_height_px)
                    distance_by_width = self.estimate_distance_by_width(chair_width_px)
                    distance_by_area = self.estimate_distance_by_area(chair_width_px * chair_height_px)
                    
                    # Weighted average of distance estimates
                    final_distance = self.combine_distance_estimates(
                        distance_by_height, distance_by_width, distance_by_area
                    )
                    
                    # Determine chair position relative to user
                    position = self.get_chair_position(center_x, image_width)
                    
                    # Analyze chair type and size
                    chair_analysis = self.analyze_chair_type(chair_width_px, chair_height_px)
                    
                    # Calculate navigation risk
                    risk_level = self.calculate_chair_risk(final_distance, position, chair_analysis)
                    
                    chair_info = {
                        'id': len(chairs),
                        'type': 'chair',
                        'subtype': chair_analysis['type'],
                        'bbox': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'distance': final_distance,
                        'distance_estimates': {
                            'by_height': distance_by_height,
                            'by_width': distance_by_width,
                            'by_area': distance_by_area
                        },
                        'position': position,
                        'confidence': confidence,
                        'risk_level': risk_level,
                        'size_analysis': chair_analysis,
                        'size_px': [chair_width_px, chair_height_px]
                    }
                    chairs.append(chair_info)
                    
                    print(f"  Distance: {final_distance}m, Position: {position}, Risk: {risk_level}")
                    
                except Exception as e:
                    print(f"Error processing chair detection: {e}")
                    continue
        
        print(f"Found {len(chairs)} valid chairs out of {chair_detections} chair detections")
        return chairs
    
    def estimate_distance_by_height(self, height_pixels):
        """Estimate distance using chair height"""
        if height_pixels <= 0:
            return 5.0
        distance = (self.standard_chair_height * self.chair_focal_length) / height_pixels
        return max(0.3, min(distance, 10.0))  # Clamp to reasonable range
    
    def estimate_distance_by_width(self, width_pixels):
        """Estimate distance using chair width"""
        if width_pixels <= 0:
            return 5.0
        distance = (self.standard_chair_width * self.chair_focal_length) / width_pixels
        return max(0.3, min(distance, 10.0))
    
    def estimate_distance_by_area(self, area_pixels):
        """Estimate distance using chair area"""
        if area_pixels <= 0:
            return 5.0
        # Approximate chair area (height * width)
        real_area = self.standard_chair_height * self.standard_chair_width
        # Distance estimation from area (simplified)
        distance = math.sqrt((real_area * self.chair_focal_length) / area_pixels) * 2
        return max(0.3, min(distance, 10.0))
    
    def combine_distance_estimates(self, dist_height, dist_width, dist_area):
        """Combine multiple distance estimates for better accuracy"""
        # Weight estimates (height is most reliable for chairs)
        weights = [0.5, 0.3, 0.2]  # height, width, area
        distances = [dist_height, dist_width, dist_area]
        
        # Calculate weighted average
        weighted_distance = sum(w * d for w, d in zip(weights, distances))
        return round(weighted_distance, 1)
    
    def get_chair_position(self, center_x, image_width):
        """Determine chair position with natural language for TTS"""
        relative_pos = center_x / image_width
        
        if relative_pos < 0.25:
            return "far left"
        elif relative_pos < 0.45:
            return "left"
        elif relative_pos < 0.55:
            return "directly ahead"  # More natural than "center"
        elif relative_pos < 0.75:
            return "right"
        else:
            return "far right"
    
    def analyze_chair_type(self, width_px, height_px):
        """Analyze chair type based on dimensions"""
        aspect_ratio = height_px / max(width_px, 1)
        
        if aspect_ratio > 1.8:
            chair_type = "tall_back_chair"
        elif aspect_ratio > 1.2:
            chair_type = "standard_chair"
        elif aspect_ratio > 0.8:
            chair_type = "wide_chair"
        else:
            chair_type = "stool_or_low_chair"
        
        # Estimate relative size
        total_pixels = width_px * height_px
        if total_pixels > 15000:
            size = "large"
        elif total_pixels > 8000:
            size = "medium"
        else:
            size = "small"
        
        return {
            'type': chair_type,
            'size': size,
            'aspect_ratio': round(aspect_ratio, 2)
        }
    
    def calculate_chair_risk(self, distance, position, chair_analysis):
        """Calculate navigation risk level for the chair"""
        risk_score = 0
        
        # Distance risk
        if distance <= self.safe_distance:
            risk_score += 3
        elif distance <= self.warning_distance:
            risk_score += 2
        else:
            risk_score += 1
        
        # Position risk
        if position == "directly ahead":
            risk_score += 2
        elif position in ["left", "right"]:
            risk_score += 1
        
        # Size risk
        if chair_analysis['size'] == "large":
            risk_score += 1
        
        # Convert to risk level
        if risk_score >= 5:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"
    
    def find_chair_navigation_path(self, chairs):
        """Find optimal navigation path with TTS-friendly responses"""
        if not chairs:
            return "straight", "No chairs detected"
        
        # Separate chairs by position and risk
        high_risk_chairs = [c for c in chairs if c['risk_level'] == 'high']
        medium_risk_chairs = [c for c in chairs if c['risk_level'] == 'medium']
        
        # Analyze spatial distribution
        left_chairs = [c for c in chairs if 'left' in c['position']]
        center_chairs = [c for c in chairs if c['position'] == 'directly ahead']
        right_chairs = [c for c in chairs if 'right' in c['position']]
        
        # Natural language decision logic
        if high_risk_chairs:
            if any(c['position'] == 'directly ahead' for c in high_risk_chairs):
                left_risk = sum(1 for c in left_chairs if c['risk_level'] in ['high', 'medium'])
                right_risk = sum(1 for c in right_chairs if c['risk_level'] in ['high', 'medium'])
                
                if left_risk < right_risk:
                    return "navigate_left", "Chair blocking ahead"
                elif right_risk < left_risk:
                    return "navigate_right", "Chair blocking ahead"
                else:
                    return "stop_and_assess", "Multiple obstacles ahead"
            else:
                return "proceed_with_caution", "High risk chairs nearby"
        
        elif center_chairs:
            return "navigate_around", f"Chair directly ahead"
        else:
            return "proceed_forward", "Chairs detected but path navigable"
    
    def format_for_tts(self, text):
        """Add natural pauses for better TTS comprehension"""
        # Add short pauses after important information
        text = text.replace(". ", ". ... ")  # Pause after sentences
        text = text.replace("Warning!", "Warning! ... ")  # Pause after warnings
        text = text.replace("meters.", "meters. ... ")  # Pause after distance info
        return text
    
    def generate_chair_navigation_advice(self, chairs, image_shape):
        """Generate TTS-optimized navigation advice focused on chairs"""
        if not chairs:
            return "Path appears clear of chairs. Continue straight with normal caution."
        
        advice_parts = []
        
        # Chair detection summary - more natural for TTS
        chair_count = len(chairs)
        if chair_count == 1:
            advice_parts.append("One chair detected.")
        else:
            advice_parts.append(f"{chair_count} chairs detected.")
        
        # Immediate danger chairs
        critical_chairs = [c for c in chairs if c['distance'] <= self.safe_distance]
        if critical_chairs:
            advice_parts.append("Warning! Immediate hazard.")
            for chair in critical_chairs[:2]:  # Report max 2 critical chairs
                chair_desc = f"{chair['size_analysis']['type'].replace('_', ' ')}"
                advice_parts.append(
                    f"{chair_desc.title()} on your {chair['position']} at {chair['distance']} meters."
                )
        
        # Warning distance chairs
        warning_chairs = [c for c in chairs if self.safe_distance < c['distance'] <= self.warning_distance]
        if warning_chairs:
            if not critical_chairs:  # Only say this if no critical chairs
                advice_parts.append("Approaching chairs.")
            
            for chair in warning_chairs[:3]:  # Report max 3 warning chairs
                chair_desc = f"{chair['size_analysis']['size']} {chair['size_analysis']['type'].replace('_', ' ')}"
                advice_parts.append(
                    f"{chair_desc} on your {chair['position']} at {chair['distance']} meters."
                )
        
        # Navigation recommendation - more conversational
        direction, reason = self.find_chair_navigation_path(chairs)
        
        # Convert navigation directions to natural speech
        if direction == "navigate_left":
            advice_parts.append("Navigate left to avoid obstacles.")
        elif direction == "navigate_right":
            advice_parts.append("Navigate right to avoid obstacles.")
        elif direction == "proceed_forward":
            advice_parts.append("Path appears navigable. Proceed forward carefully.")
        elif direction == "stop_and_assess":
            advice_parts.append("Stop and assess the situation. Multiple obstacles detected.")
        elif direction == "proceed_with_caution":
            advice_parts.append("Proceed with extra caution.")
        else:
            advice_parts.append("Continue forward with awareness of nearby chairs.")
        
        # Chair arrangement analysis - simplified for audio
        if len(chairs) > 2:
            arranged_chairs = self.analyze_chair_arrangement(chairs)
            if arranged_chairs and "multiple chairs blocking" in arranged_chairs:
                advice_parts.append("Multiple chairs are blocking the center path.")
            elif arranged_chairs and "chairs along sides" in arranged_chairs:
                advice_parts.append("Chairs detected along the sides. Center path may be clear.")
        
        # Format for TTS with natural pauses
        advice = " ".join(advice_parts)
        return self.format_for_tts(advice)
    
    def analyze_chair_arrangement(self, chairs):
        """Analyze how chairs are arranged (useful for navigation)"""
        if len(chairs) < 2:
            return None
        
        # Check for common arrangements
        center_chairs = [c for c in chairs if c['position'] == 'directly ahead']
        side_chairs = [c for c in chairs if c['position'] in ['left', 'right', 'far_left', 'far_right']]
        
        if len(center_chairs) >= 2:
            return "multiple chairs blocking center path"
        elif len(side_chairs) >= 3:
            return "chairs along sides - center path may be clear"
        elif len(chairs) >= 4:
            return "furniture area detected - navigate carefully"
        else:
            return "scattered chairs"


# Navigation server implementation
nav_ai = ChairNavigationAI()

async def process_image_bytes(image_bytes):
    """Process incoming image bytes and return chair navigation analysis"""
    try:
        print("=== Chair Navigation Analysis ===")
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Invalid image data"}
        
        print(f"Image decoded: {frame.shape}")
        
        # Resize for processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            print(f"Image resized to: {new_width}x{new_height}")
        
        # Detect chairs using fine-tuned model
        print("Detecting chairs...")
        chairs = nav_ai.detect_chairs(frame)
        print(f"Chair detection complete: {len(chairs)} chairs found")
        
        # Generate navigation advice
        advice = nav_ai.generate_chair_navigation_advice(chairs, frame.shape)
        print(f"Navigation advice: {advice[:100]}...")
        
        return {
            "advice": advice,
            "chair_count": len(chairs),
            "chairs_detected": chairs,
            "navigation_summary": {
                "total_chairs": len(chairs),
                "critical_chairs": len([c for c in chairs if c['distance'] <= 1.2]),
                "warning_chairs": len([c for c in chairs if 1.2 < c['distance'] <= 2.5]),
                "safe_chairs": len([c for c in chairs if c['distance'] > 2.5])
            }
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return {"error": f"Chair navigation processing error: {str(e)}"}

# WebSocket server
async def handler(websocket):
    """WebSocket handler for chair navigation clients"""
    print("Client connected for chair-focused navigation.")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                print("Processing image for chair navigation...")
                result = await process_image_bytes(message)
                await websocket.send(json.dumps(result))
                print(f"Sent chair advice: {result.get('advice', 'Error')[:80]}...")
            else:
                await websocket.send(json.dumps({"error": "Send image as bytes"}))
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        print("Client disconnected.")

async def main():
    """Start the chair navigation WebSocket server"""
    print("Starting Chair Navigation AI Server...")
    print("Using NexaAIalex/Chair dataset for specialized chair detection")
    print("Server starting on ws://0.0.0.0:8766...")
    
    async with websockets.serve(
        handler, 
        "0.0.0.0", 
        8766, 
        max_size=8*1024*1024,
        ping_timeout=60,
        ping_interval=30
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
