"""
Standalone video processor - Runs in separate process to avoid memory issues
Creates output video WITH SKELETON OVERLAY
Usage: python video_processor.py <video_path> <exercise_type> <output_json_path> [output_video_path]
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import time
import cv2
import gc
import subprocess
import threading
from collections import deque
import numpy as np

# Force line-buffered stdout so every log() appears immediately in the parent log
sys.stdout.reconfigure(line_buffering=True)

def log(msg):
    print(msg, flush=True)

# Try to use imageio-ffmpeg (bundled ffmpeg binary) for H.264 output
try:
    import imageio_ffmpeg
    IMAGEIO_AVAILABLE = True
    log("imageio-ffmpeg available for H.264 output")
except ImportError:
    IMAGEIO_AVAILABLE = False
    log("imageio-ffmpeg not available, using OpenCV for output")


# --- Direct ffmpeg pipe writer ---------------------------------------------
# crf 23 + preset ultrafast (vs. the previous quality=8 -> crf 10 via imageio)
# to fit peak RAM/CPU inside the 1 GB container. Stderr is drained on a
# background thread so a full pipe can't stall ffmpeg, and its tail is kept
# for error reporting when the process dies (e.g. OOM-killed).

DOWNSCALE_OUTPUT = False          # toggle: shrink annotated output to save RAM/CPU
DOWNSCALE_TARGET = (720, 1280)    # (width, height) used when DOWNSCALE_OUTPUT is True

FFMPEG_STDERR_TAIL = 40  # lines of stderr kept for error reporting


class FFmpegWriter:
    """Pipes raw RGB24 frames into ffmpeg to produce an H.264 mp4.

    Exposes the same append_data()/close() interface the old imageio writer
    used, so call sites elsewhere in process_video don't need to change.
    """

    def __init__(self, output_path, width, height, fps):
        self.in_width = width
        self.in_height = height
        self.out_width, self.out_height = (
            DOWNSCALE_TARGET if DOWNSCALE_OUTPUT else (width, height)
        )

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', f'{fps:.2f}',
            '-i', '-',
        ]
        if DOWNSCALE_OUTPUT:
            cmd += ['-vf', f'scale={self.out_width}:{self.out_height}']
        cmd += [
            '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '23', '-preset', 'ultrafast',
            '-v', 'warning', output_path,
        ]

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        self._stderr_lines = deque(maxlen=FFMPEG_STDERR_TAIL)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        try:
            for line in iter(self._proc.stderr.readline, b''):
                self._stderr_lines.append(line.decode(errors='replace').rstrip())
        except Exception:
            pass

    def _stderr_tail(self):
        return '\n'.join(self._stderr_lines) or '(no ffmpeg stderr captured)'

    def append_data(self, frame):
        """frame must be RGB uint8 of (height, width, 3). Coerces/resizes
        anything else so one off-size frame can't misalign or break the pipe."""
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected HxWx3 frame, got shape {frame.shape}")
        if frame.shape[0] != self.in_height or frame.shape[1] != self.in_width:
            frame = cv2.resize(frame, (self.in_width, self.in_height))

        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            raise BrokenPipeError(
                f"ffmpeg pipe broke (likely killed/OOM). stderr tail:\n{self._stderr_tail()}"
            ) from e

    def close(self):
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except BrokenPipeError:
            pass
        returncode = self._proc.wait()
        self._stderr_thread.join(timeout=5)
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {returncode}. stderr tail:\n{self._stderr_tail()}"
            )


def draw_skeleton(frame, landmarks, mp_pose, mp_drawing):
    """Draw enhanced skeleton on frame with neon glow effect"""
    h, w = frame.shape[:2]
    
    # Define custom connections for cleaner skeleton
    # Body connections with different colors
    BODY_CONNECTIONS = [
        # Torso (cyan)
        (11, 12),  # Shoulders
        (11, 23),  # Left shoulder to hip
        (12, 24),  # Right shoulder to hip
        (23, 24),  # Hips
    ]
    
    ARM_CONNECTIONS = [
        # Left arm (green)
        (11, 13), (13, 15),  # Left arm
        # Right arm (green)  
        (12, 14), (14, 16),  # Right arm
    ]
    
    LEG_CONNECTIONS = [
        # Left leg (blue)
        (23, 25), (25, 27),  # Left leg
        # Right leg (blue)
        (24, 26), (26, 28),  # Right leg
    ]
    
    # Get landmark positions
    def get_pos(idx):
        lm = landmarks.landmark[idx]
        return (int(lm.x * w), int(lm.y * h))
    
    def is_visible(idx):
        return landmarks.landmark[idx].visibility > 0.5
    
    # Draw connections with glow effect
    def draw_line_with_glow(p1, p2, color, thickness=3):
        # Outer glow
        cv2.line(frame, p1, p2, (color[0]//3, color[1]//3, color[2]//3), thickness + 4)
        # Main line
        cv2.line(frame, p1, p2, color, thickness)
        # Inner bright line
        cv2.line(frame, p1, p2, (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50)), max(1, thickness-1))
    
    # Draw body (cyan)
    for start, end in BODY_CONNECTIONS:
        if is_visible(start) and is_visible(end):
            draw_line_with_glow(get_pos(start), get_pos(end), (255, 200, 0), 3)  # Cyan in BGR
    
    # Draw arms (green)
    for start, end in ARM_CONNECTIONS:
        if is_visible(start) and is_visible(end):
            draw_line_with_glow(get_pos(start), get_pos(end), (0, 255, 100), 3)
    
    # Draw legs (blue-purple)
    for start, end in LEG_CONNECTIONS:
        if is_visible(start) and is_visible(end):
            draw_line_with_glow(get_pos(start), get_pos(end), (255, 100, 100), 3)
    
    # Draw key joints with glow
    key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for idx in key_joints:
        if is_visible(idx):
            pos = get_pos(idx)
            # Outer glow
            cv2.circle(frame, pos, 8, (50, 50, 50), -1)
            # Middle ring
            cv2.circle(frame, pos, 6, (0, 200, 100), -1)
            # Inner dot
            cv2.circle(frame, pos, 3, (255, 255, 255), -1)
    
    return frame


def draw_stats_overlay(frame, stats):
    """Draw professional exercise stats overlay on frame"""
    h, w = frame.shape[:2]
    
    # Calculate overlay dimensions
    box_width = 320
    box_height = 180
    margin = 15
    padding = 12
    
    # Create semi-transparent overlay with rounded corners effect
    overlay = frame.copy()
    
    # Main background box
    cv2.rectangle(overlay, (margin, margin), (margin + box_width, margin + box_height), 
                  (30, 30, 30), -1)
    
    # Add accent line on left
    cv2.rectangle(overlay, (margin, margin), (margin + 5, margin + box_height), 
                  (0, 200, 100), -1)
    
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Fonts
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX
    
    # Calculate positions
    x_start = margin + padding + 8
    y_start = margin + 35
    line_height = 38
    
    # === REPS (Large, prominent) ===
    reps_text = f"{stats['reps']}"
    cv2.putText(frame, "REPS", (x_start, y_start - 8), 
                font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, reps_text, (x_start, y_start + 28), 
                font_bold, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    
    # === SCORE with color gradient based on value ===
    score = stats.get('form_score', 100)
    grade = stats.get('grade', 'A')
    
    # Color based on score
    if score >= 90:
        score_color = (0, 230, 118)  # Bright green
    elif score >= 75:
        score_color = (0, 200, 255)  # Gold/Yellow
    elif score >= 60:
        score_color = (0, 165, 255)  # Orange
    else:
        score_color = (60, 76, 231)  # Red
    
    # Score display (middle section)
    score_x = x_start + 90
    cv2.putText(frame, "SCORE", (score_x, y_start - 8), 
                font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(score)}", (score_x, y_start + 28), 
                font_bold, 1.4, score_color, 2, cv2.LINE_AA)
    
    # === GRADE (with badge style) ===
    grade_x = score_x + 90
    cv2.putText(frame, "GRADE", (grade_x, y_start - 8), 
                font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    
    # Grade badge background
    badge_x = grade_x
    badge_y = y_start + 5
    badge_size = 35
    cv2.rectangle(frame, (badge_x - 5, badge_y - 5), (badge_x + badge_size, badge_y + badge_size - 5), 
                  score_color, -1)
    cv2.putText(frame, grade, (badge_x + 5, badge_y + 22), 
                font_bold, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # === STATE (with icon-like indicator) ===
    state = stats.get('state', 'READY')
    if state is None or state == 'None':
        state = 'READY'
    
    # Map states to user-friendly names and colors
    state_info = {
        'up': ('UP', (0, 230, 118)),
        'down': ('DOWN', (255, 180, 0)),
        'UP': ('UP', (0, 230, 118)),
        'DOWN': ('DOWN', (255, 180, 0)),
        'hold': ('HOLD', (0, 200, 255)),
        'HOLD': ('HOLD', (0, 200, 255)),
        'READY': ('READY', (100, 100, 100)),
        'ready': ('READY', (100, 100, 100)),
        'UNKNOWN': ('READY', (100, 100, 100)),
    }
    
    state_display, state_color = state_info.get(state, (state.upper(), (180, 180, 180)))
    
    state_y = y_start + line_height + 25
    cv2.putText(frame, "STATE", (x_start, state_y), 
                font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    
    # State indicator dot
    dot_y = state_y + 20
    cv2.circle(frame, (x_start + 8, dot_y), 6, state_color, -1)
    cv2.putText(frame, state_display, (x_start + 22, dot_y + 5), 
                font, 0.65, state_color, 2, cv2.LINE_AA)
    
    # === FEEDBACK (if any) ===
    feedback = stats.get('feedback', '')
    if feedback and feedback.strip():
        feedback_y = state_y + line_height + 15
        # Truncate long feedback
        if len(feedback) > 40:
            feedback = feedback[:37] + "..."
        cv2.putText(frame, feedback, (x_start, feedback_y), 
                    font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    return frame


def process_video(video_path: str, exercise_type: str, output_json_path: str, output_video_path: str = None):
    """Process video, draw skeleton, and write results"""
    import mediapipe as mp
    from exercises.engine import ExerciseEngine
    
    results = {
        'status': 'processing',
        'progress': 0,
        'reps': 0,
        'form_score': 100,
        'avg_form_score': 100,
        'grade': 'A',
        'state': 'READY',
        'feedback': '',
        'error': None,
        'output_video': output_video_path
    }
    
    def save_results():
        with open(output_json_path, 'w') as f:
            json.dump(results, f)
    
    cap = None
    out = None
    pose = None
    imageio_writer = None
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            results['status'] = 'error'
            results['error'] = 'Could not open video file'
            save_results()
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        log(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
        
        # Create output video writer if path provided
        out = None
        
        if output_video_path:
            # Ensure .mp4 extension
            if not output_video_path.endswith('.mp4'):
                output_video_path = output_video_path.rsplit('.', 1)[0] + '.mp4'
            
            if IMAGEIO_AVAILABLE:
                # Direct ffmpeg pipe (crf 23 + ultrafast to fit the 1 GB container)
                try:
                    imageio_writer = FFmpegWriter(output_video_path, width, height, fps)
                    log(f"Using direct-ffmpeg H.264 writer: {output_video_path} "
                        f"(crf=23 preset=ultrafast downscale={'on' if DOWNSCALE_OUTPUT else 'off'})")
                except Exception as e:
                    log(f"ffmpeg writer init failed: {e}, will use OpenCV")
                    imageio_writer = None
            
            if not imageio_writer:
                # Fallback to OpenCV
                codecs_to_try = [
                    ('avc1', '.mp4'),  # H.264 - best for web
                    ('H264', '.mp4'),  # Alternative H.264
                    ('XVID', '.avi'),  # Fallback
                    ('mp4v', '.mp4'),  # Last resort
                ]
                
                for codec, ext in codecs_to_try:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        if not output_video_path.endswith(ext):
                            output_video_path = output_video_path.rsplit('.', 1)[0] + ext
                        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        if out.isOpened():
                            log(f"Using OpenCV codec: {codec}")
                            break
                        out.release()
                        out = None
                    except:
                        continue
                
                if not out or not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    log("Using fallback codec: mp4v")
            
            # Update results with actual output path
            results['output_video'] = output_video_path
            log(f"Output video: {output_video_path}")
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        pose = mp_pose.Pose(
            static_image_mode=False,  # Video mode for better tracking
            model_complexity=1,  # Better accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        log("MediaPipe Pose initialized")
        
        # Initialize exercise engine
        engine = ExerciseEngine()
        if not engine.set_exercise(exercise_type):
            log(f"WARNING: Failed to load exercise: {exercise_type}")
        else:
            log(f"Exercise loaded: {exercise_type}")

        import math as _math

        def _angle3(a, b, c):
            """2-D angle at vertex b (uses landmark .x/.y)."""
            ax, ay = a.x - b.x, a.y - b.y
            cx, cy = c.x - b.x, c.y - b.y
            dot = ax * cx + ay * cy
            mag = (_math.sqrt(ax*ax + ay*ay) * _math.sqrt(cx*cx + cy*cy)) + 1e-6
            return _math.degrees(_math.acos(max(-1.0, min(1.0, dot / mag))))

        # Wrong-exercise detection — track knee angle and body orientation.
        knee_angles_tracked = []
        body_orientation_tracked = []  # 'standing' or 'lying'

        frame_count = 0
        analyze_skip = max(1, int(fps / 8))  # Analyze at ~8 fps
        log(f"Analyze skip: {analyze_skip} (analyzing at ~{fps/analyze_skip:.1f} fps)")

        # Deadlift: collect shoulder Y series for post-processing rep count.
        # Real-time thresholds are unreliable (depend on camera distance, person height).
        # Post-processing uses the actual min/max of the signal → self-calibrating.
        dl_shoulder_series = []   # raw shoulder Y per analyzed frame
        dl_reps = 0

        # Current stats for overlay
        current_stats = {
            'reps': 0,
            'form_score': 100,
            'grade': 'A',
            'state': 'READY',
            'feedback': ''
        }

        analyzed_frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            results['progress'] = int((frame_count / total_frames) * 100)

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                # Draw skeleton on frame
                frame = draw_skeleton(frame, pose_results.pose_landmarks, mp_pose, mp_drawing)

                # Analyze exercise periodically
                if frame_count % analyze_skip == 0:
                    lm = pose_results.pose_landmarks.landmark
                    analyzed_frame_idx += 1

                    # ── Knee angle (for wrong-exercise detection) ─────────────
                    lv = min(lm[23].visibility, lm[25].visibility, lm[27].visibility)
                    rv = min(lm[24].visibility, lm[26].visibility, lm[28].visibility)
                    if lv > 0.3 or rv > 0.3:
                        l_ka = _angle3(lm[23], lm[25], lm[27])
                        r_ka = _angle3(lm[24], lm[26], lm[28])
                        knee_angles_tracked.append(l_ka if lv >= rv else r_ka)

                    # ── Body orientation (standing vs lying) ──────────────────
                    # MediaPipe y increases downward. For a standing person,
                    # shoulder_y < hip_y. For someone lying flat, shoulder_y ≈ hip_y
                    # and the difference in x is large instead.
                    sh_vis = (lm[11].visibility + lm[12].visibility) / 2
                    hp_vis = (lm[23].visibility + lm[24].visibility) / 2
                    if sh_vis > 0.4 and hp_vis > 0.4:
                        sh_y = (lm[11].y + lm[12].y) / 2
                        hp_y = (lm[23].y + lm[24].y) / 2
                        sh_x = (lm[11].x + lm[12].x) / 2
                        hp_x = (lm[23].x + lm[24].x) / 2
                        dy = abs(hp_y - sh_y)
                        dx = abs(hp_x - sh_x)
                        # Standing: large vertical separation (dy > dx)
                        # Lying: large horizontal separation (dx > dy) or dy very small
                        if dy > 0.12 and dy > dx * 0.8:
                            body_orientation_tracked.append('standing')
                        elif dx > dy or dy < 0.08:
                            body_orientation_tracked.append('lying')

                    # ── Collect shoulder Y for deadlift post-processing ───────
                    if exercise_type == 'deadlift':
                        ls_v = (lm[11].visibility + lm[12].visibility) / 2
                        if ls_v > 0.1:
                            sy = (lm[11].y + lm[12].y) / 2
                            dl_shoulder_series.append(sy)

                    # ── YAML engine (form score + squat rep counting) ─────────
                    engine.process_frame(frame, lm)
                    status = engine.get_status()

                    engine_reps = status.get('counter', 0)
                    current_stats['reps'] = 0 if exercise_type == 'deadlift' else engine_reps
                    current_stats['form_score'] = status.get('form_score', 100)
                    current_stats['grade'] = status.get('form_grade', 'A')
                    current_stats['state'] = status.get('current_state', 'UNKNOWN')
                    current_stats['feedback'] = status.get('feedback', '')

                    results['reps'] = current_stats['reps']
                    results['form_score'] = current_stats['form_score']
                    results['avg_form_score'] = status.get('avg_form_score', 100)
                    results['grade'] = current_stats['grade']
                    results['state'] = current_stats['state']
                    results['feedback'] = current_stats['feedback']

                    if analyzed_frame_idx % 30 == 0:
                        log(f"[Frame {frame_count}] engine_reps={engine_reps} state={current_stats['state']} dl_samples={len(dl_shoulder_series)}")
            
            # Draw stats overlay
            frame = draw_stats_overlay(frame, current_stats)
            
            # Write frame to output video
            if imageio_writer:
                # Convert BGR to RGB for imageio
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imageio_writer.append_data(frame_rgb)
            elif out:
                out.write(frame)
            
            # Save intermediate results
            if frame_count % 60 == 0:
                save_results()
            
            # Memory management
            del rgb_frame
            if frame_count % 100 == 0:
                gc.collect()
        
        # Cleanup video capture
        if cap:
            cap.release()
        if pose:
            pose.close()
        
        # ── Deadlift post-processing rep count ───────────────────────────────
        # Count reps from the shoulder-Y series collected during the video.
        # Algorithm:
        #   1. Smooth with a 5-sample moving average
        #   2. Find actual min (standing = shoulder highest) and max (bent)
        #   3. Set threshold = midpoint between min and max
        #   4. Count each downward crossing of the threshold as 1 rep
        #      (bent phase started) — matches 1 full up→down→up cycle
        if exercise_type == 'deadlift':
            if len(dl_shoulder_series) >= 6:
                w = 5
                smoothed = []
                for i in range(len(dl_shoulder_series)):
                    chunk = dl_shoulder_series[max(0, i - w):i + w + 1]
                    smoothed.append(sum(chunk) / len(chunk))

                sig_min = min(smoothed)   # standing position (shoulder highest = smallest y)
                sig_max = max(smoothed)   # bent position (shoulder lowest = largest y)
                sig_range = sig_max - sig_min
                log(f"[DL] shoulder_y min={sig_min:.3f} max={sig_max:.3f} range={sig_range:.3f} samples={len(smoothed)}")

                # Real deadlifts produce shoulder-Y range of 15-20% of screen height.
                # Minor body sway / walking into frame / breathing produces < 8%.
                if sig_range >= 0.08:
                    threshold = sig_min + sig_range * 0.5
                    # 2-state machine with dual minimum-duration guards:
                    #   MIN_STANDING: must be below threshold for this many samples
                    #                 before a new bend phase can begin (prevents the
                    #                 person walking into frame from counting as rep 1).
                    #   MIN_BENT:     must stay above threshold this many samples
                    #                 before a return-to-standing counts as a rep.
                    MIN_STANDING = 3    # ~0.3 s — min lockout before next rep starts
                    MIN_BENT = 5        # ~0.5 s — min bend duration for valid rep
                    MIN_REP_INTERVAL = 15  # ~1.5 s — min samples between rep counts
                    in_bent = False
                    bent_count = 0
                    standing_count = MIN_STANDING  # assume person starts standing
                    dl_reps = 0
                    last_rep_idx = -MIN_REP_INTERVAL
                    for idx, val in enumerate(smoothed):
                        if val > threshold:
                            if in_bent:
                                bent_count += 1
                            else:
                                if standing_count >= MIN_STANDING:
                                    in_bent = True
                                    bent_count = 1
                                    standing_count = 0
                        else:
                            if in_bent:
                                if bent_count >= MIN_BENT:
                                    if (idx - last_rep_idx) >= MIN_REP_INTERVAL:
                                        dl_reps += 1
                                        last_rep_idx = idx
                                        log(f"[DL] rep {dl_reps} at sample {idx} (bent {bent_count} samples)")
                                    else:
                                        log(f"[DL] skipped fast rep at sample {idx} ({idx - last_rep_idx} samples since last)")
                                in_bent = False
                                bent_count = 0
                            standing_count += 1
                    log(f"[DL] post-processing → {dl_reps} rep(s) at threshold={threshold:.3f}")
                else:
                    log(f"[DL] movement too small ({sig_range:.3f} < 0.08) — 0 reps")
            else:
                log(f"[DL] not enough shoulder samples ({len(dl_shoulder_series)}) — 0 reps")

            results['reps'] = dl_reps

        # ── Write final stats ─────────────────────────────────────────────────
        if exercise_type != 'deadlift':
            results['reps'] = current_stats['reps']
        results['form_score'] = current_stats['form_score']
        results['grade'] = current_stats['grade']
        results['state'] = 'COMPLETED'
        results['feedback'] = current_stats['feedback']

        # ── Wrong-exercise detection ──────────────────────────────────────────
        # Only check squat: if someone selected squat but never bent knees below 105°
        # (avg stays above 145°), they likely did a deadlift instead.
        #
        # Deadlift wrong-exercise is intentionally NOT checked here: from a front-view
        # camera, the 2D projection of the hip-hinge creates phantom knee angles that
        # make a real deadlift appear to have deep knee bends. The knee angle signal
        # is unreliable for deadlift and causes too many false positives.
        results['wrong_exercise'] = False
        results['wrong_exercise_message'] = ''

        # Standing exercises that require an upright body
        STANDING_EXERCISES = {
            'squat', 'deadlift', 'lunge', 'side_lunge', 'calf_raise',
            'high_knees', 'jumping_jack', 'mountain_climber', 'wall_sit',
            'shoulder_press', 'lateral_raise', 'bicep_curl', 'hammer_curl',
            'tricep_dip', 'leg_raise'
        }
        # Exercises done lying/horizontal
        LYING_EXERCISES = {'bench_press', 'push_up', 'plank', 'glute_bridge'}

        # ── Body orientation check ────────────────────────────────────────────
        if body_orientation_tracked:
            n_orient = len(body_orientation_tracked)
            lying_ratio = body_orientation_tracked.count('lying') / n_orient
            log(f"[WrongExercise] orientation — lying_ratio={lying_ratio:.2f} samples={n_orient}")

            if exercise_type in STANDING_EXERCISES and lying_ratio > 0.5:
                results['wrong_exercise'] = True
                results['wrong_exercise_message'] = (
                    f"Wrong exercise: you appear to be lying down but selected '{exercise_type}' "
                    f"which requires a standing position. Did you mean bench_press or push_up?"
                )
                results['reps'] = 0
                log(f"[WrongExercise] DETECTED (lying down for standing exercise)")

            elif exercise_type in LYING_EXERCISES and lying_ratio < 0.3:
                results['wrong_exercise'] = True
                results['wrong_exercise_message'] = (
                    f"Wrong exercise: you appear to be standing but selected '{exercise_type}' "
                    f"which requires a lying/horizontal position."
                )
                results['reps'] = 0
                log(f"[WrongExercise] DETECTED (standing for lying exercise)")

        # ── Knee angle check for squat (secondary check) ─────────────────────
        if not results['wrong_exercise'] and exercise_type == 'squat' and knee_angles_tracked:
            n = len(knee_angles_tracked)
            avg_knee = sum(knee_angles_tracked) / n
            min_knee = min(knee_angles_tracked)
            log(f"[WrongExercise] squat knee check — avg={avg_knee:.1f}° min={min_knee:.1f}° samples={n}")
            if avg_knee > 145 and min_knee > 120:
                results['wrong_exercise'] = True
                results['wrong_exercise_message'] = (
                    f"Wrong exercise: knees never bent deeply enough for a squat "
                    f"(avg {avg_knee:.0f}°, min {min_knee:.0f}°) — did you mean deadlift?"
                )
                results['reps'] = 0
                log(f"[WrongExercise] DETECTED — {results['wrong_exercise_message']}")
        
        # Close video writers
        if imageio_writer:
            imageio_writer.close()  # raises RuntimeError w/ ffmpeg stderr on non-zero exit
            log(f"H.264 video saved: {output_video_path}")
        if out:
            out.release()
        
        gc.collect()
        
        results['status'] = 'completed'
        results['progress'] = 100
        
        # Debug: Print final values
        log(f"=== FINAL RESULTS ===")
        log(f"Reps from current_stats: {current_stats['reps']}")
        log(f"Reps written to results: {results['reps']}")
        log(f"Form Score: {results['form_score']}")
        log(f"Grade: {results['grade']}")
        log(f"State: {results['state']}")
        
        # Get final status from engine for verification
        final_status = engine.get_status()
        log(f"Engine final counter: {final_status.get('counter', 'N/A')}")
        log(f"Engine final state: {final_status.get('current_state', 'N/A')}")
        if final_status.get('counter_left') is not None:
            log(f"Engine counter_left: {final_status.get('counter_left')}")
            log(f"Engine counter_right: {final_status.get('counter_right')}")
        log(f"=====================")
        
        save_results()
        
        log(f"Completed: {frame_count} frames, {results['reps']} reps")
        if output_video_path:
            log(f"Output video saved: {output_video_path}")
        
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        save_results()
        log(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap:
            try:
                cap.release()
            except:
                pass
        if imageio_writer:
            try:
                imageio_writer.close()
            except:
                pass
        if out:
            try:
                out.release()
            except:
                pass
        if pose:
            try:
                pose.close()
            except:
                pass
        gc.collect()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        log("Usage: python video_processor.py <video_path> <exercise_type> <output_json_path> [output_video_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    exercise_type = sys.argv[2]
    output_json_path = sys.argv[3]
    output_video_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    process_video(video_path, exercise_type, output_json_path, output_video_path)
