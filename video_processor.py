"""
Standalone video processor - Runs in separate process to avoid memory issues
Creates output video WITH SKELETON OVERLAY
Usage: python video_processor.py <video_path> <exercise_type> <output_json_path> [output_video_path] [bench_check]
  bench_check: 'chest_touch' | 'butt_lift' | 'both'  (default: 'both')
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

DOWNSCALE_OUTPUT = True           # toggle: shrink annotated output to save RAM/CPU
DOWNSCALE_TARGET = (640, 480)     # (width, height) used when DOWNSCALE_OUTPUT is True

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
        analyze_skip = max(1, int(fps / 5))  # Analyze at ~5 fps to save memory on server
        log(f"Analyze skip: {analyze_skip} (analyzing at ~{fps/analyze_skip:.1f} fps)")

        # Deadlift: collect shoulder Y series for post-processing rep count.
        # Real-time thresholds are unreliable (depend on camera distance, person height).
        # Post-processing uses the actual min/max of the signal → self-calibrating.
        dl_shoulder_series = []   # raw shoulder Y per analyzed frame
        dl_reps = 0

        # Bench press: collect elbow angle series for post-processing rep count.
        # Same motivation as deadlift — fixed thresholds miss reps or count re-racks.
        bp_angle_series = []
        bp_reps = 0
        # (elbow_angle, hip_y, knee_y, sh_y) tuples collected each frame where arm
        # angle, hip AND knee landmarks are all visible.  sh_y is used at post-
        # processing time to filter out "sitting up" frames that contaminate stats.
        bp_hip_frames = []   # list of (angle, hip_y) tuples for butt-lift detection
        # Shoulder visibility asymmetry per analyzed frame — used to detect camera
        # position.  Side view: one shoulder dominates (high asymmetry).
        # Front/feet view: both shoulders similar (low asymmetry).
        bp_sh_vis_series = []
        # (elbow_y - wrist_y) aligned with bp_angle_series — measures forearm
        # orientation.  Large when bar is high (lockout), near-zero when bar is at
        # chest (forearm vertical from side view).
        bp_wrist_elbow_series = []
        # Per-rep values computed in the post-processing state machine:
        bp_rep_rise_after_min = []   # angle rise in first frame after the minimum
        bp_rep_wrist_elbow_gaps = [] # wrist-elbow gap at the minimum angle frame

        # Current stats for overlay
        current_stats = {
            'reps': 0,
            'form_score': 100,
            'grade': 'A',
            'state': 'READY',
            'feedback': ''
        }

        analyzed_frame_idx = 0
        # Skip the first second of the video — MediaPipe landmark confidence is
        # unreliable while it initialises on a new video, and the lifter is
        # usually still getting into position. Skipping avoids these low-quality
        # frames contaminating the angle series and causing false 0-rep results.
        warmup_frames = int(fps)   # ≈ 1 second worth of frames

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

                # Analyze exercise periodically — skip the warmup window so
                # MediaPipe has time to lock onto the pose before we collect data.
                if frame_count % analyze_skip == 0 and frame_count > warmup_frames:
                    lm = pose_results.pose_landmarks.landmark
                    analyzed_frame_idx += 1

                    # ── Knee angle (for wrong-exercise detection) ─────────────
                    # Use hip+knee visibility only — ankle is often cut off when
                    # the camera frames the upper body, which would zero out min()
                    # and prevent any knee data from being collected.
                    lv = min(lm[23].visibility, lm[25].visibility)
                    rv = min(lm[24].visibility, lm[26].visibility)
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
                        # Standing: large vertical separation (dy clearly dominates dx)
                        if dy > 0.10 and dy > dx * 0.7:
                            body_orientation_tracked.append('standing')
                        # Lying: shoulders and hips clearly horizontal (dx much larger than dy)
                        # Note: dy < 0.08 alone is NOT enough — squat bottom also has small dy
                        elif dx > dy * 1.5 and dy < 0.08:
                            body_orientation_tracked.append('lying')

                    # ── Collect shoulder Y for deadlift post-processing ───────
                    if exercise_type == 'deadlift':
                        ls_v = (lm[11].visibility + lm[12].visibility) / 2
                        if ls_v > 0.1:
                            sy = (lm[11].y + lm[12].y) / 2
                            dl_shoulder_series.append(sy)

                    # ── YAML engine (form score + squat rep counting) ─────────
                    video_time = frame_count / fps
                    engine.process_frame(frame, lm, video_time)
                    status = engine.get_status()

                    engine_reps = status.get('counter', 0)

                    # Collect average elbow angle for bench press (both arms)
                    if exercise_type == 'bench_press':
                        # Left arm:  shoulder=11, elbow=13, wrist=15
                        # Right arm: shoulder=12, elbow=14, wrist=16
                        l_vis = min(lm[11].visibility, lm[13].visibility, lm[15].visibility)
                        r_vis = min(lm[12].visibility, lm[14].visibility, lm[16].visibility)

                        # Shoulder visibility asymmetry — high when side-view camera
                        # (one shoulder faces lens, other is occluded), low for front/feet.
                        bp_sh_vis_series.append(abs(lm[11].visibility - lm[12].visibility))

                        _angles = []
                        if l_vis > 0.3:
                            _angles.append(_angle3(lm[11], lm[13], lm[15]))
                        if r_vis > 0.3:
                            _angles.append(_angle3(lm[12], lm[14], lm[16]))
                        if _angles:
                            _avg_angle = sum(_angles) / len(_angles)
                            bp_angle_series.append(_avg_angle)

                            # Forearm orientation signal — elbow_y minus wrist_y.
                            # Large at lockout (wrist far above elbow), near-zero when
                            # bar reaches chest (forearm becomes vertical, side view).
                            # Use the more-visible arm to avoid occluded landmark noise.
                            if l_vis >= r_vis and l_vis > 0.3:
                                bp_wrist_elbow_series.append(lm[13].y - lm[15].y)
                            elif r_vis > l_vis and r_vis > 0.3:
                                bp_wrist_elbow_series.append(lm[14].y - lm[16].y)
                            else:
                                bp_wrist_elbow_series.append(None)

                            # Collect (angle, hip_y) for butt-lift detection.
                            if hp_vis > 0.1:
                                bp_hip_frames.append((_avg_angle, (lm[23].y + lm[24].y) / 2))

                        # ── Per-frame landmark console dump (bench press) ─────────
                        # Printed every analyzed frame so you can watch landmark
                        # positions in real time and calibrate detection thresholds.
                        _L  = lm  # shorthand
                        _kn_vis = (_L[25].visibility + _L[26].visibility) / 2
                        _ak_vis = (_L[27].visibility + _L[28].visibility) / 2
                        _elbow_angle_str = f"{_avg_angle:.1f}" if _angles else "N/A"
                        _hip_y_str  = f"{(_L[23].y+_L[24].y)/2:.3f}" if hp_vis > 0.05 else "hidden"
                        _knee_y_str = f"{(_L[25].y+_L[26].y)/2:.3f}" if _kn_vis > 0.05 else "hidden"
                        _ank_y_str  = f"{(_L[27].y+_L[28].y)/2:.3f}" if _ak_vis > 0.05 else "hidden"
                        # print(
                        #     f"[LM f={frame_count:04d}] "
                        #     f"L_sh=({_L[11].x:.2f},{_L[11].y:.2f}) vis={_L[11].visibility:.2f} | "
                        #     f"R_sh=({_L[12].x:.2f},{_L[12].y:.2f}) vis={_L[12].visibility:.2f} | "
                        #     f"L_el=({_L[13].x:.2f},{_L[13].y:.2f}) vis={_L[13].visibility:.2f} | "
                        #     f"R_el=({_L[14].x:.2f},{_L[14].y:.2f}) vis={_L[14].visibility:.2f} | "
                        #     f"L_wr=({_L[15].x:.2f},{_L[15].y:.2f}) vis={_L[15].visibility:.2f} | "
                        #     f"R_wr=({_L[16].x:.2f},{_L[16].y:.2f}) vis={_L[16].visibility:.2f} | "
                        #     f"hip_y={_hip_y_str} (vis={hp_vis:.2f}) | "
                        #     f"knee_y={_knee_y_str} (vis={_kn_vis:.2f}) | "
                        #     f"ank_y={_ank_y_str} (vis={_ak_vis:.2f}) | "
                        #     f"elbow_angle={_elbow_angle_str}",
                        #     flush=True
                        # )

                    current_stats['reps'] = 0 if exercise_type in ('deadlift', 'bench_press') else engine_reps
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

        # ── Bench press post-processing rep count ─────────────────────────────
        # State machine using average of both elbow angles:
        #   DOWN  : avg elbow angle < 105°  (bar descending toward chest)
        #   VALID : minimum angle during down phase must reach ≤ 100°
        #           (bar close enough to chest — matches YAML min_depth_angle)
        #   UP    : avg elbow angle > 140° while in DOWN → count rep ONLY if valid
        #
        # Separating the "entered down state" threshold (105°) from the "valid depth"
        # threshold (100°) means the state machine can still track the rep arc even
        # on imperfect camera angles, while only awarding a rep count when the
        # dumbbells genuinely approached the chest.
        # The form-check section below records every down phase (including shallow ones)
        # so form feedback is still generated for partial reps.
        BP_CHEST_DEPTH_ANGLE = 100  # elbow must reach this or lower to count the rep
        if exercise_type == 'bench_press':
            bp_rep_min_angles = []    # minimum elbow angle per rep (includes partial reps)
            bp_rep_slices = []        # (start_idx, end_idx) into bp_angle_series per rep
            if len(bp_angle_series) >= 2:
                bp_min = min(bp_angle_series)
                bp_max = max(bp_angle_series)
                log(f"[BP] samples={len(bp_angle_series)} min={bp_min:.1f}° max={bp_max:.1f}°")
                bp_state = 'up'
                bp_current_min = 180.0
                bp_current_min_idx = 0   # index into bp_angle_series where the minimum occurred
                bp_rep_start = 0
                for i, val in enumerate(bp_angle_series):
                    if val < 105.0:
                        bp_state = 'down'
                        if val < bp_current_min:
                            bp_current_min = val
                            bp_current_min_idx = i
                    elif val > 140.0 and bp_state == 'down':
                        bp_rep_min_angles.append(bp_current_min)
                        # Rise in the first analyzed frame after the minimum:
                        # small (+7-9°) when bar paused at chest, large (+20-25°)
                        # when bar immediately reversed without touching.
                        if bp_current_min_idx + 1 < len(bp_angle_series):
                            _rise = bp_angle_series[bp_current_min_idx + 1] - bp_current_min
                        else:
                            _rise = 0.0
                        bp_rep_rise_after_min.append(_rise)
                        # Forearm orientation at the bottom of the rep.
                        _we = (bp_wrist_elbow_series[bp_current_min_idx]
                               if bp_current_min_idx < len(bp_wrist_elbow_series) else None)
                        bp_rep_wrist_elbow_gaps.append(_we)
                        bp_rep_slices.append((bp_rep_start, i))
                        bp_rep_start = i          # next rep reuses this lockout frame as its baseline
                        bp_state = 'up'
                        # Only award a rep if the bar reached close enough to the chest.
                        if bp_current_min <= BP_CHEST_DEPTH_ANGLE:
                            bp_reps += 1
                        else:
                            log(f"[BP] partial rep NOT counted — min angle {bp_current_min:.0f}° > {BP_CHEST_DEPTH_ANGLE}°")
                        bp_current_min = 180.0
                        bp_current_min_idx = 0
                log(f"[BP] post-processing → {bp_reps} valid rep(s) | per-rep mins={[f'{a:.0f}' for a in bp_rep_min_angles]}")
                # Fall back to engine counter if post-processing missed reps
                # (camera angle can compress elbow angles beyond our thresholds)
                if bp_reps == 0 and engine_reps > 0:
                    log(f"[BP] post-processing got 0 reps, falling back to engine count: {engine_reps}")
                    bp_reps = engine_reps
            else:
                log(f"[BP] not enough angle samples ({len(bp_angle_series)}), using engine count")
                bp_reps = engine_reps

            results['reps'] = bp_reps
            current_stats['reps'] = bp_reps

        # ── Write final stats ─────────────────────────────────────────────────
        if exercise_type not in ('deadlift', 'bench_press'):
            results['reps'] = current_stats['reps']
        results['form_score'] = current_stats['form_score']
        results['grade'] = current_stats['grade']
        results['state'] = 'COMPLETED'
        results['feedback'] = current_stats['feedback']

        # ── Exercise movement pattern validation ──────────────────────────────
        # Validates that the recorded movement actually matches the selected exercise.
        # Catches random videos, wrong exercise selection, and no movement at all.
        results['wrong_exercise'] = False
        results['wrong_exercise_message'] = ''

        LYING_EXERCISES = {'bench_press', 'push_up', 'plank', 'glute_bridge'}

        lying_ratio = 0.0
        if body_orientation_tracked:
            lying_ratio = body_orientation_tracked.count('lying') / len(body_orientation_tracked)
            log(f"[Validate] lying_ratio={lying_ratio:.2f} samples={len(body_orientation_tracked)}")

        def _fail(msg):
            results['wrong_exercise'] = True
            results['wrong_exercise_message'] = msg
            results['reps'] = 0
            log(f"[Validate] FAILED — {msg}")

        # ── Squat: must show actual knee bend (ROM ≥ 35° and min angle < 135°) ──
        if exercise_type == 'squat':
            if lying_ratio > 0.75:
                _fail("You appear to be lying down. Stand upright to perform a squat.")
            elif knee_angles_tracked:
                min_knee = min(knee_angles_tracked)
                knee_rom = max(knee_angles_tracked) - min_knee
                log(f"[Validate] squat — min_knee={min_knee:.1f}° ROM={knee_rom:.1f}°")
                if min_knee > 135 or knee_rom < 35:
                    _fail(
                        f"No squat movement detected (knee ROM {knee_rom:.0f}°, min {min_knee:.0f}°). "
                        f"Bend your knees deeply — go below parallel."
                    )
            else:
                if results['reps'] == 0:
                    _fail("Could not detect your legs. Make sure your full body is visible in the frame.")

        # ── Bench press: must be lying down AND show elbow movement ──────────
        elif exercise_type == 'bench_press':
            # Only flag "standing" when we have enough orientation samples to be
            # confident AND no reps were counted. Side-view cameras often produce
            # very few orientation samples (dx ≈ 0 for a body parallel to lens),
            # which collapses lying_ratio to 0.0 and falsely triggers this flag.
            if lying_ratio < 0.25 and len(body_orientation_tracked) > 10 and results['reps'] == 0:
                _fail("You appear to be standing. Lie on the bench to perform bench press.")
            elif bp_angle_series:
                elbow_rom = max(bp_angle_series) - min(bp_angle_series)
                log(f"[Validate] bench — elbow_ROM={elbow_rom:.1f}°")
                if elbow_rom < 25 and results['reps'] == 0:
                    _fail(
                        f"No bench press movement detected (elbow ROM {elbow_rom:.0f}°). "
                        f"Lower the bar to your chest and press back up."
                    )
            else:
                if results['reps'] == 0:
                    _fail("Could not detect arm movement. Make sure your arms are visible in the frame.")

        # ── Deadlift: must show significant shoulder vertical movement ────────
        elif exercise_type == 'deadlift':
            if lying_ratio > 0.75:
                _fail("You appear to be lying down. Stand upright to perform a deadlift.")
            elif dl_shoulder_series:
                dl_range = max(dl_shoulder_series) - min(dl_shoulder_series)
                log(f"[Validate] deadlift — shoulder_range={dl_range:.3f}")
                if dl_range < 0.06 and results['reps'] == 0:
                    _fail(
                        f"No deadlift movement detected (shoulder range {dl_range:.3f}). "
                        f"Hinge at the hips and lift the bar from the floor."
                    )
            else:
                if results['reps'] == 0:
                    _fail("Could not detect your movement. Make sure your full body is visible.")
        
        # ── Bench press form quality checks ──────────────────────────────────
        # Runs after reps + validation so it only fires on confirmed bench press sets.
        if exercise_type == 'bench_press' and not results.get('wrong_exercise'):
            bp_form_issues = []
            chest_bad_count = 0

            # ── Check: Barbell not touching chest ────────────────────────────
            # Only runs when the user selected chest_touch or both.
            # A rep is flagged when the elbow angle never reached 100° (bar stopped
            # above chest level).  Only fires when EVERY rep missed depth so a
            # mixed set (some good, some shallow) still gets GOOD LIFT.
            if bench_check in ('chest_touch', 'both') and bp_rep_min_angles:
                # Exclude partial reps (min angle > 100°) — the engine didn't count
                # them either, and they pollute the check (e.g. 103° triggers hard
                # threshold even though the rep was not a real lift).
                _valid_idxs = [i for i, a in enumerate(bp_rep_min_angles) if a <= 100]
                valid_mins  = [bp_rep_min_angles[i] for i in _valid_idxs]
                valid_rises = [bp_rep_rise_after_min[i] if i < len(bp_rep_rise_after_min) else 0.0
                               for i in _valid_idxs]

                if valid_mins:
                    # Chest touch check — calibrated from observed data:
                    # • angle > 75°        → bar clearly stopped above chest (always a miss)
                    # • 68° < angle ≤ 75° → borderline; miss if rise > 50° (fast reversal = no contact)
                    # • 55° < angle ≤ 68° → deep safe zone; protected unless rise > 55°
                    #   (rise > 55° at this depth = fast reversal, bar likely never rested on chest)
                    # • angle ≤ 55°       → very deep safe zone; bar is at/past chest level.
                    #   Only flag if rise > 75° — explosive reps from genuine contact can
                    #   legitimately produce high rises at this depth.
                    CHEST_ANGLE_HARD         = 75
                    CHEST_ANGLE_COMBO        = 68
                    CHEST_ANGLE_DEEP         = 55   # splits shallow/deep safe zone
                    CHEST_RISE_FAST          = 50   # combo zone threshold
                    CHEST_RISE_EXTREME       = 55   # shallow safe zone (55–68°) threshold
                    CHEST_RISE_EXTREME_DEEP  = 75   # deep safe zone (≤55°) threshold

                    chest_bad_reps = []
                    for i, (a, rise) in enumerate(zip(valid_mins, valid_rises)):
                        if a > CHEST_ANGLE_HARD:
                            chest_bad_reps.append(i)
                        elif a > CHEST_ANGLE_COMBO:
                            if rise > CHEST_RISE_FAST:
                                chest_bad_reps.append(i)
                        elif a > CHEST_ANGLE_DEEP:
                            # shallow safe zone
                            if rise > CHEST_RISE_EXTREME:
                                chest_bad_reps.append(i)
                        else:
                            # deep safe zone — very high confidence of chest contact
                            if rise > CHEST_RISE_EXTREME_DEEP:
                                chest_bad_reps.append(i)

                    chest_bad_count = len(chest_bad_reps)
                    log(f"[BP-Form] bench_check={bench_check} valid_reps={len(valid_mins)} "
                        f"per-rep mins={[f'{a:.0f}' for a in valid_mins]} "
                        f"rises={[f'{r:.1f}' for r in valid_rises]} "
                        f"chest_bad={chest_bad_reps} ({chest_bad_count}/{len(valid_mins)} reps)")
                    # Flag when ≥ 50% of valid reps failed chest touch.
                    if chest_bad_count >= len(valid_mins) / 2:
                        bp_form_issues.append("Not touching chest — lower bar all the way down")

            if bp_form_issues:
                results['feedback'] = bp_form_issues[0]
                results['form_score'] = 0
                results['avg_form_score'] = 0
                results['grade'] = 'F'
                log(f"[BP-Form] chest_bad={chest_bad_count} → score forced to 0 (NO LIFT)")

            # ── Butt lift detection ───────────────────────────────────────────
            # Compare hip_y at lockout (angle > 140°, butt should be on bench)
            # against hip_y during the press (angle < 110°, where bridging happens).
            # MediaPipe Y=0=top, Y=1=bottom. Butt lift → hip rises → hip_y decreases.
            # If avg hip_y during press is lower than at lockout by > 0.04 (4% of
            # frame height), the lifter is bridging during the press.
            if bench_check in ('butt_lift', 'both') and not bp_form_issues and bp_hip_frames:
                butt_lift = False

                lockout_hips = [hy for a, hy in bp_hip_frames if a > 140]
                press_hips   = [hy for a, hy in bp_hip_frames if a < 110]

                # ── DIAGNOSTIC PRINTS (remove after calibration) ──────────────
                print("=" * 60, flush=True)
                print(f"[BUTT-DIAG] total hip frames collected : {len(bp_hip_frames)}", flush=True)
                print(f"[BUTT-DIAG] lockout frames (angle>140) : {len(lockout_hips)}", flush=True)
                print(f"[BUTT-DIAG] press frames   (angle<110) : {len(press_hips)}", flush=True)
                print(f"[BUTT-DIAG] all hip_y values           : {[f'{hy:.3f}' for _,hy in bp_hip_frames]}", flush=True)
                if lockout_hips:
                    print(f"[BUTT-DIAG] lockout hip_y values       : {[f'{h:.3f}' for h in lockout_hips]}", flush=True)
                if press_hips:
                    print(f"[BUTT-DIAG] press hip_y values         : {[f'{h:.3f}' for h in press_hips]}", flush=True)
                print("=" * 60, flush=True)
                # ─────────────────────────────────────────────────────────────

                def _iqr_avg(vals):
                    """Mean after removing IQR outliers — filters teardown/setup frames."""
                    sv = sorted(vals)
                    q1 = sv[len(sv) // 4]
                    q3 = sv[3 * len(sv) // 4]
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    clean = [v for v in vals if lo <= v <= hi] or vals
                    return sum(clean) / len(clean)

                if len(lockout_hips) >= 2 and len(press_hips) >= 2:
                    avg_lockout_hip = _iqr_avg(lockout_hips)

                    # Filter press frames to exercise-phase only.
                    # Teardown frames (sitting up after reps) land in the press bucket
                    # with hip_y well above the lockout baseline and skew the average.
                    # Keep only frames within [lockout−0.015, lockout+0.025].
                    filtered_press = [h for h in press_hips
                                      if avg_lockout_hip - 0.015 < h < avg_lockout_hip + 0.025]
                    if not filtered_press:
                        filtered_press = press_hips  # fallback if filter too aggressive

                    avg_press_hip = sum(filtered_press) / len(filtered_press)
                    hip_rise = avg_lockout_hip - avg_press_hip
                    print(f"[BUTT-DIAG] avg_lockout_hip_y={avg_lockout_hip:.4f}  "
                          f"avg_press_hip_y={avg_press_hip:.4f}  "
                          f"press_frames_used={len(filtered_press)}/{len(press_hips)}  "
                          f"rise={hip_rise:+.4f}  threshold=-0.008", flush=True)
                    log(f"[BP-Butt] avg_lockout_hip_y={avg_lockout_hip:.3f} "
                        f"avg_press_hip_y={avg_press_hip:.3f} rise={hip_rise:.3f}")
                    if hip_rise < -0.008:
                        butt_lift = True

                print(f"[BUTT-DIAG] butt_lift={butt_lift}", flush=True)

                if butt_lift:
                    results['feedback'] = "Keep butt on bench — hips are lifting off"
                    results['form_score'] = 0
                    results['avg_form_score'] = 0
                    results['grade'] = 'F'
                    log("[BP-Butt] butt_lift=True → score forced to 0 (NO LIFT)")

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
    bench_check = sys.argv[5] if len(sys.argv) > 5 else 'both'
    
    process_video(video_path, exercise_type, output_json_path, output_video_path)
