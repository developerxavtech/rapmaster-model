import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import time
import sys
import traceback
import logging
import uuid
import subprocess
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    from exercises.engine import ExerciseEngine
    from exercises.loader import get_available_exercises
    logger.info("Exercise modules loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import exercise modules: {e}")
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video_analyses = {}

MAX_VIDEO_SIZE_MB = 200
MAX_VIDEO_DURATION_SEC = 120


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'exercises': len(get_available_exercises())})


@app.route('/api/video/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'})

    video_file = request.files['video']
    exercise_type = request.form.get('exercise_type')

    if not exercise_type:
        return jsonify({'success': False, 'error': 'No exercise type specified'})

    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    video_file.seek(0, 2)
    file_size = video_file.tell()
    video_file.seek(0)

    if file_size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        return jsonify({
            'success': False,
            'error': f'Video too large. Max {MAX_VIDEO_SIZE_MB}MB, got {file_size / (1024*1024):.1f}MB'
        })

    available = get_available_exercises()
    if exercise_type not in available:
        return jsonify({'success': False, 'error': f'Invalid exercise type. Available: {available}'})

    video_id = str(uuid.uuid4())
    filename = f"{video_id}_{video_file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(filepath)

    try:
        import cv2
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            cap.release()
            if duration > MAX_VIDEO_DURATION_SEC:
                os.remove(filepath)
                return jsonify({
                    'success': False,
                    'error': f'Video too long. Max {MAX_VIDEO_DURATION_SEC}s, got {duration:.0f}s'
                })
    except Exception as e:
        logger.warning(f"Could not check video duration: {e}")

    engine = ExerciseEngine()
    engine.set_exercise(exercise_type)

    video_analyses[video_id] = {
        'status': 'processing',
        'progress': 0,
        'filepath': filepath,
        'exercise_type': exercise_type,
        'reps': 0,
        'form_score': 100,
        'avg_form_score': 100,
        'grade': 'A',
        'state': 'READY',
        'feedback': '',
        'engine': engine,
        'processed_video': None,
    }

    thread = threading.Thread(target=_process_video, args=(video_id,), daemon=True)
    thread.start()

    return jsonify({
        'success': True,
        'video_id': video_id,
        'message': 'Video uploaded, processing started'
    })


def _process_video(video_id):
    analysis = video_analyses.get(video_id)
    if not analysis:
        return

    output_json = os.path.join(UPLOAD_FOLDER, f"{video_id}_results.json")
    output_video = os.path.join(UPLOAD_FOLDER, f"{video_id}_processed.mp4")

    try:
        cmd = [
            sys.executable,
            'video_processor.py',
            analysis['filepath'],
            analysis['exercise_type'],
            output_json,
            output_video
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            bufsize=1
        )

        while process.poll() is None:
            try:
                line = process.stdout.readline()
                if line:
                    logger.info(f"[processor] {line.strip()}")
            except Exception:
                pass

            time.sleep(0.3)

            try:
                if os.path.exists(output_json):
                    with open(output_json) as f:
                        r = json.load(f)
                    analysis.update({
                        'progress': r.get('progress', 0),
                        'reps': r.get('reps', 0),
                        'form_score': r.get('form_score', 100),
                        'avg_form_score': r.get('avg_form_score', 100),
                        'grade': r.get('grade', 'A'),
                        'state': r.get('state', 'UNKNOWN'),
                        'feedback': r.get('feedback', ''),
                    })
            except Exception:
                pass

        stdout, _ = process.communicate()
        if stdout:
            for line in stdout.strip().splitlines():
                if line.strip():
                    logger.info(f"[processor] {line.strip()}")

        if process.returncode == 0 and os.path.exists(output_json):
            with open(output_json) as f:
                r = json.load(f)

            actual_video = r.get('output_video', output_video)
            if actual_video and os.path.exists(actual_video):
                processed = actual_video
            elif os.path.exists(output_video):
                processed = output_video
            else:
                avi = output_video.rsplit('.', 1)[0] + '.avi'
                processed = avi if os.path.exists(avi) else None

            analysis.update({
                'status': 'error' if r.get('error') else r.get('status', 'completed'),
                'progress': 100,
                'reps': r.get('reps', 0),
                'form_score': r.get('form_score', 100),
                'avg_form_score': r.get('avg_form_score', 100),
                'grade': r.get('grade', 'A'),
                'state': r.get('state', 'COMPLETED'),
                'feedback': r.get('feedback', ''),
                'error': r.get('error'),
                'wrong_exercise': r.get('wrong_exercise', False),
                'wrong_exercise_message': r.get('wrong_exercise_message', ''),
                'processed_video': processed,
            })
        else:
            analysis['status'] = 'error'
            analysis['error'] = f'Processing failed (exit code {process.returncode})'

    except Exception as e:
        logger.error(f"Processing error for {video_id}: {e}")
        analysis['status'] = 'error'
        analysis['error'] = str(e)
    finally:
        try:
            if os.path.exists(output_json):
                os.remove(output_json)
            if os.path.exists(analysis['filepath']):
                os.remove(analysis['filepath'])
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


@app.route('/api/video/status/<video_id>', methods=['GET'])
def get_video_status(video_id):
    analysis = video_analyses.get(video_id)

    if not analysis:
        return jsonify({'status': 'not_found', 'error': 'Video ID not found'})

    has_video = bool(
        analysis.get('processed_video') and
        os.path.exists(analysis['processed_video'])
    )

    return jsonify({
        'status': analysis['status'],
        'progress': analysis['progress'],
        'reps': analysis['reps'],
        'form_score': analysis['form_score'],
        'avg_form_score': analysis['avg_form_score'],
        'grade': analysis['grade'],
        'state': analysis['state'],
        'feedback': analysis['feedback'],
        'error': analysis.get('error'),
        'wrong_exercise': analysis.get('wrong_exercise', False),
        'wrong_exercise_message': analysis.get('wrong_exercise_message', ''),
        'has_processed_video': has_video,
        'processed_video_url': f'/api/video/processed/{video_id}' if has_video else None,
    })


@app.route('/api/video/processed/<video_id>', methods=['GET'])
def get_processed_video(video_id):
    analysis = video_analyses.get(video_id)

    if not analysis:
        return jsonify({'error': 'Video ID not found'}), 404

    path = analysis.get('processed_video')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Processed video not ready'}), 404

    if path.endswith('.avi'):
        mime = 'video/x-msvideo'
    elif path.endswith('.webm'):
        mime = 'video/webm'
    else:
        mime = 'video/mp4'

    return send_file(path, mimetype=mime, as_attachment=False)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
