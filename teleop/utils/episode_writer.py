import os
import cv2
import json
import datetime
import numpy as np
import time
from .rerun_visualizer import RerunLogger
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Lock, Thread
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

# Queued behind the frames of the episode being closed, so the save runs once
# every earlier frame is on disk and never depends on the queue happening to be
# empty when the worker looks.
_SAVE_SENTINEL = object()

class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc = None, task_steps = None, frequency=30, image_size=[640, 480], rerun_log = True,
                 image_workers=4):
        """
        image_size: [width, height]
        image_workers: parallel JPEG encoders; 4 matches the G1 camera count.
        """
        logger_mp.info("==> EpisodeWriter initializing...")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            logger_mp.info("==> RerunLogger initializing ok.")
        
        self.item_id = -1
        self.episode_id = -1
        self.success = True
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir and not episode_dir.endswith('.zip')]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.")
        self.data_info()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self._next_progress_log = 0.0
        self._save_lock = Lock()
        self.image_pool = ThreadPoolExecutor(max_workers=image_workers)
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.")
    
    def is_ready(self):
        return self.is_available

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }

 
    def create_episode(self, episode_id: int = None):
        """
        Create a new episode.
        Args:
            episode_id (int, optional): The starting item ID for the episode. Defaults to None
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.success = True
        self.episode_id = self.episode_id + 1 if episode_id is None else episode_id
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
        self.first_item = True   # Flag to handle commas in JSON array

        # Do not build a RerunLogger here. create_episode() runs inside the 30 Hz
        # control loop, and RerunLogger() calls rr.spawn(), which blocks ~3.7 s on
        # a robot with no display. That stall freezes IK while the operator keeps
        # moving, so the arm jumps when the loop resumes. _process_item_data logs
        # through self.rerun_logger, built once in __init__.

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None):
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        next_backlog_report = 0.0
        while not self.stop_worker or not self.item_data_queue.empty():
            try:
                item_data = self.item_data_queue.get(timeout=1)
            except Empty:
                continue

            if item_data is _SAVE_SENTINEL:
                self._save_episode()
                self.item_data_queue.task_done()
                continue

            try:
                self._process_item_data(item_data)
            except Exception as e:
                logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
            self.item_data_queue.task_done()

            # A stop request cannot complete until the backlog drains, and the
            # recorder stays busy until then, so report progress instead of
            # letting the operator press the save key into silence.
            pending = self.item_data_queue.qsize()
            if self.need_save and pending > 1 and time.time() >= next_backlog_report:
                logger_mp.info(f"==> Saving episode_{self.episode_id:04d}: {pending - 1} frames left to write")
                next_backlog_report = time.time() + 1.0

    def _save_image_group(self, idx, images, out_dir, rel_dir):
        """Encode one frame's images in parallel and rewrite the dict to relative paths.

        JPEG encoding is ~90% of the per-frame cost and cv2.imwrite releases the
        GIL, so the four G1 cameras are written concurrently. Ordering across
        frames still comes from the single worker thread.
        """
        if not images:
            return
        jobs = []
        for key in list(images):
            name = f'{str(idx).zfill(6)}_{key}.jpg'
            jobs.append((key, name, os.path.join(out_dir, name), images[key]))
        written = list(self.image_pool.map(lambda job: cv2.imwrite(job[2], job[3]), jobs))
        for (key, name, path, _), ok in zip(jobs, written):
            if not ok:
                logger_mp.info(f"Failed to save image: {path}")
            images[key] = os.path.join(rel_dir, name)

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        self._save_image_group(idx, colors, self.color_dir, 'colors')
        self._save_image_group(idx, depths, self.depth_dir, 'depths')

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(item_data, ensure_ascii=False, indent=4))
            self.first_item = False

        # Throttled progress. One rich-formatted log line per frame costs real
        # time at 30 Hz and scrolls the useful messages off screen.
        now = time.time()
        if now >= self._next_progress_log:
            logger_mp.info(f"==> episode_{self.episode_id:04d}: wrote {idx + 1} frames")
            self._next_progress_log = now + 1.0

        if self.rerun_log:
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self, success=None):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.

        Args:
            success: True/False to label the episode. None keeps the current label
                     so close() can finish a pending save without overwriting it.
        """
        with self._save_lock:
            if success is not None:
                self.success = bool(success)
            if self.is_available or self.need_save:
                # No episode is open, or a save is already queued. Queueing a
                # second sentinel would close the same JSON array twice.
                return
            self.need_save = True
            pending = self.item_data_queue.qsize()
            self.item_data_queue.put(_SAVE_SENTINEL)
        logger_mp.info(f"==> Episode saved start... success={self.success}, {pending} frames queued")

    def _save_episode(self):
        """
        Save the episode data to a JSON file. Runs on the worker thread only.
        """
        with self._save_lock:
            with open(self.json_path, "a", encoding="utf-8") as f:
                f.write("\n],\n")
                f.write('"success": ' + json.dumps(bool(self.success)) + "\n}")

            if not self.success:
                failed_marker = os.path.join(self.episode_dir, "FAILED")
                with open(failed_marker, "w", encoding="utf-8") as f:
                    f.write("failed\n")
                logger_mp.info(f"==> Episode marked as failed: {self.episode_dir}")

            self.need_save = False     # Reset the save flag
            self.is_available = True   # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
        self.image_pool.shutdown(wait=True)
