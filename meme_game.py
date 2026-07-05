"""
HEX MEME CHALLENGE 🎮
====================
Motion-based meme gesture game for college exhibition.
Match the meme pose → unlock the next one → final score reveal!

Requirements: pip install opencv-python mediapipe numpy pillow
Python: 3.10+ (tested on 3.13)
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import os
import sys
import platform
import datetime
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MEMES_DIR = os.path.join(os.path.dirname(__file__), "memes")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
THRESHOLD_EASY   = 70   # % accuracy to pass (tune difficulty here)
HOLD_FRAMES      = 18   # frames user must hold pose to advance
WINDOW_NAME      = "🎭 MimicMe CHALLENGE"

# ─── EMAIL CONFIG — fill these in to let people email themselves their photos ───
# For Gmail: turn on 2-Step Verification on the sending account, then create an
# "App Password" at https://myaccount.google.com/apppasswords and paste it below
# (NOT your normal Gmail password). Any SMTP provider works if you change the host/port.
SMTP_SERVER         = "smtp.gmail.com"
SMTP_PORT           = 587
from email_config import SENDER_EMAIL, SENDER_APP_PASSWORD

# Fun slang shown when pose is NOT matched
FAIL_SLANGS = [
    "Bro seriously? 💀",
    "No cap that's WRONG",
    "It's giving… nothing 💅",
    "Main character energy: ZERO",
    "Your ancestors are crying rn",
    "Touch grass first, then try again",
    "404: Pose not found 🤖",
    "L + ratio + try harder",
    "Even my grandma does better 👵",
    "Yikes bestie… yikes",
    "That's not it chief 😬",
    "Are you even trying rn?? 😭",
]

SUCCESS_SLANGS = [
    "SLAY! 👑",
    "No cap, you cooked! 🔥",
    "W pose! GOATED!",
    "Sheeeesh! 🥶",
    "Based and accurate fr fr",
    "ATE and LEFT NO CRUMBS! ✨",
]

# ─────────────────────────────────────────────
#  MEME DEFINITIONS  (filename, gesture_type, display_name, instructions)
# ─────────────────────────────────────────────
MEMES = [
    {
        "file": "meme1_monkey_hearthands.jpg",
        "name": "Heart Hands Monkey 🩷",
        "gesture": "heart_hands",
        "instruction": "Make a HEART shape with both hands in front of your chest",
    },
    {
        "file": "meme2_monkey_shocked.jpg",
        "name": "WAIT WHAT?! 😱",
        "gesture": "hands_clasped_open_mouth",
        "instruction": "Clasp both hands in front of your chest",
    },
    {
        "file": "meme3_monkey_screaming.jpg",
        "name": "Monkey Screaming 🙉",
        "gesture": "hands_on_ears",
        "instruction": "Raise BOTH hands up to the sides of your head",
    },
    {
        "file": "meme4_monkey_fingerguns.jpg",
        "name": "Finger Heart 😎",
        "gesture": "finger_guns",
        "instruction": "Point BOTH index fingers forward (finger Heart!)",
    },
    {
        "file": "meme5_cat_sneaky.jpg",
        "name": "Sneaky Cat 🐱",
        "gesture": "arms_low_wide",
        "instruction": "Lean forward & spread both arms LOW and wide",
    },
    {
        "file": "meme6_monkey_thumbsup.jpg",
        "name": "Big Approval 👍",
        "gesture": "thumbs_up",
        "instruction": "Make a fist and give a big THUMBS UP!",
    },
]

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─────────────────────────────────────────────
#  GESTURE DETECTION FUNCTIONS
# ─────────────────────────────────────────────

def get_landmarks_normalized(results_hands, results_pose, frame_w, frame_h):
    """Extract useful landmarks from both hands and pose."""
    data = {
        "hands": [],          # list of dicts per hand: {wrist, index_tip, thumb_tip, ...}
        "pose_visible": False,
        "left_wrist":  None,
        "right_wrist": None,
        "left_elbow":  None,
        "right_elbow": None,
        "left_shoulder": None,
        "right_shoulder": None,
        "nose": None,
    }

    # ── HANDS ──
    if results_hands.multi_hand_landmarks:
        for hand_lms in results_hands.multi_hand_landmarks:
            lm = hand_lms.landmark
            hand_data = {
                "wrist":       np.array([lm[0].x,  lm[0].y]),
                "index_tip":   np.array([lm[8].x,  lm[8].y]),
                "index_mcp":   np.array([lm[5].x,  lm[5].y]),
                "middle_tip":  np.array([lm[12].x, lm[12].y]),
                "ring_tip":    np.array([lm[16].x, lm[16].y]),
                "pinky_tip":   np.array([lm[20].x, lm[20].y]),
                "thumb_tip":   np.array([lm[4].x,  lm[4].y]),
                "middle_mcp":  np.array([lm[9].x,  lm[9].y]),
            }
            # Finger extension flags
            hand_data["index_extended"]  = lm[8].y  < lm[6].y
            hand_data["middle_extended"] = lm[12].y < lm[10].y
            hand_data["ring_extended"]   = lm[16].y < lm[14].y
            hand_data["pinky_extended"]  = lm[20].y < lm[18].y
            hand_data["thumb_extended"]  = lm[4].x  < lm[3].x  # rough

            data["hands"].append(hand_data)

    # ── POSE ──
    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark
        PL = mp_pose.PoseLandmark

        def xy(idx):
            return np.array([lm[idx].x, lm[idx].y]) if lm[idx].visibility > 0.3 else None

        data["pose_visible"]     = True
        data["nose"]             = xy(PL.NOSE)
        data["left_shoulder"]    = xy(PL.LEFT_SHOULDER)
        data["right_shoulder"]   = xy(PL.RIGHT_SHOULDER)
        data["left_elbow"]       = xy(PL.LEFT_ELBOW)
        data["right_elbow"]      = xy(PL.RIGHT_ELBOW)
        data["left_wrist"]       = xy(PL.LEFT_WRIST)
        data["right_wrist"]      = xy(PL.RIGHT_WRIST)

    return data


def score_gesture(gesture_type, lm):
    """
    Returns (score 0-100, confidence_hint_string).
    Score >= THRESHOLD_EASY means the gesture matches.
    """
    hands   = lm["hands"]
    n_hands = len(hands)

    # ── helper: wrist Y position relative to shoulder ──
    def wrist_above_shoulder(wrist_key, shoulder_key, margin=0.05):
        w = lm.get(wrist_key)
        s = lm.get(shoulder_key)
        if w is None or s is None:
            return False
        return w[1] < s[1] - margin   # y is inverted (0=top)

    def wrist_near_head(wrist_key):
        """Returns True if wrist is near nose/face region."""
        w  = lm.get(wrist_key)
        ns = lm.get("nose")
        if w is None or ns is None:
            return False
        dist = np.linalg.norm(w - ns)
        return dist < 0.25

    # ════════════════════════════════════════════
    #  GESTURE 1: hand_to_face
    #  One hand raised up near face/chin
    # ════════════════════════════════════════════
    if gesture_type == "hand_to_face":
        score = 0
        ns = lm.get("nose")

        # Use the HAND-tracking wrist (from mp_hands), not the body-pose wrist.
        # The body-pose wrist can lose confidence when someone is framed close to
        # the camera (elbow cropped out of frame), even though the dedicated hand
        # tracker still sees the hand clearly.
        near_face_count = 0
        best_dist = None
        for h in hands:
            w = h["wrist"]
            if ns is not None:
                dist = np.linalg.norm(w - ns)
                at_chin_height = w[1] > ns[1] - 0.08
            else:
                # Nose not detected either (rare) — fall back to a generous check
                # so the gesture isn't unfairly blocked
                dist = 0.15
                at_chin_height = True
            if dist < 0.30 and at_chin_height:
                near_face_count += 1
                best_dist = dist if best_dist is None else min(best_dist, dist)

        if near_face_count >= 1:
            score += 55
            if best_dist is not None and best_dist < 0.18:
                score += 25

        # Penalize the classic "hands on ears" shape: two hands, near the face,
        # but far apart from each other and both raised high
        if n_hands >= 2:
            w0 = hands[0]["wrist"]
            w1 = hands[1]["wrist"]
            x_dist = abs(w0[0] - w1[0])
            both_high = (ns is not None and w0[1] < ns[1] + 0.05 and w1[1] < ns[1] + 0.05)
            if x_dist > 0.28 and both_high:
                score -= 50

        score = max(0, score)
        return min(score, 100), f"Hands near face: {'✓' if score >= THRESHOLD_EASY else '✗'}"

    # ════════════════════════════════════════════
    #  GESTURE 2: hands_clasped_open_mouth
    #  Both hands close together at chest level
    # ════════════════════════════════════════════
    elif gesture_type == "hands_clasped_open_mouth":
        score = 0
        if n_hands >= 2:
            score += 30
            # Both wrists should be close to each other
            w0 = hands[0]["wrist"]
            w1 = hands[1]["wrist"]
            dist = np.linalg.norm(w0 - w1)
            if dist < 0.18:
                score += 50
            # Both should be roughly at chest level (between shoulder & hip)
            ls = lm.get("left_shoulder")
            rs = lm.get("right_shoulder")
            if ls is not None and rs is not None:
                mid_shoulder_y = (ls[1] + rs[1]) / 2
                avg_wrist_y = (w0[1] + w1[1]) / 2
                if mid_shoulder_y < avg_wrist_y < mid_shoulder_y + 0.35:
                    score += 20
        elif n_hands == 1:
            score += 10
        return min(score, 100), f"Hands clasped: {'✓' if score >= THRESHOLD_EASY else '✗'}"

    # ════════════════════════════════════════════
    #  GESTURE 3: hands_on_ears
    #  Both hands raised high, wide apart (near ears/head)
    # ════════════════════════════════════════════
    elif gesture_type == "hands_on_ears":
        score = 0
        if n_hands >= 2:
            score += 20
            w0 = hands[0]["wrist"]
            w1 = hands[1]["wrist"]
            # Hands should be far apart horizontally
            x_dist = abs(w0[0] - w1[0])
            if x_dist > 0.3:
                score += 30
            # Both wrists should be elevated (above shoulders)
            lw_up = wrist_above_shoulder("left_wrist", "left_shoulder", margin=0.0)
            rw_up = wrist_above_shoulder("right_wrist", "right_shoulder", margin=0.0)
            if lw_up and rw_up:
                score += 30
            elif lw_up or rw_up:
                score += 15
            # Both near head
            if wrist_near_head("left_wrist") and wrist_near_head("right_wrist"):
                score += 20
        elif n_hands == 1:
            score += 5
        return min(score, 100), f"Hands on ears: {'✓' if score >= THRESHOLD_EASY else '✗'}"

    # ════════════════════════════════════════════
    #  GESTURE 4: finger_guns
    #  Both hands, index finger extended, others curled
    # ════════════════════════════════════════════
    elif gesture_type == "finger_guns":
        score = 0
        if n_hands >= 1:
            score += 10
        gun_count = 0
        for h in hands:
            # Index extended, other fingers curled
            if (h["index_extended"] and
                not h["middle_extended"] and
                not h["ring_extended"] and
                not h["pinky_extended"]):
                gun_count += 1
        if gun_count >= 2:
            score += 70
        elif gun_count == 1:
            score += 35
        # Bonus for both hands extended forward (wrists in upper region)
        both_up = (wrist_above_shoulder("left_wrist", "left_shoulder", margin=-0.1) and
                   wrist_above_shoulder("right_wrist", "right_shoulder", margin=-0.1))
        if both_up:
            score += 20
        return min(score, 100), f"Finger guns: {gun_count}/2 hands"

    # ════════════════════════════════════════════
    #  GESTURE 5: arms_low_wide
    #  Arms spread wide AND low (below shoulder)
    # ════════════════════════════════════════════
    elif gesture_type == "arms_low_wide":
        score = 0
        if n_hands >= 2:
            score += 20
            w0 = hands[0]["wrist"]
            w1 = hands[1]["wrist"]
            x_dist = abs(w0[0] - w1[0])
            if x_dist > 0.5:
                score += 40
            elif x_dist > 0.3:
                score += 20
        # Wrists should be BELOW shoulder (not raised)
        ls = lm.get("left_shoulder")
        rs = lm.get("right_shoulder")
        lw = lm.get("left_wrist")
        rw = lm.get("right_wrist")
        below_count = 0
        if ls is not None and lw is not None and lw[1] > ls[1]:
            below_count += 1
        if rs is not None and rw is not None and rw[1] > rs[1]:
            below_count += 1
        score += below_count * 20
        return min(score, 100), f"Arms low & wide: {'✓' if score >= THRESHOLD_EASY else '✗'}"

    # ════════════════════════════════════════════
    #  GESTURE 6: hands_on_cheeks
    #  Both hands raised to face level, wide apart
    # ════════════════════════════════════════════
    elif gesture_type == "hands_on_cheeks":
        score = 0
        if n_hands >= 2:
            score += 20
            w0 = hands[0]["wrist"]
            w1 = hands[1]["wrist"]
            # Should be wide apart (on each side of face)
            x_dist = abs(w0[0] - w1[0])
            if x_dist > 0.25:
                score += 25
            # Both should be near face
            near_face_l = wrist_near_head("left_wrist")
            near_face_r = wrist_near_head("right_wrist")
            if near_face_l and near_face_r:
                score += 40
            elif near_face_l or near_face_r:
                score += 20
            # Both elevated
            if (wrist_above_shoulder("left_wrist", "left_shoulder", margin=0.0) and
                wrist_above_shoulder("right_wrist", "right_shoulder", margin=0.0)):
                score += 15
        elif n_hands == 1:
            score += 5
        return min(score, 100), f"Hands on cheeks: {'✓' if score >= THRESHOLD_EASY else '✗'}"

    # ════════════════════════════════════════════
    #  GESTURE 7: thumbs_up
    #  A fist with the thumb clearly pointing upward
    # ════════════════════════════════════════════
    elif gesture_type == "thumbs_up":
        best_score = 0
        for h in hands:
            s = 0
            fingers_curled = (not h["index_extended"] and not h["middle_extended"]
                              and not h["ring_extended"] and not h["pinky_extended"])
            thumb_up = h["thumb_tip"][1] < h["wrist"][1] - 0.05
            if fingers_curled:
                s += 40
            if thumb_up:
                s += 40
            if fingers_curled and thumb_up:
                s += 20   # clean combo bonus
            best_score = max(best_score, s)
        return min(best_score, 100), f"Thumbs up: {'✓' if best_score >= THRESHOLD_EASY else '✗'}"

    # ════════════════════════════════════════════
    #  GESTURE 8: praying_hands
    #  Both palms pressed together, fingertips pointing up, near chest/face
    # ════════════════════════════════════════════
    elif gesture_type == "praying_hands":
        score = 0
        if n_hands >= 2:
            w0 = hands[0]["wrist"]
            w1 = hands[1]["wrist"]
            dist = np.linalg.norm(w0 - w1)
            if dist < 0.15:
                score += 45
            elif dist < 0.25:
                score += 25

            idx_up0 = hands[0]["index_tip"][1] < w0[1] - 0.03
            idx_up1 = hands[1]["index_tip"][1] < w1[1] - 0.03
            if idx_up0 and idx_up1:
                score += 25

            ns = lm.get("nose")
            avg_wrist_y = (w0[1] + w1[1]) / 2
            if ns is not None:
                if avg_wrist_y < ns[1] + 0.35:
                    score += 30
            else:
                score += 20
        elif n_hands == 1:
            score += 10
        return min(score, 100), f"Palms together: {'✓' if score >= THRESHOLD_EASY else '✗'}"

    # ════════════════════════════════════════════
    #  GESTURE 9: heart_hands
    #  Index fingertips touching (top of heart) + thumb tips touching (bottom of heart)
    # ════════════════════════════════════════════
    elif gesture_type == "heart_hands":
        score = 0
        if n_hands >= 2:
            idx0, idx1 = hands[0]["index_tip"], hands[1]["index_tip"]
            th0, th1   = hands[0]["thumb_tip"], hands[1]["thumb_tip"]
            idx_dist   = np.linalg.norm(idx0 - idx1)
            thumb_dist = np.linalg.norm(th0 - th1)

            if idx_dist < 0.08:
                score += 35
            elif idx_dist < 0.15:
                score += 20

            if thumb_dist < 0.10:
                score += 35
            elif thumb_dist < 0.18:
                score += 20

            w0, w1 = hands[0]["wrist"], hands[1]["wrist"]
            if np.linalg.norm(w0 - w1) < 0.30:
                score += 30
        elif n_hands == 1:
            score += 10
        return min(score, 100), f"Heart hands: {'✓' if score >= THRESHOLD_EASY else '✗'}"

    return 0, "Unknown gesture"


# ─────────────────────────────────────────────
#  DRAWING / OVERLAY HELPERS
# ─────────────────────────────────────────────

def load_meme_image(filename, target_h=240):
    """Load meme image and resize to fit panel."""
    path = os.path.join(MEMES_DIR, filename)
    img  = cv2.imread(path)
    if img is None:
        # Placeholder
        img = np.ones((target_h, target_h, 3), dtype=np.uint8) * 40
    h, w = img.shape[:2]
    scale = target_h / h
    img   = cv2.resize(img, (int(w * scale), target_h))
    return img


def draw_rounded_rect(img, pt1, pt2, color, radius=20, thickness=-1):
    """Draw a filled rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0,   0, 90, color, thickness)


def draw_progress_bar(frame, x, y, w, h, percent, color_fg=(0,200,100), color_bg=(50,50,50)):
    """Draw a horizontal progress bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bg, -1)
    filled = int(w * percent / 100)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), color_fg, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 1)


def put_text_with_shadow(frame, text, pos, font=cv2.FONT_HERSHEY_DUPLEX,
                          scale=0.8, color=(255,255,255), shadow=(0,0,0), thickness=2):
    """Draw text with a drop shadow."""
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2), font, scale, shadow, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),     font, scale, color,  thickness,     cv2.LINE_AA)


def fit_text(frame, text, x, y, max_w, font=cv2.FONT_HERSHEY_DUPLEX,
             base_scale=0.8, color=(255,255,255), thickness=2):
    """Shrink text to fit within max_w pixels."""
    scale = base_scale
    while scale > 0.3:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_w:
            break
        scale -= 0.05
    put_text_with_shadow(frame, text, (x, y), font, scale, color, thickness=thickness)


def draw_star(img, center, size, color, rotation_deg=-90):
    """Draw a small filled 5-point star sticker."""
    cx, cy = center
    pts = []
    for i in range(10):
        angle = math.radians(rotation_deg + i * 36)
        r = size if i % 2 == 0 else size * 0.42
        pts.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(img, pts, color, lineType=cv2.LINE_AA)


def draw_heart(img, center, size, color):
    """Draw a small filled heart sticker using the classic parametric heart curve."""
    cx, cy = center
    pts = []
    for i in range(40):
        t = math.pi * 2 * i / 40
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        pts.append([int(cx + x * size / 16), int(cy + y * size / 16)])
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(img, pts, color, lineType=cv2.LINE_AA)


def draw_sparkle(img, center, size, color):
    """Draw a tiny 4-point sparkle/diamond twinkle."""
    cx, cy = center
    pts = np.array([[
        [cx, cy - size], [cx + int(size*0.28), cy], [cx, cy + size], [cx - int(size*0.28), cy]
    ]], dtype=np.int32)
    cv2.fillPoly(img, pts, color, lineType=cv2.LINE_AA)
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)


def scatter_cute_stickers(img, count, w, h, avoid_rects=None, seed=7):
    """Sprinkle small pastel stars/hearts/sparkles across the background, avoiding photo areas."""
    avoid_rects = avoid_rects or []
    rng = random.Random(seed)
    palette = [(255, 200, 230), (255, 235, 160), (200, 230, 255), (220, 200, 255)]
    kinds = ["star", "heart", "sparkle"]
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 8:
        attempts += 1
        x = rng.randint(10, w - 10)
        y = rng.randint(10, h - 10)
        inside_photo = any(x1 <= x <= x2 and y1 <= y <= y2 for (x1, y1, x2, y2) in avoid_rects)
        if inside_photo:
            continue
        kind = rng.choice(kinds)
        color = rng.choice(palette)
        size = rng.randint(5, 10)
        if kind == "star":
            draw_star(img, (x, y), size, color)
        elif kind == "heart":
            draw_heart(img, (x, y), size, color)
        else:
            draw_sparkle(img, (x, y), size, color)
        placed += 1


def draw_scalloped_edge(img, w, h, hole_r=9, gap=26, color=(15, 8, 25)):
    """Draw punch-hole notches along the left & right edges, like a real photobooth strip."""
    y = gap
    while y < h:
        cv2.circle(img, (0, y), hole_r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (w, y), hole_r, color, -1, cv2.LINE_AA)
        y += gap * 2


def build_photobooth_strip(meme_list, user_photos):
    """
    Build a cute photobooth-style results image: one row per completed meme,
    showing the reference meme photo next to the user's matching pose photo,
    decorated with stickers and a scalloped strip edge.
    Returns the finished image (numpy array, BGR).
    """
    ROW_H = 200
    THUMB_W = 260
    GAP = 24
    LABEL_H = 34
    PAD = 30
    HEADER_H = 90

    n = len(meme_list)
    strip_w = PAD * 2 + THUMB_W * 2 + GAP
    strip_h = HEADER_H + n * (ROW_H + LABEL_H + 18) + PAD

    strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    # Soft pastel purple-pink gradient background
    for i in range(strip_h):
        t = i / max(1, strip_h)
        strip[i, :] = (int(60 + 40 * t), int(20 + 20 * t), int(45 + 35 * t))

    # Scatter cute stickers in the background, avoiding the photo thumbnails themselves
    photo_rects = []
    y_probe = HEADER_H
    for _ in range(n):
        photo_rects.append((PAD - 10, y_probe - 10, PAD + THUMB_W + 10, y_probe + ROW_H + 10))
        photo_rects.append((PAD + THUMB_W + GAP - 10, y_probe - 10, PAD + THUMB_W*2 + GAP + 10, y_probe + ROW_H + 10))
        y_probe += ROW_H + LABEL_H + 18
    scatter_cute_stickers(strip, count=n * 6 + 14, w=strip_w, h=strip_h, avoid_rects=photo_rects)

    draw_star(strip, (PAD - 4, 30), 10, (255, 215, 100))
    draw_heart(strip, (strip_w - PAD - 10, 28), 9, (255, 160, 200))
    put_text_with_shadow(strip, "MimicMe CHALLENGE",
                         (PAD + 22, 45), font=cv2.FONT_HERSHEY_DUPLEX,
                         scale=1.0, color=(255, 205, 235), thickness=2)
    put_text_with_shadow(strip, "meme  vs  you", (PAD, 75),
                         scale=0.65, color=(220, 200, 230), thickness=1)
    draw_heart(strip, (PAD + 155, 71), 6, (255, 170, 205))

    y = HEADER_H
    for idx, (meme, user_frame) in enumerate(zip(meme_list, user_photos)):
        # Reference meme thumbnail (left) — Polaroid-style white border
        meme_thumb = load_meme_image(meme["file"], target_h=ROW_H)
        mh, mw = meme_thumb.shape[:2]
        mw2 = min(mw, THUMB_W)
        lx = PAD
        cv2.rectangle(strip, (lx-6, y-6), (lx+THUMB_W+6, y+ROW_H+6), (250, 245, 250), -1)
        strip[y:y+ROW_H, lx:lx+mw2] = meme_thumb[:ROW_H, :mw2]
        draw_rounded_rect(strip, (lx-6, y-6), (lx+THUMB_W+6, y+ROW_H+6),
                          (255, 190, 220), radius=8, thickness=3)

        # User's captured photo (right) — Polaroid-style white border
        uh, uw = user_frame.shape[:2]
        uscale = ROW_H / uh
        user_resized = cv2.resize(user_frame, (int(uw * uscale), ROW_H))
        uw2 = min(user_resized.shape[1], THUMB_W)
        rx = PAD + THUMB_W + GAP
        cv2.rectangle(strip, (rx-6, y-6), (rx+THUMB_W+6, y+ROW_H+6), (250, 245, 250), -1)
        strip[y:y+ROW_H, rx:rx+uw2] = user_resized[:, :uw2]
        draw_rounded_rect(strip, (rx-6, y-6), (rx+THUMB_W+6, y+ROW_H+6),
                          (180, 230, 190), radius=8, thickness=3)

        # Little heart divider between the pair
        draw_heart(strip, (lx + THUMB_W + GAP // 2, y + ROW_H // 2), 11, (255, 170, 205))

        # Cute number badge (star-shaped) instead of a plain rectangle
        draw_star(strip, (lx + 16, y + 16), 15, (255, 205, 90))
        cv2.putText(strip, str(idx+1), (lx + 11, y + 21),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 40, 10), 1, cv2.LINE_AA)

        # Label below the row
        put_text_with_shadow(strip, f"#{idx+1}  {meme['name']}",
                             (PAD, y + ROW_H + 26),
                             scale=0.7, color=(220, 220, 220), thickness=2)
        y += ROW_H + LABEL_H + 18

    draw_scalloped_edge(strip, strip_w, strip_h)
    return strip


def save_photobooth_strip(strip):
    """Save the strip to RESULTS_DIR with a timestamped filename. Returns the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hex_meme_results_{timestamp}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(filepath, strip)
    return filepath


def send_email_with_photo(to_email, filepath):
    """
    Emails the photobooth strip image as an attachment.
    Returns (success: bool, message: str).
    """
    if SENDER_APP_PASSWORD == "PASTE_YOUR_16_CHAR_APP_PASSWORD_HERE":
        return False, "Email not configured yet (edit email_config.py with your App Password)"

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email.mime.text import MIMEText
    from email import encoders

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        msg["Subject"] = "Your MimicMe Challenge Photos! 🎭"
        msg.attach(MIMEText("Here are your pose-matching results from the MimicMe Challenge. Thanks for playing. We hope you enjoyed this! Vote for us <33"))

        with open(filepath, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(filepath)}"')
        msg.attach(part)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        return True, f"Sent to {to_email}!"
    except Exception as e:
        return False, f"Failed to send: {e}"


def show_results_window(strip, filepath):
    """
    Displays the photobooth strip in a proper app window with an inline
    email field + Send button, so people can email themselves their photos.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    max_h = 700
    if pil_img.height > max_h:
        scale = max_h / pil_img.height
        pil_img = pil_img.resize((int(pil_img.width * scale), max_h))

    root = tk.Tk()
    root.title("📸 MimicMe Challenge — Your Results")
    root.configure(bg="#141020")

    tk_img = ImageTk.PhotoImage(pil_img)
    img_label = tk.Label(root, image=tk_img, bg="#141020")
    img_label.image = tk_img  # keep a reference so it isn't garbage-collected
    img_label.pack(padx=16, pady=16)

    form = tk.Frame(root, bg="#141020")
    form.pack(pady=(0, 8))

    tk.Label(form, text="Email your photos to:", font=("Segoe UI", 11),
             fg="#e8e8ff", bg="#141020").pack(side=tk.LEFT, padx=(0, 8))

    email_var = tk.StringVar()
    email_entry = tk.Entry(form, textvariable=email_var, width=32, font=("Segoe UI", 11))
    email_entry.pack(side=tk.LEFT, padx=(0, 8))
    email_entry.focus_set()

    status_var = tk.StringVar(value="")

    def on_send():
        email = email_var.get().strip()
        if "@" not in email or "." not in email.split("@")[-1]:
            status_var.set("⚠️  Please enter a valid email address")
            return
        status_var.set("Sending…")
        root.update_idletasks()
        success, message = send_email_with_photo(email, filepath)
        status_var.set(("✅  " if success else "❌  ") + message)

    send_btn = tk.Button(form, text="Send", command=on_send, bg="#33aa77", fg="white",
                         font=("Segoe UI", 11, "bold"), relief="flat", padx=16)
    send_btn.pack(side=tk.LEFT)

    # Allow pressing Enter in the email field to trigger Send too
    email_entry.bind("<Return>", lambda e: on_send())

    status_label = tk.Label(root, textvariable=status_var, font=("Segoe UI", 10),
                            fg="#a8ffcf", bg="#141020")
    status_label.pack(pady=(0, 6))

    tk.Button(root, text="Close", command=root.destroy, bg="#333", fg="white",
             relief="flat", padx=16).pack(pady=(0, 14))

    root.mainloop()


def show_final_screen(user_photo, meme_images, captured_photos=None):
    """Show the epic final results screen side-by-side, and save a downloadable photobooth strip."""
    captured_photos = captured_photos or []

    # Build & save the photobooth-style comparison strip (meme vs. your photo, per meme)
    saved_path = None
    strip = None
    if captured_photos:
        strip = build_photobooth_strip(meme_images, captured_photos)
        saved_path = save_photobooth_strip(strip)
    # Generate a fun random score
    base_score = random.randint(6, 10)
    decimal    = random.choice([0, 0, 0.5])
    score      = min(10, base_score + decimal)
    score_str  = str(int(score)) if decimal == 0 else f"{base_score}.5"

    score_comments = {
        10:   "GOAT STATUS 🐐  Undeniable W!",
        9:    "Elite performer! Touch grass? Never! 🌿",
        8:    "Slaying but there's ROOM to slay harder 👑",
        7:    "Mid? No… well… kinda. But still W!",
        6:    "We respect the grind tho ngl 💪",
    }
    comment_key = int(score)
    if comment_key < 6:
        comment_key = 6
    comment = score_comments.get(comment_key, "You tried bestie 💀")

    PANEL_H = 720
    PANEL_W = 1280
    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    # Background gradient
    for i in range(PANEL_H):
        t = i / PANEL_H
        r = int(15 + 10 * t)
        g = int(5  + 5  * t)
        b = int(40 + 30 * t)
        canvas[i, :] = (b, g, r)

    # Title
    title = "🎭  MimicMe CHALLENGE  RESULTS  🎭"
    put_text_with_shadow(canvas, "MimicMe CHALLENGE  -  FINAL RESULTS",
                         (PANEL_W // 2 - 380, 55),
                         font=cv2.FONT_HERSHEY_DUPLEX,
                         scale=1.2, color=(255, 220, 50), thickness=3)

    # ── User photo (left) ──
    photo_target_h = 360
    ph, pw = user_photo.shape[:2]
    scale  = photo_target_h / ph
    user_resized = cv2.resize(user_photo, (int(pw * scale), photo_target_h))
    uw, uh_actual = user_resized.shape[1], user_resized.shape[0]
    ux, uy = 40, 100
    # Border
    draw_rounded_rect(canvas, (ux-6, uy-6), (ux+uw+6, uy+uh_actual+6),
                      (255, 200, 50), radius=12)
    canvas[uy:uy+uh_actual, ux:ux+uw] = user_resized
    put_text_with_shadow(canvas, "YOU (the legend)", (ux + 10, uy + uh_actual + 30),
                         scale=0.85, color=(255, 200, 50))

    # ── Meme collage (right) ──
    COLLAGE_X = int(PANEL_W * 0.38)
    meme_thumbs = [load_meme_image(m["file"], target_h=165) for m in memes_completed]
    n = len(meme_thumbs)
    cols = 3
    rows = (n + cols - 1) // cols
    THUMB_W = 185
    THUMB_H = 165
    for idx, thumb in enumerate(meme_thumbs):
        col = idx % cols
        row = idx // cols
        tx = COLLAGE_X + col * (THUMB_W + 10)
        ty = 100 + row * (THUMB_H + 10)
        th, tw = thumb.shape[:2]
        th2 = min(th, THUMB_H)
        tw2 = min(tw, THUMB_W)
        canvas[ty:ty+th2, tx:tx+tw2] = thumb[:th2, :tw2]
        # number badge
        draw_rounded_rect(canvas, (tx+2, ty+2), (tx+28, ty+28), (255,100,0), radius=5)
        put_text_with_shadow(canvas, str(idx+1), (tx+8, ty+22), scale=0.65, thickness=2)

    collage_label = "MEMES YOU CRUSHED 💀"
    put_text_with_shadow(canvas, collage_label,
                         (COLLAGE_X, 90),
                         scale=0.9, color=(180,220,255), thickness=2)

    # ── Score box ──
    sx, sy = 40, 510
    draw_rounded_rect(canvas, (sx, sy), (sx+360, sy+160), (30,15,70), radius=20)
    draw_rounded_rect(canvas, (sx, sy), (sx+360, sy+160), (255,200,50), radius=20, thickness=3)
    score_color = (50, 255, 100) if score >= 8 else (50, 200, 255) if score >= 6 else (100,100,255)
    put_text_with_shadow(canvas, f"SCORE: {score_str} / 10",
                         (sx + 30, sy + 65),
                         font=cv2.FONT_HERSHEY_DUPLEX,
                         scale=1.3, color=score_color, thickness=3)
    fit_text(canvas, comment, sx + 20, sy + 120, 330,
             base_scale=0.7, color=(220,220,220), thickness=2)

    # ── Instruction ──
    put_text_with_shadow(canvas, "Press  Q  or  ESC  to exit",
                         (COLLAGE_X, PANEL_H - 30),
                         scale=0.75, color=(160,160,160), thickness=1)

    if saved_path:
        put_text_with_shadow(canvas, "Press Q to see your photos + email them →",
                             (40, PANEL_H - 30),
                             scale=0.65, color=(120, 255, 180), thickness=1)

    # Show loop
    cv2.namedWindow("🏆 FINAL SCORE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("🏆 FINAL SCORE", PANEL_W, PANEL_H)
    while True:
        cv2.imshow("🏆 FINAL SCORE", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
    cv2.destroyWindow("🏆 FINAL SCORE")

    if strip is not None:
        show_results_window(strip, saved_path)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  MAIN GAME LOOP
# ─────────────────────────────────────────────

def main():
    global memes_completed
    memes_completed = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open webcam. Check camera permissions.")
        return

    # Auto-detect camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📷  Camera: {frame_w}×{frame_h}")
    print("🎭  MimicMe CHALLENGE started!")
    print("     Match the pose shown on screen to advance.")
    print("     Press Q or ESC to quit at any time.\n")

    # MediaPipe models
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    )
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    )

    current_meme_idx = 0
    hold_counter     = 0
    last_slang       = random.choice(FAIL_SLANGS)
    slang_timer      = 0
    success_flash    = 0   # frames for green flash
    last_score       = 0

    # For final photo
    best_user_photo  = None
    last_good_frame  = None
    captured_photos  = []   # one user photo captured per completed meme, in order

    # Pre-load meme images (small panel)
    meme_panels = {m["file"]: load_meme_image(m["file"], target_h=220) for m in MEMES}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, frame_w, frame_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌  Camera frame lost.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        r_hands = hands_detector.process(rgb)
        r_pose  = pose_detector.process(rgb)

        rgb.flags.writeable = True

        # ── Landmark extraction ──
        lm = get_landmarks_normalized(r_hands, r_pose, frame_w, frame_h)

        # ── Current meme ──
        meme = MEMES[current_meme_idx]
        score, hint = score_gesture(meme["gesture"], lm)
        last_score = score

        passed = score >= THRESHOLD_EASY

        if passed:
            hold_counter += 1
            last_good_frame = frame.copy()
        else:
            hold_counter = max(0, hold_counter - 3)   # gentle decay, not a hard reset
            slang_timer += 1
            if slang_timer % 45 == 0:   # change slang every ~1.5s
                last_slang = random.choice(FAIL_SLANGS)

        # ── Advance to next meme ──
        if hold_counter >= HOLD_FRAMES:
            memes_completed.append(meme)
            captured_photos.append(last_good_frame.copy() if last_good_frame is not None else frame.copy())
            success_flash = 30

            if best_user_photo is None or random.random() < 0.5:
                best_user_photo = last_good_frame.copy() if last_good_frame is not None else frame.copy()

            current_meme_idx += 1
            hold_counter = 0

            if current_meme_idx >= len(MEMES):
                # All done!
                cap.release()
                hands_detector.close()
                pose_detector.close()
                cv2.destroyAllWindows()
                if best_user_photo is None:
                    best_user_photo = frame.copy()
                show_final_screen(best_user_photo, memes_completed, captured_photos)
                return

        # ════════════════════════════════════════
        #  DRAW UI OVERLAY
        # ════════════════════════════════════════

        overlay = frame.copy()

        # Semi-transparent top bar
        draw_rounded_rect(overlay, (0, 0), (frame_w, 70), (10, 5, 30), radius=0)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # ── Title bar ──
        put_text_with_shadow(frame,
                             f"HEX MEME CHALLENGE  |  #{current_meme_idx+1}/{len(MEMES)}  {meme['name']}",
                             (15, 42),
                             font=cv2.FONT_HERSHEY_DUPLEX,
                             scale=0.9, color=(255, 220, 50), thickness=2)

        # ── Meme panel (top right) ──
        panel = meme_panels.get(meme["file"])
        if panel is not None:
            ph, pw = panel.shape[:2]
            px = frame_w - pw - 15
            py = 80
            # Background box
            draw_rounded_rect(frame, (px - 8, py - 8), (px + pw + 8, py + ph + 8),
                              (20, 10, 60), radius=12)
            draw_rounded_rect(frame, (px - 8, py - 8), (px + pw + 8, py + ph + 8),
                              (200, 150, 50), radius=12, thickness=2)
            frame[py:py+ph, px:px+pw] = panel
            put_text_with_shadow(frame, "MATCH THIS →",
                                 (px - 130, py + ph // 2),
                                 scale=0.75, color=(255, 200, 50), thickness=2)

        # ── Instruction text ──
        inst_y = frame_h - 120
        draw_rounded_rect(frame, (10, inst_y - 10), (frame_w - 10, inst_y + 45),
                          (10, 5, 40), radius=10)
        fit_text(frame, f"👉  {meme['instruction']}",
                 20, inst_y + 28, frame_w - 40,
                 base_scale=0.9, color=(180, 230, 255), thickness=2)

        # ── Progress bar ──
        bar_y = frame_h - 65
        put_text_with_shadow(frame, "ACCURACY:", (15, bar_y - 5), scale=0.7,
                             color=(200,200,200), thickness=1)
        bar_color = (50,220,100) if passed else (255, 100, 50)
        draw_progress_bar(frame, 130, bar_y - 22, frame_w - 300, 26,
                          score, color_fg=bar_color)
        put_text_with_shadow(frame, f"{int(score)}%",
                             (frame_w - 160, bar_y - 5),
                             scale=0.85, color=bar_color, thickness=2)

        # ── Hold progress ──
        if passed and hold_counter > 0:
            hold_pct = int(hold_counter / HOLD_FRAMES * 100)
            draw_progress_bar(frame, 130, bar_y + 12, frame_w - 300, 16,
                              hold_pct, color_fg=(50, 255, 200))
            put_text_with_shadow(frame, f"HOLD! {hold_pct}%",
                                 (frame_w - 160, bar_y + 24),
                                 scale=0.7, color=(50,255,200), thickness=2)

        # ── Slang / feedback ──
        if not passed:
            slang_y = frame_h // 2 - 20
            (tw, th_), _ = cv2.getTextSize(last_slang, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
            sx = max(10, (frame_w - tw) // 2)
            # Translucent pill background
            pill_pad = 18
            draw_rounded_rect(frame,
                              (sx - pill_pad, slang_y - th_ - pill_pad),
                              (sx + tw + pill_pad, slang_y + pill_pad),
                              (60, 10, 10), radius=18)
            draw_rounded_rect(frame,
                              (sx - pill_pad, slang_y - th_ - pill_pad),
                              (sx + tw + pill_pad, slang_y + pill_pad),
                              (200, 50, 50), radius=18, thickness=2)
            put_text_with_shadow(frame, last_slang,
                                 (sx, slang_y),
                                 font=cv2.FONT_HERSHEY_DUPLEX,
                                 scale=1.1, color=(255, 130, 130), thickness=3)

        if success_flash > 0:
            green_overlay = frame.copy()
            green_overlay[:] = (30, 200, 30)
            alpha_s = 0.25 * (success_flash / 30)
            cv2.addWeighted(green_overlay, alpha_s, frame, 1 - alpha_s, 0, frame)
            suc_text = random.choice(SUCCESS_SLANGS) if success_flash == 30 else SUCCESS_SLANGS[0]
            put_text_with_shadow(frame, "✓  NEXT MEME UNLOCKED!",
                                 (frame_w // 2 - 220, frame_h // 2),
                                 font=cv2.FONT_HERSHEY_DUPLEX,
                                 scale=1.4, color=(100, 255, 100), thickness=3)
            success_flash -= 1

        # ── Pose/hand landmarks (optional subtle overlay) ──
        if r_hands.multi_hand_landmarks:
            for hl in r_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # ── Footer hint ──
        put_text_with_shadow(frame, "Q / ESC = quit",
                             (15, frame_h - 12),
                             scale=0.55, color=(100,100,100), thickness=1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    hands_detector.close()
    pose_detector.close()
    cv2.destroyAllWindows()
    print("\n✌️  Thanks for playing HEX MEME CHALLENGE!")


if __name__ == "__main__":
    memes_completed = []
    main()